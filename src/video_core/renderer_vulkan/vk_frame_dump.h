// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <vector>

#include "common/common_types.h"
#include "video_core/surface.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class ImageView;
class MemoryAllocator;
class Sampler;
class Scheduler;

/// EDEN_NX2_FRAME_DUMP=<dir> reads back one whole frame: the color targets of every pass, the
/// images written by every compute dispatch and the presented image. EDEN_NX2_FRAME_DUMP_AT picks
/// the frame, in seconds after boot ("45") or as a frame number ("2700f").
/// EDEN_NX2_FRAME_DUMP_SAMPLED=1 also reads back every texture a pass samples (3D and depth
/// included), once per pass, as the first draw sampling it sees it, and the textures each compute
/// shader samples, once per shader, before its first dispatch, with each texture's sampler state,
/// and the uniform and storage buffers of every draw and dispatch, once per shader and binding, with
/// each dispatch's group counts. Readbacks are encoded and written on a worker thread so the
/// capture does not stall the game.
class FrameDump {
public:
    struct Sampled {
        u32 stage;
        u32 tic;
        ImageView* view;
        const Sampler* sampler;
    };

    /// A uniform or storage buffer binding. Its bytes come from guest memory, or from the host
    /// buffer when the GPU wrote the range and guest memory may be stale.
    struct BoundBuffer {
        u32 stage;
        bool storage;
        u32 index;
        bool written;
        GPUVAddr gpu_addr;
        u64 device_addr;
        u32 bound_size;
        u32 size;
        bool gpu_modified;
        VkBuffer host;
        u32 host_offset;
        std::vector<u8> guest;
        /// EDEN_NX2_FRAME_DUMP_HOST: also read the host buffer the draw reads ("host_copy").
        bool also_host{};
    };

    explicit FrameDump(const Device& device, MemoryAllocator& memory_allocator,
                       Scheduler& scheduler);
    ~FrameDump();

    [[nodiscard]] bool Capturing() const noexcept {
        return capturing;
    }

    [[nodiscard]] bool CapturingSampled() const noexcept {
        return capturing && sampled;
    }

    /// The debug server's capture_frame, mid-game and without EDEN_NX2_FRAME_DUMP at boot
    /// (2026-11-10: the owner found stretched terrain on screen, and a relaunch would lose it).
    /// Captures the next frame into `dir`; `buffers` records every draw and dispatch's uniform
    /// and storage buffers (as EDEN_NX2_FRAME_DUMP_SAMPLED does), `host` also the host buffer's
    /// copy of guest-sourced ones (as EDEN_NX2_FRAME_DUMP_HOST). FinishCapture(id) is called once
    /// every file is written. Returns false while another capture runs.
    bool RequestNow(u64 id, const std::filesystem::path& dir, bool buffers, bool host);

    /// The running capture also reads the host copies of guest-sourced buffers.
    [[nodiscard]] bool HostCopies() const noexcept {
        return capturing && host_copies;
    }

    /// Reads back the textures of the current draw not yet seen in this pass. Returns true when
    /// copies were recorded, which ends the current render pass.
    bool NoteSampled(std::span<const Sampled> textures);
    /// Reads back the textures of the dispatch about to run not yet seen for this shader.
    void NoteDispatchSampled(u64 shader_hash, std::span<const Sampled> textures);
    /// Records the buffers of the current draw (shader hashes by program) or of the dispatch about
    /// to run not yet seen for their shader. Returns true when copies were recorded.
    bool NoteBuffers(bool compute, std::span<const u64> shader_hashes,
                     std::span<BoundBuffer> buffers);
    /// Records the group counts of the dispatch about to run.
    void NoteLaunch(u64 shader_hash, const std::array<u32, 3>& grid,
                    const std::array<u32, 3>& block, std::optional<GPUVAddr> indirect,
                    std::optional<std::array<u32, 3>> indirect_grid);

    /// Starts a pass when the framebuffer changes and captures the previous one.
    /// Returns true when copies were recorded, which ends the current render pass.
    bool BeginPass(const void* framebuffer, std::span<ImageView* const> colors);
    void NoteDraw(std::span<const u64> shader_hashes);
    /// Starts counting the primitives and fragments of the draw about to be recorded; returns the
    /// query to end after it. Sampled captures only.
    std::optional<u32> BeginProbe();
    struct Probe {
        u32 draw;
        bool mesh;
        bool indirect;
        u32 first;
        u32 elements;
        u32 instances;
        u32 alpha_test; ///< Comparison op while the alpha test is enabled, else 0
        f32 alpha_ref;
        bool alpha_to_coverage;      ///< As the guest set it
        bool host_alpha_to_coverage; ///< As last given to the host
        bool color0;                 ///< The fragment program writes colour 0
        bool kill;                   ///< The fragment program can discard
        u32 zero_handles;            ///< Texture handles whose handle program gave 0
    };
    /// Ends the query; the draw index comes from BeginProbe.
    void EndProbe(u32 query, Probe probe);
    /// Records a draw or dispatch dropped because its program or pipeline is unavailable.
    void NoteSkipped(bool compute, u64 program, GPUVAddr rt0, std::string_view reason);
    /// Captures the current pass; returns true when copies were recorded.
    bool EndPass();
    void CaptureDispatch(u64 shader_hash, std::span<ImageView* const> written);
    void CapturePresent(ImageView& view);
    /// Waits for pending readbacks and writes them once they pass the staging budget.
    void DrainIfFull();
    void TickFrame();

private:
    enum class DepthEncoding { None, Unorm16, Unorm24, Float32 };

    struct Target {
        u32 slot;
        VkImage image;
        VideoCore::Surface::PixelFormat format;
        u32 width;
        u32 height;
        u32 depth;
        s32 level;
        s32 layer;
        GPUVAddr gpu_addr;
        bool rescaled;
        bool msaa;
        DepthEncoding depth_encoding;
    };

    /// An image readback, or a buffer one (`raw`) of `bytes` bytes.
    struct Readback {
        vk::Buffer buffer;
        Target target;
        std::string name;
        std::string tags;
        bool raw{};
        size_t bytes{};
    };

    /// Work for the writer thread: an image readback (`name` set), buffer bytes (`raw`), a
    /// finished manifest line, or the summary.
    struct Job {
        Target target{};
        std::string name;
        std::string tags;
        std::vector<u8> data;
        bool raw{};
        bool summary{};
        bool truncated{};
    };

    std::optional<Target> MakeTarget(ImageView& view, u32 slot) const;
    static u32 TexelBytes(const Target& target);
    static size_t TargetBytes(const Target& target);
    static int Decode(const Target& target, const u8* pixel, std::array<float, 4>& out);
    bool Record(const Target& target, std::string name, std::string tags);
    bool RecordSampled(const Sampled& texture, u32 index, std::string name, std::string tags);
    bool RecordBuffer(VkBuffer source, u32 offset, u32 size, std::string tags);
    void Drain();
    void Enqueue(Job job);
    void WriterLoop();
    void Write(const Job& job);
    void WriteBuffer(const Job& job);
    void WriteFile(const std::string& name, std::span<const u8> data);
    void Rearm();
    /// Ends the writer (every queued file written) and clears the capture's state.
    void ResetAfterCapture();

    const Device& device;
    MemoryAllocator& memory_allocator;
    Scheduler& scheduler;

    bool enabled{};
    bool sampled{};
    /// EDEN_NX2_FRAME_DUMP_SAMPLED_FROM: passes before this one do not read back sampled textures.
    u32 sampled_from_pass{};
    /// EDEN_NX2_FRAME_DUMP_CAP_MIB: read-back budget for one capture.
    size_t size_cap{};
    bool capturing{};
    bool done{};
    bool truncated{};
    std::filesystem::path directory;
    std::filesystem::path trigger_file;
    std::filesystem::path base_directory;
    u32 captures{};
    u64 trigger_frame{~0ULL};
    /// EDEN_NX2_FRAME_DUMP_PAIR: each file-triggered capture is followed by one of the next frame.
    bool pair{};
    bool pair_next{};
    /// RequestNow: capture at the next frame end, then FinishCapture(request_id).
    bool request_now{};
    u64 request_id{};
    bool host_copies{};
    double trigger_seconds{};
    std::chrono::steady_clock::time_point start;
    u64 frame{};

    std::ofstream manifest;
    static constexpr u32 MAX_PROBES = 8192;
    vk::QueryPool probe_pool;
    std::vector<Probe> probes;
    std::vector<Readback> readbacks;
    size_t pending_bytes{};
    size_t recorded_bytes{};
    size_t written_bytes{};
    u32 sequence{};
    u32 images{};

    const void* pass_framebuffer{};
    std::vector<Target> pass_targets;
    std::vector<u64> pass_hashes;
    std::vector<std::pair<GPUVAddr, u32>> pass_sampled;
    std::vector<std::pair<u32, u32>> pass_null_sampled; ///< Stage and TIC of views with no image
    /// (compute shader hash, image address, format/level/layer) already read back this frame.
    std::vector<std::tuple<u64, GPUVAddr, u32>> dispatch_sampled;
    /// (address, format, level, layer) of block-compressed textures already read back; the GPU
    /// never writes them, so one copy per frame suffices.
    std::vector<std::tuple<GPUVAddr, u32, s32, s32>> recorded_blocks;
    /// (shader hash, storage, binding index, device address, size) already recorded this frame.
    std::vector<std::tuple<u64, bool, u32, u64, u32>> seen_buffers;
    /// Content hashes of the buffer files written; writer thread only.
    std::vector<u64> written_buffers;
    u32 pass_index{};
    u32 pass_first_draw{};
    u32 draws{};
    u32 dispatches{};

    std::thread writer;
    std::mutex queue_mutex;
    std::condition_variable queue_ready;
    std::deque<Job> queue;
    bool stop_writer{};
};

} // namespace Vulkan
