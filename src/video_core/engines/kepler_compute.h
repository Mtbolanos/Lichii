// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <optional>
#include <span>
#include <unordered_set>
#include <vector>
#include "common/bit_field.h"
#include "common/common_funcs.h"
#include "common/common_types.h"
#include "video_core/engines/engine_interface.h"
#include "video_core/engines/engine_upload.h"
#include "video_core/fire_emblem_upscale.h"
#include "video_core/game_dlss_status.h"
#include "video_core/macro.h"
#include "video_core/native_qmd.h"
#include "video_core/split_write_tracker.h"
#include "video_core/textures/texture.h"

namespace Core {
class System;
}

namespace VideoCommon {
class GpuSplit;
}
namespace VideoCommon::QmdPrediction {
struct Record;
}

namespace Tegra {
class MemoryManager;
}

namespace VideoCore {
class RasterizerInterface;
}

namespace Tegra::Engines {

/// EXP13=2-4: bytes another engine's front knows (a 3D inline upload) land at GPU [address,
/// address + size) of `memory` when the back runs the work that ends at ring `position`, written
/// with write-stamp tag `tag`: the QMD copy's inline QMD overlays and upload shadow blocks that
/// still hold those bytes take them, as compute inline uploads patch them.
void Nx2PatchQmdCopy(VideoCommon::GpuSplit& split, const MemoryManager& memory, GPUVAddr address,
                     const u8* data, u64 size, u64 position, u64 tag);

/// Draws left to trace after an upscale fallback (debug aid).
extern std::atomic<u32> g_nx2_trace_draws_after_upscale;

/**
 * This Engine is known as GK104_Compute. Documentation can be found in:
 * https://github.com/envytools/envytools/blob/master/rnndb/graph/gk104_compute.xml
 * https://cgit.freedesktop.org/mesa/mesa/tree/src/gallium/drivers/nouveau/nvc0/nve4_compute.xml.h
 */

#define KEPLER_COMPUTE_REG_INDEX(field_name)                                                       \
    (offsetof(Tegra::Engines::KeplerCompute::Regs, field_name) / sizeof(u32))

#define LAUNCH_REG_INDEX(field_name)                                                               \
    (offsetof(Tegra::Engines::KeplerCompute::LaunchParams, field_name) / sizeof(u32))

class KeplerCompute final : public EngineInterface {
public:
    explicit KeplerCompute(MemoryManager& memory_manager);
    ~KeplerCompute();

    /// Binds a rasterizer to this engine.
    void BindRasterizer(VideoCore::RasterizerInterface* rasterizer);

    /// Whether a write to `method` can write memory (a dispatch, upload or report): with the
    /// native class, every method besides those that only store their value; macros always.
    [[nodiscard]] bool MayWriteMemory(u32 method) const {
        if (method >= native_plain_methods.size()) {
            return true;
        }
        return native_mode ? !native_plain_methods[method] : execution_mask[method];
    }

    /// The channel's MME data RAM, shared with its 3D engine.
    void SetNativeDataRam(std::array<u32, 0xc00>* data_ram) { native_data_ram = data_ram; }
    void SetNativeMacroMode(MacroEngine* shared_macro, std::span<u32> positions, NativeMacroContext* context = nullptr) {
        if (native_context != context) native_compute_uploaded = false;
        native_context = context;
        native_mode = shared_macro != nullptr;
        upload_state.SetNativePartialTransfers(native_mode);
        native_macro = shared_macro;
        native_macro_positions = positions;
        if (native_mode) execution_mask.set();
        else {
            execution_mask.reset();
            execution_mask[KEPLER_COMPUTE_REG_INDEX(exec_upload)] = true;
            execution_mask[KEPLER_COMPUTE_REG_INDEX(data_upload)] = true;
            execution_mask[KEPLER_COMPUTE_REG_INDEX(launch)] = true;
        }
    }

    static constexpr std::size_t NumConstBuffers = 8;

    struct Regs {
        static constexpr std::size_t NUM_REGS = 0xCF8;

        union {
            struct {
                INSERT_PADDING_WORDS_NOINIT(0x60);

                Upload::Registers upload;

                struct {
                    union {
                        BitField<0, 1, u32> linear;
                    };
                } exec_upload;

                u32 data_upload;

                INSERT_PADDING_WORDS_NOINIT(0x3F);

                struct {
                    u32 address;
                    GPUVAddr Address() const {
                        return GPUVAddr{address} << 8;
                    }
                } launch_desc_loc;

                INSERT_PADDING_WORDS_NOINIT(0x1);

                u32 launch;

                INSERT_PADDING_WORDS_NOINIT(0x4A7);

                struct {
                    u32 address_high;
                    u32 address_low;
                    u32 limit;
                    GPUVAddr Address() const {
                        return (GPUVAddr{address_high} << 32) | GPUVAddr{address_low};
                    }
                } tsc;

                INSERT_PADDING_WORDS_NOINIT(0x3);

                struct {
                    u32 address_high;
                    u32 address_low;
                    u32 limit;
                    GPUVAddr Address() const {
                        return (GPUVAddr{address_high} << 32) | GPUVAddr{address_low};
                    }
                } tic;

                INSERT_PADDING_WORDS_NOINIT(0x22);

                struct {
                    u32 address_high;
                    u32 address_low;
                    GPUVAddr Address() const {
                        return (GPUVAddr{address_high} << 32) | GPUVAddr{address_low};
                    }
                } code_loc;

                INSERT_PADDING_WORDS_NOINIT(0x3FE);

                u32 tex_cb_index;

                INSERT_PADDING_WORDS_NOINIT(0x374);
            };
            std::array<u32, NUM_REGS> reg_array;
        };
    } regs{};

    struct LaunchParams {
        static constexpr std::size_t NUM_LAUNCH_PARAMETERS = 0x40;

        INSERT_PADDING_WORDS(0x8);

        u32 program_start;

        INSERT_PADDING_WORDS(0x2);

        BitField<30, 1, u32> linked_tsc;

        BitField<0, 31, u32> grid_dim_x;
        union {
            BitField<0, 16, u32> grid_dim_y;
            BitField<16, 16, u32> grid_dim_z;
        };

        INSERT_PADDING_WORDS(0x3);

        BitField<0, 18, u32> shared_alloc;

        BitField<16, 16, u32> block_dim_x;
        union {
            BitField<0, 16, u32> block_dim_y;
            BitField<16, 16, u32> block_dim_z;
        };

        union {
            BitField<0, 8, u32> const_buffer_enable_mask;
            BitField<29, 2, u32> cache_layout;
        };

        INSERT_PADDING_WORDS(0x8);

        struct ConstBufferConfig {
            u32 address_low;
            union {
                BitField<0, 8, u32> address_high;
                BitField<15, 17, u32> size;
            };
            GPUVAddr Address() const {
                return (GPUVAddr{address_high.Value()} << 32) | GPUVAddr{address_low};
            }
        };
        std::array<ConstBufferConfig, NumConstBuffers> const_buffer_config;

        union {
            BitField<0, 20, u32> local_pos_alloc;
            BitField<27, 5, u32> barrier_alloc;
        };

        union {
            BitField<0, 20, u32> local_neg_alloc;
            BitField<24, 5, u32> gpr_alloc;
        };

        union {
            BitField<0, 20, u32> local_crs_alloc;
            BitField<24, 5, u32> sass_version;
        };

        INSERT_PADDING_WORDS(0x10);
    } launch_description{};

    struct {
        u32 write_offset = 0;
        u32 copy_size = 0;
        std::vector<u8> inner_buffer;
    } state{};

    static_assert(sizeof(Regs) == Regs::NUM_REGS * sizeof(u32),
                  "KeplerCompute Regs has wrong size");

    static_assert(sizeof(LaunchParams) == LaunchParams::NUM_LAUNCH_PARAMETERS * sizeof(u32),
                  "KeplerCompute LaunchParams has wrong size");

    /// Write the value to the register identified by method.
    void CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) override;

    /// Write multiple values to the register identified by method.
    void CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) override;

    std::optional<GPUVAddr> GetIndirectComputeAddress() const {
        return indirect_compute;
    }

    bool IsNativeMode() const { return native_mode; }
    bool ProcessGameDlss(u64 program_id, u64 native_key);
    /// Ocarina of Time with anti-aliasing at No sharpening (oot_network.h): true when this
    /// dispatch of the game's network is skipped, or its output pass replaced by a copy of the
    /// tonemapped image.
    bool BypassOotNetwork(u64 program_id, u64 native_key);
    void TraceDlssContract(u64 native_key, bool warn_unvalidated = true);

    /// GPU thread split with macros on its front half (gpu_split.h, EDEN_NX2_SPLIT=2), called by
    /// the front instead of CallMethod for native compute engines: runs their macros and macro
    /// uploads against a copy of the registers and sends the register writes to the back.
    bool FrontRunsMacros() const { return native_mode; }
    void FrontCallMethod(Core::System& system, VideoCommon::GpuSplit& split, u32 method,
                         u32 argument, bool is_last_call, const GPUVAddr* segment);
    void FrontCallMultiMethod(Core::System& system, VideoCommon::GpuSplit& split, u32 method,
                              const u32* base_start, u32 amount, u32 methods_pending,
                              const GPUVAddr* segment);

private:
    struct FrontState;
    FrontState& Front(VideoCommon::GpuSplit& split);
    void FrontApply(FrontState& state, u32 method, u32 argument);
    MacroEngine& FrontMacroEngine(FrontState& state);
    std::span<u32> FrontMacroPositions(FrontState& state);
    void FrontRunMacro(Core::System& system, VideoCommon::GpuSplit& split, FrontState& state);
    /// Tells the split a semaphore release just sent writes a known payload.
    void FrontNoteRelease(VideoCommon::GpuSplit& split, const FrontState& state, u32 control) const;
    bool FrontPlainWrite(VideoCommon::GpuSplit& split, FrontState& state, u32 method,
                         u32 argument);
    void FrontVerify(VideoCommon::GpuSplit& split, FrontState& state);
    /// What a method sent from the front may write on the back (GpuSplit::Effect).
    VideoCommon::SplitWriteTracker::Effect FrontEffect(u32 method) const;
    /// After sending it: the memory it writes, for the front's reads.
    void FrontNoteWrites(VideoCommon::GpuSplit& split, FrontState& state, u32 method);
    /// After sending it: what it may write as compute work, for the front's macro memory reads
    /// (GpuSplit::NoteWriterRange, NoteUnboundedWriter, NoteDispatchWriter).
    void FrontNoteWriter(VideoCommon::GpuSplit& split, FrontState& state, u32 method, u32 argument);
    /// EXP13: predicts the releases of the QMD a launch just sent signals (nx2_qmd_prediction.h).
    void FrontPredictQmd(VideoCommon::GpuSplit& split, const FrontState& state, u32 method,
                         u32 argument) const;
    /// EXP13=2: feeds the front's QMD scheduler copy the signal just sent (0xaf, 0xb0, 0x107).
    void FrontMirrorQmd(VideoCommon::GpuSplit& split, const FrontState& state, u32 method,
                        u32 argument) const;
    /// EXP13=2: runs the copy's scheduler for a signal of the QMD at `qmd_address`.
    void FrontQmdSignal(VideoCommon::GpuSplit& split, GPUVAddr qmd_address, u32 argument) const;
    /// EXP13=2: the QMD at `address` as the back will read it; `unknown` when the copy can't
    /// tell (it resyncs then).
    bool FrontReadQmd(VideoCommon::GpuSplit& split, u64 address,
                      NativeQmdScheduler::Descriptor& descriptor, bool& unknown,
                      VideoCommon::QmdPrediction::Record& info) const;
    /// An exec_upload with `exec` was sent: the range its data writes.
    void FrontNoteUpload(FrontState& state, u32 exec);
    /// EXP13=2: data_upload words just sent: the inline QMD overlays they fall in take them.
    void FrontPatchOverlays(VideoCommon::GpuSplit& split, FrontState& state, const u32* words,
                            u32 count, u64 position) const;
    /// EXP13=2: the data_upload words batched since (FrontState::upload_batch) to the copy.
    void FrontFlushUploads(VideoCommon::GpuSplit& split, FrontState& state) const;
    std::unique_ptr<FrontState> front;
    bool front_macros{}; ///< The front owns the macro code and bindings (the back skips them)

    bool DispatchNativeCompute(const NativeQmdScheduler::Descriptor& descriptor);
    void ProcessNativeReportSemaphore(u32 argument);
    void DescribeNativeResources(const NativeQmdScheduler::Descriptor& qmd);
    /// Texture descriptor that the handle at `offset` in constant buffer 0 selects.
    std::optional<Texture::TICEntry> ConstBuffer0Image(u32 offset) const;
    void RememberDlssMotionVectors();
    bool TryFireEmblemUpscale(bool allow_history = true);
    void CaptureFireEmblemJitter();
    bool FinishDlssEvaluation();
    void CheckDlssNetwork();
    [[nodiscard]] VideoCore::GameDlssReason GameDlssBlocker() const;
    void UpdateGameDlssStatus(bool complete, VideoCore::GameDlssReason blocker);
    std::pair<std::vector<u32>, std::vector<u32>> NativeImageHandles(GPUVAddr program);
    void RememberNativeUpscaleInput();

    /// This frame's DLSS evaluation (Fortune's Weave): whether its passes all ran, and whether
    /// the game's own network output is shown instead of the replacement upscale.
    struct DlssChain {
        bool active{};
        bool spatial_only{};
        bool output_pass{};
        u32 rejected{};
        u32 clean_evaluations{};
        bool native{};
        /// Health checks of the network's output against its input (CheckDlssNetwork): whether
        /// this backend has them, good and bad results in a row, and why the network failed them
        /// (None while it hasn't; then the replacement upscale draws the output for the session).
        bool checked{};
        u32 good_checks{};
        u32 bad_checks{};
        VideoCore::GameDlssReason failure{};
        /// Programs of evaluations that reached their output pass; they are skipped only after
        /// the network fails its health checks.
        std::unordered_set<u64> programs;
        /// This evaluation's programs and dispatches so far. Its programs join `programs` only
        /// at its output pass, so compute work after a look-alike first pass keeps running.
        std::unordered_set<u64> pending;
        u32 dispatches{};
        u32 evaluations{};
        /// What the Graphics tab was last told.
        VideoCore::GameDlssStatus status{};
    } dlss_chain;

    /// Ocarina of Time's movie network: a pass that reads one colour image and stores none
    /// retains its input here; the pass that stores an image upscaling it is replaced by a plain
    /// copy (the network's matrix passes cost tens of milliseconds per frame). Cleared after the
    /// copy.
    std::optional<Texture::TICEntry> native_upscale_input;
    /// Native dispatches left to skip while a retained network is being replaced (bounded, so a
    /// non-network pass cannot silence the engine forever).
    u32 network_skip_remaining{};
    /// Set when the replacement copy fails, so the full network runs from then on. Settings
    /// oot_movie_upscale chooses the network itself (OotMovieNetworkChosen in the .cpp).
    bool native_upscale_disabled{};

    /// Motion vectors and depth of Fire Emblem: Fortune's Weave's DLSS evaluation, from its first
    /// pass (run earlier in the frame than the network pass the stand-in replaces), with the input
    /// colour image and camera jitter they go with.
    std::optional<Texture::TICEntry> dlss_motion_vectors;
    std::optional<Texture::TICEntry> dlss_depth;
    u32 dlss_contract_trace_count{};
    GPUVAddr dlss_motion_color{};
    VideoCommon::FireEmblemUpscale::FrameJitter fe_upscale_jitter;
    bool ProcessNativeMethod(Core::System& system, u32 method, u32 argument, bool is_last_call);
    NativeQmdScheduler native_qmd;
    std::array<u32, 0xc00>* native_data_ram{};
    bool native_mme_dumped{};
    std::optional<u32> native_executing_macro_logged;
    bool native_mode{};
    NativeMacroContext* native_context{};
    bool native_compute_uploaded{};
    MacroEngine* native_macro{};
    std::array<u32, 0x1000> native_registers{};
    /// Native methods that only store their value (no macro, QMD, upload, semaphore or code
    /// side effect): CallMethod writes those directly.
    std::bitset<0xe00> native_plain_methods;
    std::span<u32> native_macro_positions;
    std::vector<u32> native_parameters;
    std::vector<u32> native_memo_parameters; ///< The current call's parameters, for the memo
    std::optional<u32> native_executing_macro;
    u32 native_bind_pointer{};

    void ProcessLaunch();

    void ConsumeSinkImpl(Core::System& system) override;
    void ApplyPairs(Core::System& system, const u64* pairs, u32 count) override;

    /// Retrieves information about a specific TIC entry from the TIC buffer.
    Texture::TICEntry GetTICEntry(u32 tic_index) const;

    /// Retrieves information about a specific TSC entry from the TSC buffer.
    Texture::TSCEntry GetTSCEntry(u32 tsc_index) const;

    MemoryManager& memory_manager;
    VideoCore::RasterizerInterface* rasterizer = nullptr;
    Upload::State upload_state;
    GPUVAddr upload_address;
    bool upload_dirty{};

    struct UploadInfo {
        GPUVAddr upload_address;
        GPUVAddr exec_address;
        u32 copy_size;
        bool was_dirty;
    };
    std::vector<UploadInfo> uploads;
    std::optional<GPUVAddr> indirect_compute{};
};

#define ASSERT_REG_POSITION(field_name, position)                                                  \
    static_assert(offsetof(KeplerCompute::Regs, field_name) == position * 4,                       \
                  "Field " #field_name " has invalid position")

#define ASSERT_LAUNCH_PARAM_POSITION(field_name, position)                                         \
    static_assert(offsetof(KeplerCompute::LaunchParams, field_name) == position * 4,               \
                  "Field " #field_name " has invalid position")

ASSERT_REG_POSITION(upload, 0x60);
ASSERT_REG_POSITION(exec_upload, 0x6C);
ASSERT_REG_POSITION(data_upload, 0x6D);
ASSERT_REG_POSITION(launch, 0xAF);
ASSERT_REG_POSITION(tsc, 0x557);
ASSERT_REG_POSITION(tic, 0x55D);
ASSERT_REG_POSITION(code_loc, 0x582);
ASSERT_REG_POSITION(tex_cb_index, 0x982);
ASSERT_LAUNCH_PARAM_POSITION(program_start, 0x8);
ASSERT_LAUNCH_PARAM_POSITION(grid_dim_x, 0xC);
ASSERT_LAUNCH_PARAM_POSITION(shared_alloc, 0x11);
ASSERT_LAUNCH_PARAM_POSITION(block_dim_x, 0x12);
ASSERT_LAUNCH_PARAM_POSITION(const_buffer_enable_mask, 0x14);
ASSERT_LAUNCH_PARAM_POSITION(const_buffer_config, 0x1D);

#undef ASSERT_REG_POSITION

} // namespace Tegra::Engines
