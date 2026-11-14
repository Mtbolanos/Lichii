// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <thread>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "common/scope_exit.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "tests/core/system_test_context.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/rasterizer_interface.h"

namespace {

using Tegra::Engines::Maxwell3D;
namespace FastDraw = VideoCommon::Nx2FastDraw;
constexpr u32 ViewportScaleX = MAXWELL3D_REG_INDEX(viewport_transform);

class NoDma final : public Tegra::Engines::AccelerateDMAInterface {
public:
    bool BufferCopy(GPUVAddr, GPUVAddr, u64) override { return true; }
    bool BufferClear(GPUVAddr, u64, u32) override { return true; }
    bool ImageToBuffer(const Tegra::DMA::ImageCopy&, const Tegra::DMA::ImageOperand&,
                       const Tegra::DMA::BufferOperand&) override { return true; }
    bool BufferToImage(const Tegra::DMA::ImageCopy&, const Tegra::DMA::BufferOperand&,
                       const Tegra::DMA::ImageOperand&) override { return true; }
};

/// Records the fast draw serial each draw sees.
class SerialRasterizer final : public VideoCore::RasterizerInterface {
public:
    std::vector<u64> draws;

    void Draw(bool, u32) override { draws.push_back(FastDraw::Serial()); }
    void DrawTexture() override {}
    void Clear(u32) override {}
    void DispatchCompute() override {}
    void ResetCounter(VideoCommon::QueryType) override {}
    void Query(GPUVAddr, VideoCommon::QueryType, VideoCommon::QueryPropertiesFlags, u32,
               u32) override {}
    void BindGraphicsUniformBuffer(size_t, u32, GPUVAddr, u32) override {}
    void DisableGraphicsUniformBuffer(size_t, u32) override {}
    void SignalFence(std::function<void()>&& func) override { func(); }
    void SyncOperation(std::function<void()>&& func) override { func(); }
    void SignalSyncPoint(u32) override {}
    void SignalReference() override {}
    void ReleaseFences(bool) override {}
    void FlushAll() override {}
    void FlushRegion(DAddr, u64, VideoCommon::CacheType) override {}
    bool MustFlushRegion(DAddr, u64, VideoCommon::CacheType) override { return false; }
    VideoCore::RasterizerDownloadArea GetFlushArea(DAddr addr, u64 size) override {
        return {addr, addr + size, false};
    }
    void InvalidateRegion(DAddr, u64, VideoCommon::CacheType) override {}
    void OnCacheInvalidation(PAddr, u64) override {}
    bool OnCPUWrite(PAddr, u64) override { return false; }
    void InvalidateGPUCache() override {}
    void UnmapMemory(DAddr, u64) override {}
    void ModifyGPUMemory(size_t, GPUVAddr, u64) override {}
    void FlushAndInvalidateRegion(DAddr, u64, VideoCommon::CacheType) override {}
    void WaitForIdle() override {}
    void FragmentBarrier() override {}
    void TiledCacheBarrier() override {}
    void FlushCommands() override {}
    void TickFrame() override {}
    Tegra::Engines::AccelerateDMAInterface& AccessAccelerateDMA() override { return dma; }
    void AccelerateInlineToMemory(GPUVAddr, size_t, std::span<const u8>) override {}

private:
    NoDma dma;
};

struct FastDrawFixture {
    FastDrawFixture() : memory{system, device_memory, 32, 1ULL << 31}, maxwell3d{memory} {
        SetSwitch(true);
        FastDraw::g_delta = {};
        maxwell3d.BindRasterizer(&rasterizer);
    }
    ~FastDrawFixture() {
        FastDraw::SetEnabledForTest(previous);
        FastDraw::g_delta = {};
    }

    static void SetSwitch(bool on) {
        FastDraw::SetEnabledForTest(on);
    }

    u32 Reg(u32 method) const {
        return maxwell3d.regs.reg_array[method];
    }

    void Write(u32 method, u32 value) {
        maxwell3d.CallMethod(system, method, value, true);
    }

    void DrawArrays(u32 first, u32 count) {
        Write(MAXWELL3D_REG_INDEX(vertex_buffer.first), first);
        Write(MAXWELL3D_REG_INDEX(vertex_buffer.count), count);
        Write(MAXWELL3D_REG_INDEX(draw.begin), 0x4); // first instance, triangles
        Write(MAXWELL3D_REG_INDEX(draw.end), 0);
    }

    /// Whether the last two draws saw the same serial (the second could reuse the first).
    bool Reusable() const {
        const auto& draws = rasterizer.draws;
        REQUIRE(draws.size() >= 2);
        return draws[draws.size() - 1] == draws[draws.size() - 2];
    }

    const bool previous = FastDraw::Enabled();
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device_memory{physical};
    Tegra::MemoryManager memory;
    SerialRasterizer rasterizer;
    Maxwell3D maxwell3d;
};

} // namespace

TEST_CASE("Fast draw keeps the serial across draws that change only draw parameters",
          "[fast_draw]") {
    FastDrawFixture f;
    f.DrawArrays(0, 3);
    f.DrawArrays(3, 6);
    REQUIRE(f.rasterizer.draws.size() == 2);
    CHECK(f.Reusable());

    f.Write(ViewportScaleX, f.Reg(ViewportScaleX));
    f.DrawArrays(9, 3);
    CHECK(f.Reusable());

    f.Write(MAXWELL3D_REG_INDEX(const_buffer.offset), 0x40);
    f.Write(MAXWELL3D_REG_INDEX(const_buffer.address_low), 0x1000);
    f.DrawArrays(12, 3);
    CHECK(f.Reusable());
}

TEST_CASE("Fast draw breaks reuse on anything that changes what a draw binds", "[fast_draw]") {
    FastDrawFixture f;
    const auto breaks = [&](auto&& change) {
        f.DrawArrays(0, 3);
        change();
        f.DrawArrays(0, 3);
        return !f.Reusable();
    };
    constexpr u32 blend = MAXWELL3D_REG_INDEX(blend.enable[0]);
    CHECK(breaks([&] { f.Write(blend, f.Reg(blend) ^ 1); }));
    CHECK(breaks([&] {
        f.Write(ViewportScaleX,
                f.Reg(ViewportScaleX) + 1);
    }));
    CHECK(breaks([&] { f.Write(MAXWELL3D_REG_INDEX(clear_surface), 0x3c); }));
    CHECK(breaks([&] { f.Write(MAXWELL3D_REG_INDEX(tiled_cache_barrier), 0); }));
    CHECK(breaks([&] { FastDraw::Break(FastDraw::Reason::Fence); }));
}

TEST_CASE("Fast draw serial never moves while disabled", "[fast_draw]") {
    FastDrawFixture f;
    FastDrawFixture::SetSwitch(false);
    f.DrawArrays(0, 3);
    f.Write(MAXWELL3D_REG_INDEX(blend.enable[0]), f.Reg(MAXWELL3D_REG_INDEX(blend.enable[0])) ^ 1);
    f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), 1);
    f.DrawArrays(0, 3);
    CHECK(f.Reusable());
}

namespace {

struct DeltaFixture : FastDrawFixture {
    void SetConstantBuffer(GPUVAddr address, u32 size, u32 offset) {
        Write(MAXWELL3D_REG_INDEX(const_buffer.size), size);
        Write(MAXWELL3D_REG_INDEX(const_buffer.address_high), static_cast<u32>(address >> 32));
        Write(MAXWELL3D_REG_INDEX(const_buffer.address_low), static_cast<u32>(address));
        Write(MAXWELL3D_REG_INDEX(const_buffer.offset), offset);
    }
};

} // namespace

TEST_CASE("Fast draw records bind and constant data changes, keeps index buffer changes light",
          "[fast_draw]") {
    DeltaFixture f;
    f.DrawArrays(0, 3);
    const u64 serial = FastDraw::Serial();
    FastDraw::g_delta = {};

    f.Write(MAXWELL3D_REG_INDEX(index_buffer.start_addr_low),
            f.Reg(MAXWELL3D_REG_INDEX(index_buffer.start_addr_low)) + 0x100);
    f.Write(MAXWELL3D_REG_INDEX(index_buffer.format), f.Reg(MAXWELL3D_REG_INDEX(index_buffer.format)) ^ 1);
    CHECK(FastDraw::Serial() == serial);
    CHECK(FastDraw::g_delta.Empty());

    f.SetConstantBuffer(0x10000, 0x100, 0);
    f.Write(MAXWELL3D_REG_INDEX(bind_groups[4].raw_config), 0x31); // slot 3, valid
    CHECK(FastDraw::g_delta.binds[4] == 1U << 3);
    FastDraw::g_delta.binds[4] = 0;
    f.Write(MAXWELL3D_REG_INDEX(bind_groups[4].raw_config), 0x31);
    CHECK(FastDraw::g_delta.binds[4] == 0);

    f.SetConstantBuffer(0x20000, 0x100, 0x10);
    f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), 1);
    f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1, 2);
    REQUIRE(FastDraw::g_delta.num_writes == 1);
    CHECK(FastDraw::g_delta.write_begin[0] == 0x20010);
    CHECK(FastDraw::g_delta.write_end[0] == 0x20018);

    CHECK(FastDraw::Serial() == serial);
    f.DrawArrays(0, 3);
    CHECK(f.Reusable());
    CHECK_FALSE(FastDraw::g_delta.Empty());
}

TEST_CASE("Fast draw takes the delta's ranges in use and clears it", "[fast_draw]") {
    FastDraw::g_delta = {};
    FastDraw::NoteBind(2, 5);
    FastDraw::NoteConstantData(0x1000, 0x1010);
    FastDraw::NoteConstantData(0x3000, 0x3008);
    FastDraw::NoteDeviceData(0x9000, 0x9010);
    FastDraw::Delta out;
    out.write_begin.fill(0xdead);
    FastDraw::TakeDelta(out);
    CHECK(out.binds[2] == 1U << 5);
    REQUIRE(out.num_writes == 2);
    CHECK(out.write_begin[0] == 0x1000);
    CHECK(out.write_end[1] == 0x3008);
    REQUIRE(out.num_device_writes == 1);
    CHECK(out.device_end[0] == 0x9010);
    CHECK_FALSE(out.no_uniform_keeps);
    CHECK(FastDraw::g_delta.Empty());
    CHECK(FastDraw::g_delta.num_device_writes == 0);
    // Another take keeps nothing of the first one.
    FastDraw::TakeDelta(out);
    CHECK(out.Empty());
    CHECK(out.num_device_writes == 0);
}

TEST_CASE("Fast draw breaks on state and texture headers", "[fast_draw]") {
    DeltaFixture f;
    f.Write(MAXWELL3D_REG_INDEX(tex_header.address_high), 0);
    f.Write(MAXWELL3D_REG_INDEX(tex_header.address_low), 0x40000);
    f.Write(MAXWELL3D_REG_INDEX(tex_header.limit), 7);
    f.Write(MAXWELL3D_REG_INDEX(tex_sampler.address_high), 0);
    f.Write(MAXWELL3D_REG_INDEX(tex_sampler.address_low), 0x50000);
    f.Write(MAXWELL3D_REG_INDEX(tex_sampler.limit), 7);
    const auto breaks = [&](auto&& change) {
        FastDraw::g_delta = {};
        const u64 serial = FastDraw::Serial();
        change();
        return FastDraw::Serial() != serial;
    };
    constexpr u32 blend = MAXWELL3D_REG_INDEX(blend.enable[0]);
    CHECK(breaks([&] { f.Write(blend, f.Reg(blend) ^ 1); }));
    CHECK(breaks([&] {
        f.SetConstantBuffer(0x40000, 0x1000, 0xe0); // the last texture header
        f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), 1);
    }));
    CHECK(breaks([&] {
        f.SetConstantBuffer(0x50000, 0x1000, 0);
        f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), 1);
    }));
    CHECK_FALSE(breaks([&] {
        f.SetConstantBuffer(0x40000, 0x1000, 0x100); // just past the headers
        f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), 1);
    }));
    // More separate writes than the delta holds widen its ranges instead of ending draw reuse.
    CHECK_FALSE(breaks([&] {
        f.SetConstantBuffer(0x60000, 0x1000, 0);
        for (u32 i = 0; i <= FastDraw::Delta::MaxWrites; ++i) {
            f.Write(MAXWELL3D_REG_INDEX(const_buffer.offset), i * 0x20);
            f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), i);
        }
    }));
    const auto& delta = FastDraw::g_delta;
    REQUIRE(delta.num_writes == FastDraw::Delta::MaxWrites);
    CHECK(delta.no_uniform_keeps);
    for (u32 i = 0; i <= FastDraw::Delta::MaxWrites; ++i) {
        const GPUVAddr address = 0x60000 + i * 0x20;
        bool covered = false;
        for (u32 write = 0; write < delta.num_writes; ++write) {
            covered |= delta.write_begin[write] <= address && address + 4 <= delta.write_end[write];
        }
        CHECK(covered);
    }
}

TEST_CASE("Fast draw breaks on texture header data written through another mapping",
          "[fast_draw]") {
    DeltaFixture f;
    f.memory.BindRasterizer(&f.rasterizer);
    constexpr GPUVAddr headers = 0x40000;
    constexpr GPUVAddr alias = 0x90000;
    f.memory.Map(headers, 0x100000, 0x1000, Tegra::PTEKind::PITCH, false);
    f.memory.Map(alias, 0x100000, 0x1000, Tegra::PTEKind::PITCH, false);
    f.Write(MAXWELL3D_REG_INDEX(tex_header.address_high), 0);
    f.Write(MAXWELL3D_REG_INDEX(tex_header.address_low), static_cast<u32>(headers));
    f.Write(MAXWELL3D_REG_INDEX(tex_header.limit), 7);
    const auto breaks = [&](u32 offset) {
        FastDraw::g_delta = {};
        const u64 serial = FastDraw::Serial();
        f.SetConstantBuffer(alias, 0x1000, offset);
        f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), 1);
        return FastDraw::Serial() != serial;
    };
    CHECK(breaks(0xe0));        // the last texture header
    CHECK_FALSE(breaks(0x100)); // just past the headers
}

TEST_CASE("Fast draw finds texture handle data written through another mapping",
          "[fast_draw]") {
    DeltaFixture f;
    f.memory.BindRasterizer(&f.rasterizer);
    constexpr GPUVAddr handles = 0x100000; // a buffer of texture handles, 0x100 bytes
    constexpr GPUVAddr alias = 0x300000;   // its device memory, mapped again
    constexpr GPUVAddr other = 0x500000;   // other device memory
    constexpr GPUVAddr unmapped = 0x700000;
    constexpr auto Kind = Tegra::PTEKind::PITCH;
    f.memory.Map(handles, 0x200000, 0x1000, Kind, false);
    f.memory.Map(alias, 0x200000, 0x1000, Kind, false);
    f.memory.Map(other, 0x400000, 0x1000, Kind, false);
    FastDraw::DeviceSpans spans;
    const auto reaches = [&](GPUVAddr handle_buffer, GPUVAddr buffer, u32 offset,
                             std::vector<u32> words) {
        FastDraw::g_delta = {};
        f.SetConstantBuffer(buffer, 0x2000, offset);
        f.maxwell3d.CallMultiMethod(f.system, MAXWELL3D_REG_INDEX(const_buffer.buffer),
                                    words.data(), static_cast<u32>(words.size()),
                                    static_cast<u32>(words.size()));
        spans.Update(f.memory, handle_buffer, 0x100);
        return FastDraw::g_delta.Reaches(handle_buffer, 0x100, spans);
    };
    CHECK(reaches(handles, handles, 0x10, {1}));
    CHECK(reaches(handles, alias, 0x10, {1})); // missed when compared by GPU address only
    CHECK(reaches(handles, alias, 0xfc, {1, 2}));
    CHECK_FALSE(reaches(handles, alias, 0x100, {1})); // just past the handles
    CHECK_FALSE(reaches(handles, other, 0x10, {1}));
    // Nothing mapped there: compared by GPU address, as before.
    CHECK(reaches(unmapped, unmapped, 0x10, {1}));
    CHECK_FALSE(reaches(unmapped, other, 0x10, {1}));

    // A write across two pages that map device memory apart lands on both.
    f.memory.Map(alias, 0x201000, 0x1000, Kind, false);
    f.memory.Map(alias + 0x1000, 0x200000, 0x1000, Kind, false);
    CHECK(reaches(handles, alias, 0xffc, {1, 2}));
    CHECK(FastDraw::g_delta.num_device_writes == 2);
    CHECK_FALSE(reaches(handles, alias, 0xff8, {1, 2}));

    // Mapping changes translate the handle buffer again.
    CHECK_FALSE(reaches(handles, other, 0x10, {1}));
    f.memory.Map(handles, 0x400000, 0x1000, Kind, false);
    CHECK(reaches(handles, other, 0x10, {1}));
    CHECK_FALSE(reaches(handles, alias + 0x1000, 0x10, {1}));
}

TEST_CASE("Fast draw compares texture handle data word by word", "[fast_draw]") {
    DeltaFixture f;
    f.memory.BindRasterizer(&f.rasterizer);
    constexpr GPUVAddr handles = 0x100000; // handles at 0x10 and 0x80, draw data between
    constexpr GPUVAddr alias = 0x300000;
    constexpr GPUVAddr split = 0x500000; // two pages that map device memory apart
    constexpr auto Kind = Tegra::PTEKind::PITCH;
    f.memory.Map(handles, 0x200000, 0x1000, Kind, false);
    f.memory.Map(alias, 0x200000, 0x1000, Kind, false);
    const std::array<std::pair<u32, u32>, 2> words{{{0x10, 0x14}, {0x80, 0x84}}};
    FastDraw::DeviceSpans spans;
    const auto reaches = [&](GPUVAddr handle_buffer, GPUVAddr buffer, u32 offset) {
        FastDraw::g_delta = {};
        f.SetConstantBuffer(buffer, 0x2000, offset);
        f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), 1);
        spans.Update(f.memory, handle_buffer + 0x10, 0x74);
        return FastDraw::g_delta.Reaches(handle_buffer + 0x10, 0x74, spans) &&
               FastDraw::g_delta.ReachesWords(handle_buffer, words, spans);
    };
    CHECK(reaches(handles, handles, 0x10));
    CHECK(reaches(handles, handles, 0x80));
    CHECK_FALSE(reaches(handles, handles, 0x40)); // draw data between the handles
    CHECK(reaches(handles, alias, 0x80));
    CHECK_FALSE(reaches(handles, alias, 0x40));

    // Not one run of device memory: a device write anywhere in the range counts.
    f.memory.Map(split, 0x401000, 0x1000, Kind, false);
    f.memory.Map(split + 0x1000, 0x400000, 0x1000, Kind, false);
    f.memory.Map(alias, 0x401000, 0x1000, Kind, false);
    CHECK(reaches(split + 0xf80, alias, 0xfc0));
}

TEST_CASE("Fast draw records and breaks nothing while switched off", "[fast_draw]") {
    DeltaFixture f;
    FastDrawFixture::SetSwitch(false);
    const u64 serial = FastDraw::Serial();
    f.Write(MAXWELL3D_REG_INDEX(blend.enable[0]), f.Reg(MAXWELL3D_REG_INDEX(blend.enable[0])) ^ 1);
    f.SetConstantBuffer(0x10000, 0x100, 0);
    f.Write(MAXWELL3D_REG_INDEX(bind_groups[0].raw_config), 0x11);
    f.Write(MAXWELL3D_REG_INDEX(const_buffer.buffer), 1);
    FastDraw::Break(FastDraw::Reason::Fence);
    CHECK(FastDraw::Serial() == serial);
    CHECK(FastDraw::g_delta.Empty());
}

TEST_CASE("Fast draw keeps base vertex and instance as draw parameters", "[fast_draw]") {
    FastDrawFixture f;
    f.DrawArrays(0, 3);
    f.Write(MAXWELL3D_REG_INDEX(global_base_vertex_index),
            f.Reg(MAXWELL3D_REG_INDEX(global_base_vertex_index)) + 64);
    f.Write(MAXWELL3D_REG_INDEX(global_base_instance_index),
            f.Reg(MAXWELL3D_REG_INDEX(global_base_instance_index)) + 1);
    f.DrawArrays(0, 3);
    CHECK(f.Reusable());
}

TEST_CASE("Fast draw keeps vertex stream addresses and sizes light", "[fast_draw]") {
    DeltaFixture f;
    const auto stream = [](u32 index, u32 word) {
        return static_cast<u32>(MAXWELL3D_REG_INDEX(vertex_streams) + index * 4 + word);
    };
    const auto light = [&](u32 method) {
        FastDraw::g_delta = {};
        const u64 serial = FastDraw::Serial();
        f.Write(method, f.Reg(method) + 0x40);
        return FastDraw::Serial() == serial;
    };
    CHECK(light(stream(0, 1)));
    CHECK(light(stream(3, 2)));
    CHECK(light(0x600 / 4 + 1));          // SET_VERTEX_STREAM_SIZE_B(0)
    CHECK(light(0x600 / 4 + 31 * 2));     // SET_VERTEX_STREAM_SIZE_A(31)
    CHECK(light(MAXWELL3D_REG_INDEX(vertex_stream_limits) + 5 * 2 + 1));
    CHECK_FALSE(light(stream(0, 0)));     // stride and enable
    CHECK_FALSE(light(stream(0, 3)));     // frequency
    CHECK_FALSE(light(0x600 / 4 + 64));   // past the sizes
}

TEST_CASE("Fast draw keeps vertex attribute formats light only with dynamic vertex input",
          "[fast_draw]") {
    DeltaFixture f;
    constexpr u32 formats = MAXWELL3D_REG_INDEX(vertex_attrib_format);
    const auto light = [&](u32 method) {
        FastDraw::g_delta = {};
        const u64 serial = FastDraw::Serial();
        f.Write(method, f.Reg(method) + 0x40); // the attribute's offset
        return FastDraw::Serial() == serial;
    };
    const bool previous = FastDraw::g_dynamic_vertex_input.exchange(false);
    CHECK_FALSE(light(formats)); // the formats are in the pipeline key
    FastDraw::g_dynamic_vertex_input.store(true);
    CHECK(light(formats));
    CHECK(light(formats + Maxwell3D::Regs::NumVertexAttributes - 1));
    CHECK_FALSE(light(formats + Maxwell3D::Regs::NumVertexAttributes));
    f.DrawArrays(0, 3);
    f.Write(formats + 2, f.Reg(formats + 2) + 0x40);
    f.DrawArrays(3, 3);
    CHECK(f.Reusable());
    FastDraw::g_dynamic_vertex_input.store(previous);
}

TEST_CASE("Fast draw breaks skip the add only on the drawing thread, after its first since a read",
          "[fast_draw]") {
    FastDraw::SetEnabledForTest(true);
    const auto current = [] { return FastDraw::g_serial.load(std::memory_order_relaxed); };
    const u64 read = FastDraw::Serial();
    FastDraw::Break(FastDraw::Reason::Register, 1);
    CHECK(current() == read + 1);
    // More breaks before the next read change nothing a draw compares.
    FastDraw::Break(FastDraw::Reason::Register, 2);
    FastDraw::Break(FastDraw::Reason::Method, 3);
    CHECK(current() == read + 1);
    // A thread that never reads the serial bumps every time.
    std::thread other{[] {
        FastDraw::Break(FastDraw::Reason::Invalidation);
        FastDraw::Break(FastDraw::Reason::Invalidation);
    }};
    other.join();
    CHECK(current() == read + 3);
    // After a read, the drawing thread's next break bumps again.
    const u64 again = FastDraw::Serial();
    CHECK(again == read + 3);
    FastDraw::Break(FastDraw::Reason::Register, 4);
    CHECK(current() == again + 1);
}

TEST_CASE("Unsplit callbacks end draw reuse before they execute", "[fast_draw][unsplit_fast_draw]") {
    const bool previous_reads_serial = FastDraw::t_reads_serial;
    const bool previous_serial_bumped = FastDraw::t_serial_bumped;
    SCOPE_EXIT {
        FastDraw::t_reads_serial = previous_reads_serial;
        FastDraw::t_serial_bumped = previous_serial_bumped;
    };
    FastDrawFixture f;
    VideoCommon::GpuSplit split;
    REQUIRE_FALSE(split.Active());
    f.DrawArrays(0, 3);
    const u64 before = f.rasterizer.draws.back();
    u64 inside{};
    u32 calls{};
    split.Run([&] {
        ++calls;
        inside = FastDraw::Serial();
    });
    f.DrawArrays(3, 3);
    REQUIRE(calls == 1);
    REQUIRE(inside != before);
    REQUIRE_FALSE(f.Reusable());
}
