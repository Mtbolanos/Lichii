// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <bit>
#include <initializer_list>
#include <vector>

#include <nlohmann/json.hpp>

#include "video_core/dma_pusher.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/gpu_trace/sample.h"
#include "video_core/gpu_trace/trace_file.h"
#include "video_core/pte_kind.h"
#include "video_core/textures/decoders.h"

namespace VideoCommon::GpuTrace {
namespace {
using Tegra::Engines::Maxwell3D;

constexpr u64 DeviceBase = 0x40000000;
constexpr u64 PhysicalBase = 0x40000000;
constexpr u64 GpuBase = 0x200000000;
constexpr u64 Size = 16ULL << 20;
constexpr u64 PushOffset = 0;
constexpr u64 ImageOffset = 0x100000;
constexpr u64 TargetOffset = 0x200000;
constexpr u32 Width = 256;
constexpr u32 Height = 144;
constexpr s32 Channel = 1;

struct Rect {
    u32 x0, y0, x1, y1;
};

class Commands {
public:
    void Method(u32 method, std::initializer_list<u32> arguments) {
        Tegra::CommandHeader header{};
        header.method.Assign(method);
        header.subchannel.Assign(0);
        header.arg_count.Assign(static_cast<u32>(arguments.size()));
        header.mode.Assign(Tegra::SubmissionMode::Increasing);
        words.push_back(header.argument);
        words.insert(words.end(), arguments);
    }

    void Clear(std::array<f32, 4> color, const Rect* scissor) {
        if (scissor != nullptr) {
            Method(MAXWELL3D_REG_INDEX(scissor_test),
                   {1, scissor->x0 | (scissor->x1 << 16), scissor->y0 | (scissor->y1 << 16)});
        }
        Method(MAXWELL3D_REG_INDEX(clear_control), {scissor != nullptr ? 0x100u : 0u});
        Method(MAXWELL3D_REG_INDEX(clear_color),
               {std::bit_cast<u32>(color[0]), std::bit_cast<u32>(color[1]),
                std::bit_cast<u32>(color[2]), std::bit_cast<u32>(color[3])});
        // R, G, B and A of render target 0, layer 0.
        Method(MAXWELL3D_REG_INDEX(clear_surface), {0x3c});
        if (scissor != nullptr) {
            Method(MAXWELL3D_REG_INDEX(scissor_test), {0});
        }
    }

    std::vector<u32> words;
};

std::span<const u8> AsBytes(const auto& values) {
    return {reinterpret_cast<const u8*>(values.data()), values.size() * sizeof(values[0])};
}

/// Presented images and render targets are block linear with 16-GOB blocks, as a game's are.
constexpr u32 BlockHeightLog2 = 4;

std::vector<u32> Image(u32 seed) {
    std::vector<u32> pixels(Width * Height);
    for (u32 y = 0; y < Height; ++y) {
        for (u32 x = 0; x < Width; ++x) {
            const u32 r = (x * 255 / (Width - 1)) ^ (seed * 0x3f);
            const u32 g = (y * 255 / (Height - 1));
            const u32 b = ((x / 16 + y / 16 + seed) % 2) * 0xff;
            pixels[y * Width + x] = (r & 0xff) | (g << 8) | (b << 16) | 0xff000000;
        }
    }
    std::vector<u32> tiled(Tegra::Texture::CalculateSize(true, 4, Width, Height, 1,
                                                         BlockHeightLog2, 0) /
                           sizeof(u32));
    Tegra::Texture::SwizzleTexture(
        std::span<u8>{reinterpret_cast<u8*>(tiled.data()), tiled.size() * sizeof(u32)},
        AsBytes(pixels), 4, Width, Height, 1, BlockHeightLog2, 0, 1, true);
    return tiled;
}

void Present(Writer& writer, u64 offset, u64 time_ns) {
    PresentLayer layer{};
    layer.address = DeviceBase + offset;
    layer.width = Width;
    layer.height = Height;
    layer.stride = Width;
    layer.pixel_format = 1; // Rgba8888
    layer.crop_right = static_cast<s32>(Width);
    layer.crop_bottom = static_cast<s32>(Height);
    layer.layer_stack_mask = 1;
    layer.native_tiling = 1;
    const PresentRecord header{time_ns, 1, 0};
    const std::array<std::span<const u8>, 2> parts{
        std::span<const u8>{reinterpret_cast<const u8*>(&header), sizeof(header)},
        std::span<const u8>{reinterpret_cast<const u8*>(&layer), sizeof(layer)}};
    writer.Write(RecordType::Present, parts);
}

/// The commands go to memory first (a CPU write), as a game writes them.
void Submit(Writer& writer, u64 offset, const Commands& commands) {
    writer.WriteMemory(PhysicalBase + offset, AsBytes(commands.words));
    Tegra::CommandListHeader entry{};
    entry.addr.Assign(GpuBase + offset);
    entry.size.Assign(commands.words.size());
    const SubmitRecord header{Channel, 1, 0, 0, 0, 0};
    const std::array<std::span<const u8>, 2> parts{
        std::span<const u8>{reinterpret_cast<const u8*>(&header), sizeof(header)},
        std::span<const u8>{reinterpret_cast<const u8*>(&entry.raw), sizeof(entry.raw)}};
    writer.Write(RecordType::Submit, parts);
}

Commands BindTarget() {
    Commands commands;
    const u64 address = GpuBase + TargetOffset;
    commands.Method(MAXWELL3D_REG_INDEX(rt),
                    {static_cast<u32>(address >> 32), static_cast<u32>(address), Width, Height,
                     static_cast<u32>(Tegra::RenderTargetFormat::A8B8G8R8_UNORM),
                     BlockHeightLog2 << 4, 1, 0, 0});
    commands.Method(MAXWELL3D_REG_INDEX(rt_control), {1});
    commands.Method(MAXWELL3D_REG_INDEX(surface_clip), {Width << 16, Height << 16});
    return commands;
}
} // Anonymous namespace

bool WriteSampleTrace(const std::filesystem::path& path) {
    Writer writer{path};
    if (!writer.IsOpen()) {
        return false;
    }
    const nlohmann::json config{
        {"program_id", "0400000000000001"}, {"switch2", true},  {"synthetic", true},
        {"memory_layout", 0},               {"width", Width},   {"height", Height},
    };
    const std::string text = config.dump();
    writer.Write(RecordType::Config, AsBytes(text));
    writer.WritePod(RecordType::AddressSpace, AddressSpaceRecord{0, 40, 1ULL << 34, 16, 12});
    writer.WritePod(RecordType::SmmuMap, SmmuMapRecord{DeviceBase, PhysicalBase, Size});
    writer.WritePod(RecordType::GmmuMap,
                    GmmuMapRecord{0, static_cast<u32>(Tegra::PTEKind::PITCH), GpuBase, DeviceBase,
                                  Size, 0, 0});
    writer.WritePod(RecordType::Channel, ChannelRecord{Channel, 0, 0x0400000000000001ULL});
    writer.Write(RecordType::SnapshotEnd, std::span<const u8>{});

    constexpr u64 Frame = 16'666'667;
    writer.WriteMemory(PhysicalBase + ImageOffset, AsBytes(Image(0)));
    Present(writer, ImageOffset, Frame);

    Commands first = BindTarget();
    const Rect square{64, 36, 192, 108};
    first.Clear({1.0f, 0.0f, 0.0f, 1.0f}, nullptr);
    first.Clear({0.0f, 1.0f, 0.0f, 1.0f}, &square);
    Submit(writer, PushOffset, first);
    Present(writer, TargetOffset, Frame * 2);

    writer.WriteMemory(PhysicalBase + ImageOffset, AsBytes(Image(1)));
    Present(writer, ImageOffset, Frame * 3);

    Commands second;
    const Rect left{16, 16, 96, 128};
    const Rect right{160, 16, 240, 128};
    second.Clear({0.0f, 0.0f, 1.0f, 1.0f}, nullptr);
    second.Clear({1.0f, 1.0f, 0.0f, 1.0f}, &left);
    second.Clear({1.0f, 0.0f, 1.0f, 1.0f}, &right);
    Submit(writer, PushOffset + 0x8000, second);
    Present(writer, TargetOffset, Frame * 4);
    return writer.Finish();
}

} // namespace VideoCommon::GpuTrace
