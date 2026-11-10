// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <span>
#include <string>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include "audio_core/common/common.h"
#include "audio_core/common/wave_buffer.h"
#include "audio_core/renderer/command/data_source/decode.h"
#include "audio_core/renderer/voice/voice_state.h"
#include "common/page_table.h"
#include "common/scope_exit.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/kernel.h"
#include "core/memory.h"
#include "tests/core/system_test_context.h"

namespace {

using AdpcmContext = AudioCore::Renderer::VoiceState::AdpcmContext;

constexpr u32 SamplesPerFrame = 14;
constexpr u32 FrameCount = 12;
constexpr u32 SampleCount = SamplesPerFrame * FrameCount;

// Predictor pairs in 1.11 fixed point, all stable.
constexpr std::array<s16, 16> Coefficients{
    0, 0, 1024, 0, 1843, 0, 2048, -1024, 3040, -1218, 3482, -1700, -1024, 512, 1638, -410,
};

// Frames of a header byte (predictor << 4 | scale) and 14 nibbles: every predictor pair, every
// nibble value, and scales that keep the samples inside s16.
std::array<u8, FrameCount * 8> MakeFrames() {
    std::array<u8, FrameCount * 8> frames;
    u32 state = 0x2545F491;
    for (u32 frame = 0; frame < FrameCount; ++frame) {
        for (u32 i = 0; i < 8; ++i) {
            state = state * 1664525 + 1013904223;
            frames[frame * 8 + i] = static_cast<u8>(state >> 24);
        }
        frames[frame * 8] = static_cast<u8>((frame % 8) << 4 | (frame % 4 + 8));
    }
    return frames;
}

// A plain decoder for samples [begin, end) of the frames, starting from a context.
std::vector<s32> Reference(std::span<const u8> frames, u32 begin, u32 end, AdpcmContext context) {
    static constexpr std::array<s32, 16> Steps{
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
    };
    std::vector<s32> samples;
    s32 yn0 = context.yn0;
    s32 yn1 = context.yn1;
    u32 header = context.header;
    for (u32 sample = begin; sample < end; ++sample) {
        const u32 frame = sample / SamplesPerFrame;
        const u32 index = sample % SamplesPerFrame;
        if (index == 0) {
            header = frames[frame * 8];
        }
        const u8 byte = frames[frame * 8 + 1 + index / 2];
        const s32 step = Steps[index % 2 == 0 ? byte >> 4 : byte & 0xF];
        const u32 predictor = (header >> 4) & 7;
        const s32 value = (step * (1 << (header & 0xF)) * 2048 + 0x400 +
                           Coefficients[predictor * 2] * yn0 +
                           Coefficients[predictor * 2 + 1] * yn1) >> 11;
        yn1 = yn0;
        yn0 = std::clamp(value, -0x8000, 0x7FFF);
        samples.push_back(yn0);
    }
    return samples;
}

// Where two sample runs first differ, as a string Catch prints instead of both whole vectors.
std::string FirstDifference(const std::vector<s32>& decoded, const std::vector<s32>& expected) {
    const auto [a, b] = std::ranges::mismatch(decoded, expected);
    if (a == decoded.end() && b == expected.end()) {
        return "none";
    }
    const auto index = std::distance(decoded.begin(), a);
    return fmt::format("sample {} of {}: decoded {}, expected {}", index, expected.size(),
                       a == decoded.end() ? std::string{"nothing"} : std::to_string(*a),
                       b == expected.end() ? std::string{"nothing"} : std::to_string(*b));
}

} // Anonymous namespace

TEST_CASE("ADPCM wave buffers decode to the reference decoder's samples", "[audio_core][adpcm]") {
    constexpr u64 Base = 0x4000;
    constexpr u64 CoefficientAddress = Base + 0x800;
    constexpr u64 ContextAddress = Base + 0x900;
    // The second frame's sixth sample: decoding from here takes the frame header from the context.
    constexpr u32 MidFrame = SamplesPerFrame + 5;

    ScopedSystemTestContext test_context;
    Core::System system;
    system.Initialize();
    Kernel::KProcess process{system.Kernel()};
    SCOPE_EXIT {
        system.Kernel().UnregisterKernelObject(&process);
    };
    auto& table = process.GetPageTable().GetImpl();
    table.Resize(20, 12);
    auto* page = system.DeviceMemory().buffer.BackingBasePointer() + 0x10000;
    std::memset(page, 0, 0x1000);
    table.entries.GetAndFault(Base >> 12)
        .Store(false, Common::PageType::Memory, 1, reinterpret_cast<uintptr_t>(page) - Base);
    auto& memory = process.GetMemory();
    memory.SetCurrentPageTable(process);

    const auto frames = MakeFrames();
    std::memcpy(page, frames.data(), frames.size());
    std::memcpy(page + (CoefficientAddress - Base), Coefficients.data(), sizeof(Coefficients));
    const auto place_context = [&](const AdpcmContext& context) {
        std::memcpy(page + (ContextAddress - Base), &context, sizeof(context));
    };

    AudioCore::Renderer::VoiceState voice{};
    std::array<AudioCore::WaveBufferVersion2, AudioCore::MaxWaveBuffers> wave_buffers{};
    // 40 samples a call: calls end inside frames, so whole frames and single samples both decode.
    const auto render = [&](u32 total) {
        std::vector<s32> output(total);
        for (u32 done = 0; done < total; done += 40) {
            const u32 count = (std::min)(40U, total - done);
            AudioCore::Renderer::DecodeFromWaveBuffers(
                memory, {
                            .sample_format = AudioCore::SampleFormat::Adpcm,
                            .output = std::span{output}.subspan(done, count),
                            .voice_state = &voice,
                            .wave_buffers = wave_buffers,
                            .channel = 0,
                            .channel_count = 1,
                            .src_quality = AudioCore::SrcQuality::Medium,
                            .pitch = 1.0f,
                            .source_sample_rate = 48000,
                            .target_sample_rate = 48000,
                            .sample_count = count,
                            .data_address = CoefficientAddress,
                            .data_size = sizeof(Coefficients),
                            .IsVoicePlayedSampleCountResetAtLoopPointSupported = false,
                            .IsVoicePitchAndSrcSkippedSupported = true,
                        });
        }
        return output;
    };
    const AudioCore::WaveBufferVersion2 whole{
        .buffer = Base,
        .context = 0,
        .buffer_size = frames.size(),
        .context_size = 0,
        .start_offset = 0,
        .end_offset = SampleCount,
        .loop_start_offset = 0,
        .loop_end_offset = 0,
        .loop_count = 0,
        .loop = false,
        .stream_ended = false,
    };
    const auto all = Reference(frames, 0, SampleCount, {});

    SECTION("from a zero history") {
        wave_buffers[0] = whole;
        voice.wave_buffer_valid[0] = true;
        REQUIRE(FirstDifference(render(SampleCount), all) == "none");
    }

    SECTION("from the guest's context") {
        const AdpcmContext context{.header = 0x39, .yn0 = -12345, .yn1 = 6789};
        place_context(context);
        wave_buffers[0] = whole;
        wave_buffers[0].context = ContextAddress;
        wave_buffers[0].context_size = sizeof(context);
        wave_buffers[0].start_offset = MidFrame;
        voice.wave_buffer_valid[0] = true;
        REQUIRE(FirstDifference(render(SampleCount - MidFrame),
                                Reference(frames, MidFrame, SampleCount, context)) == "none");
    }

    SECTION("a loop continues from its loop context at every pass") {
        // As a game plays a sustained note: the whole sample once, then its loop region with the
        // context the encoder stored for the loop start.
        const AdpcmContext context{.header = frames[MidFrame / SamplesPerFrame * 8],
                                   .yn0 = static_cast<s16>(all[MidFrame - 1]),
                                   .yn1 = static_cast<s16>(all[MidFrame - 2])};
        place_context(context);
        wave_buffers[0] = whole;
        wave_buffers[1] = whole;
        wave_buffers[1].context = ContextAddress;
        wave_buffers[1].context_size = sizeof(context);
        wave_buffers[1].start_offset = MidFrame;
        wave_buffers[1].loop_start_offset = MidFrame;
        wave_buffers[1].loop_end_offset = SampleCount;
        wave_buffers[1].loop_count = -1;
        wave_buffers[1].loop = true;
        voice.wave_buffer_valid[0] = true;
        voice.wave_buffer_valid[1] = true;

        auto expected = all;
        for (u32 pass = 0; pass < 3; ++pass) {
            expected.insert(expected.end(), all.begin() + MidFrame, all.end());
        }
        REQUIRE(FirstDifference(render(static_cast<u32>(expected.size())), expected) == "none");
    }
}
