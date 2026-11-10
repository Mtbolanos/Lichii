// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <limits>
#include <span>
#include <thread>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "common/atomic_ops.h"
#include "common/scope_exit.h"
#include "core/core_timing.h"
#include "core/hle/kernel/k_readable_event.h"
#include "core/hle/service/audio/errors.h"
#include "core/hle/service/audio/nx2_audio_device_manager.h"
#include "core/hle/service/audio/nx2_sysmix.h"
#include "tests/audio_core/sysmix_test_context.h"

using Service::Audio::SysmixOutput;
using SysmixTests::Context;

TEST_CASE("sysmix input accepts a ring without output timestamp storage", "[sysmix][input]") {
    Context context;
    SysmixOutput stream{context.system, *context.process, context.sink, 2,
                        Context::Base, 0x2000, 2, true};
    REQUIRE(stream.Open(2, 1, 48000, 2000, 0, 0).IsSuccess());
}

TEST_CASE("sysmix input publishes fresh mono capture across ring wrap", "[sysmix][input]") {
    Context context;
    const auto previous_gain = Settings::values.microphone_volume.GetValue();
    Settings::values.microphone_volume.SetValue(100);
    SCOPE_EXIT {
        Settings::values.microphone_volume.SetValue(previous_gain);
    };
    const u32 cursor = 8;
    REQUIRE(context.Memory().WriteBlock(Context::Base + 0x44, &cursor, sizeof(cursor)));
    REQUIRE(context.Memory().WriteBlock(Context::Base + 0x1004, &cursor, sizeof(cursor)));
    SysmixOutput stream{
        context.system, *context.process, context.sink, 2, Context::Base, Context::Size, 2, true};
    REQUIRE(stream.IsInput());
    REQUIRE(stream.Open(2, 1, 48000, 8, 0, 0).IsSuccess());
    REQUIRE(stream.SetDataNotification(2, 2).IsSuccess());
    REQUIRE(stream.SetEngaged(true).IsSuccess());
    stream.Pump();
    u32 write{};
    REQUIRE(context.Memory().Read32Acquire(Context::Base + 0x1004, write));
    REQUIRE(write == 8);
    REQUIRE_FALSE(stream.GetDataEvent()->IsSignaled(context.system.Kernel()));
    const std::array<s16, 6> samples{1000, 3000, -1000, -3000, 100, 300};
    context.sink.last->ProcessAudioIn(samples, 3);
    stream.Pump();
    REQUIRE(context.Memory().Read32Acquire(Context::Base + 0x1004, write));
    REQUIRE(write == 2);
    u32 read{};
    REQUIRE(context.Memory().Read32Acquire(Context::Base + 0x44, read));
    REQUIRE(read == 8);
    s16 last{};
    std::array<s16, 2> first{};
    REQUIRE(context.Memory().ReadBlock(Context::Base + 0x1040 + 8 * 2, &last, sizeof(last)));
    REQUIRE(context.Memory().ReadBlock(Context::Base + 0x1040, first.data(), sizeof(first)));
    REQUIRE(last == 2000);
    REQUIRE(first == std::array<s16, 2>{-2000, 200});
    REQUIRE(stream.GetDataEvent()->IsSignaled(context.system.Kernel()));
    REQUIRE(stream.GetCounters().accepted_frames == 3);
    stream.Pump();
    REQUIRE(stream.GetCounters().accepted_frames == 3);
    u32 purged{};
    REQUIRE(stream.Purge(0, purged).IsSuccess());
    REQUIRE(purged == 3);
    REQUIRE(context.Memory().Read32Acquire(Context::Base + 0x44, read));
    REQUIRE(read == write);
}

TEST_CASE("sysmix input captures silence when the host microphone cannot open", "[sysmix][input]") {
    Context context;
    context.sink.ready = false;
    std::array<s16, 2400> stale{};
    stale.fill(0x1234);
    REQUIRE(context.Memory().WriteBlock(Context::Base + 0x1040, stale.data(), sizeof(stale)));
    SysmixOutput stream{
        context.system, *context.process, context.sink, 2, Context::Base, Context::Size, 2, true};
    REQUIRE(stream.Open(2, 1, 48000, 4096, 0, 0).IsSuccess());
    REQUIRE(context.sink.last == nullptr);
    REQUIRE(context.sink.closed == 1);
    REQUIRE(stream.SetEngaged(true).IsSuccess());
    REQUIRE(stream.GetState() == 2);
    const auto advance = [&] {
        context.system.CoreTiming().cpu_ticks += Common::WallClock::CPUTickFreq / 200;
        context.system.CoreTiming().Advance();
        stream.Pump();
    };
    for (int block = 0; block < 10; ++block) {
        advance();
    }
    u32 write{};
    REQUIRE(context.Memory().Read32Acquire(Context::Base + 0x1004, write));
    REQUIRE(write >= 9 * 240);
    REQUIRE(write <= 10 * 240);
    std::vector<s16> captured(write);
    REQUIRE(context.Memory().ReadBlock(Context::Base + 0x1040, captured.data(),
                                       captured.size() * sizeof(s16)));
    REQUIRE(std::ranges::all_of(captured, [](s16 sample) { return sample == 0; }));
    REQUIRE(stream.GetCounters().nonzero_frames == 0);
    REQUIRE(stream.SetEngaged(false).IsSuccess());
    const u32 stopped = write;
    advance();
    REQUIRE(context.Memory().Read32Acquire(Context::Base + 0x1004, write));
    REQUIRE(write == stopped);
}

TEST_CASE("sysmix input captures silence when the host microphone cannot start",
          "[sysmix][input]") {
    Context context;
    context.sink.fail_start = true;
    SysmixOutput stream{
        context.system, *context.process, context.sink, 2, Context::Base, Context::Size, 2, true};
    REQUIRE(stream.Open(2, 1, 48000, 1024, 0, 0).IsSuccess());
    REQUIRE(context.sink.last != nullptr);
    REQUIRE(stream.SetEngaged(true).IsSuccess());
    REQUIRE(context.sink.last == nullptr);
    for (int block = 0; block < 2; ++block) {
        context.system.CoreTiming().cpu_ticks += Common::WallClock::CPUTickFreq / 200;
        context.system.CoreTiming().Advance();
        stream.Pump();
    }
    u32 write{};
    REQUIRE(context.Memory().Read32Acquire(Context::Base + 0x1004, write));
    REQUIRE(write >= 240);
}

TEST_CASE("sysmix input applies gain to float capture and protects producer ownership",
          "[sysmix][input]") {
    Context context;
    const auto previous_gain = Settings::values.microphone_volume.GetValue();
    Settings::values.microphone_volume.SetValue(100);
    SCOPE_EXIT {
        Settings::values.microphone_volume.SetValue(previous_gain);
    };
    SysmixOutput stream{
        context.system, *context.process, context.sink, 2, Context::Base, Context::Size, 2, true};
    REQUIRE(stream.Open(5, 2, 48000, 8, 0, 0).IsSuccess());
    REQUIRE(stream.SetGain(0.5f, 0).IsSuccess());
    REQUIRE(stream.SetEngaged(true).IsSuccess());
    const std::array<s16, 2> samples{16384, -16384};
    context.sink.last->ProcessAudioIn(samples, 1);
    stream.Pump();
    std::array<f32, 2> recorded{};
    REQUIRE(context.Memory().ReadBlock(Context::Base + 0x1040, recorded.data(), sizeof(recorded)));
    REQUIRE(recorded == std::array<f32, 2>{0.25f, -0.25f});
    const u32 tampered = 2;
    REQUIRE(context.Memory().WriteBlock(Context::Base + 0x1004, &tampered, sizeof(tampered)));
    stream.Pump();
    REQUIRE(stream.GetState() == 0);
    REQUIRE(stream.GetCounters().faults == 1);
}

TEST_CASE("sysmix input purge preserves capture during concurrent guest consumption",
          "[sysmix][input]") {
    Context context;
    constexpr u32 frames = 2048;
    SysmixOutput stream{
        context.system, *context.process, context.sink, 2, Context::Base, Context::Size, 2, true};
    REQUIRE(stream.Open(2, 2, 48000, frames, 0, 0).IsSuccess());
    REQUIRE(stream.SetEngaged(true).IsSuccess());
    auto* read_cursor = reinterpret_cast<u32_le*>(context.backing + 0x44);
    auto* write_cursor = reinterpret_cast<u32_le*>(context.backing + 0x1004);
    std::jthread guest{[&](std::stop_token stop) {
        while (!stop.stop_requested()) {
            u32_le read{}, write{};
            static_cast<void>(Common::AtomicCompareAndSwap(read_cursor, u32_le{0}, u32_le{0}, read));
            static_cast<void>(Common::AtomicCompareAndSwap(write_cursor, u32_le{0}, u32_le{0}, write));
            if (read != write) {
                static_cast<void>(Common::AtomicCompareAndSwap(
                    read_cursor, u32_le{(static_cast<u32>(read) + 1) % (frames + 1)}, read));
            }
        }
    }};
    const std::array<s16, 480> samples{};
    for (u32 iteration = 0; iteration < 4096; ++iteration) {
        context.sink.last->ProcessAudioIn(samples, 240);
        stream.Pump();
        u32 purged = 99;
        const auto result = stream.Purge(iteration % 2 == 0 ? 1 : 0, purged);
        REQUIRE(stream.GetState() == 2);
        REQUIRE(stream.GetCounters().faults == 0);
        if (result.IsFailure()) {
            REQUIRE(result == Service::Audio::ResultOperationFailed);
            REQUIRE(purged == 0);
        }
    }
    REQUIRE(stream.GetCounters().accepted_frames > 0);
}

namespace {
void Cursor(Context& context, u32 read, u32 write) {
    REQUIRE(context.Memory().WriteBlock(Context::Base + 0x44, &read, sizeof(read)));
    REQUIRE(context.Memory().WriteBlock(Context::Base + 0x1004, &write, sizeof(write)));
}
u32 ReadCursor(Context& context) {
    u32 read{};
    REQUIRE(context.Memory().Read32Acquire(Context::Base + 0x44, read));
    return read;
}
void Sample(Context& context, u32 slot, f32 left, f32 right) {
    const std::array<f32, 2> frame{left, right};
    REQUIRE(context.Memory().WriteBlock(Context::Base + 0x1040 + u64{slot} * 8,
                                        frame.data(), sizeof(frame)));
}
}

TEST_CASE("sysmix validates the exact F32 ring allocation and rejects unsupported formats", "[sysmix]") {
    Context context;
    SysmixOutput stream{context.system, *context.process, context.sink, 1, Context::Base,
                        Context::Size, 0x24};
    REQUIRE(stream.Open(3, 2, 48000, 2048, 0, 0).IsFailure());
    REQUIRE(stream.Open(5, 1, 48000, 2048, 0, 0).IsFailure());
    REQUIRE(stream.Open(5, 2, 44100, 2048, 0, 0).IsFailure());
    REQUIRE(stream.Open(5, 2, 48000, 2048, 1, 0).IsFailure());
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 1).IsFailure());
    REQUIRE(stream.Open(5, 2, 48000, 0, 0, 0).IsFailure());
    REQUIRE(stream.Open(5, 2, 48000, 0xffffffff, 0, 0).IsFailure());
    REQUIRE(context.sink.last == nullptr);
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsFailure());
    REQUIRE(context.process->GetReferenceCount() == 2);
    REQUIRE(SysmixOutput::ValidateRegion(*context.process, Context::Base + 1,
                                         Context::Size, 0x24).IsFailure());
    REQUIRE(SysmixOutput::ValidateRegion(*context.process, Context::Base,
                                         Context::Size, 4).IsFailure());
    SysmixOutput short_stream{context.system, *context.process, context.sink, 2, Context::Base,
                              0x5000, 0x24};
    REQUIRE(short_stream.Open(5, 2, 48000, 2048, 0, 0).IsFailure());
}

TEST_CASE("sysmix copies the physical extra slot and wrap then publishes consumer", "[sysmix]") {
    Context context;
    Cursor(context, 2048, 1);
    Sample(context, 2048, 0.5f, -0.5f);
    Sample(context, 0, 1.0f, -1.0f);
    SysmixOutput stream{context.system, *context.process, context.sink, 1, Context::Base,
                        Context::Size, 0x24};
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
    REQUIRE(stream.SetEngaged(true).IsSuccess());
    stream.Pump();
    REQUIRE(ReadCursor(context) == 1);
    REQUIRE(stream.GetCounters().accepted_frames == 2);
    REQUIRE(stream.GetCounters().nonzero_frames == 2);
    std::array<s16, 480> output{};
    context.sink.last->ProcessAudioOutAndRender(output, 240);
    REQUIRE(output[0] == 16384);
    REQUIRE(output[1] == -16384);
    REQUIRE(output[2] == 32767);
    REQUIRE(output[3] == -32768);
    REQUIRE(std::ranges::all_of(std::span{output}.subspan(4), [](s16 value) { return value == 0; }));
    REQUIRE(context.backing[0x5048] == 0);
    REQUIRE(context.backing[0x5fff] == 0);
}

TEST_CASE("sysmix plays the Sports Resort Q15 movie ring including wrap and gain", "[sysmix]") {
    Context context;
    Cursor(context, 960, 1);
    const std::array<s16, 2> last{16384, -16384};
    const std::array<s16, 2> first{32767, -32768};
    REQUIRE(context.Memory().WriteBlock(Context::Base + 0x1040 + 960 * 4,
                                        last.data(), sizeof(last)));
    REQUIRE(context.Memory().WriteBlock(Context::Base + 0x1040, first.data(), sizeof(first)));
    SysmixOutput stream{context.system, *context.process, context.sink, 1, Context::Base,
                        0x3000, 0x24};
    REQUIRE(stream.Open(2, 2, 48000, 960, 0, 0).IsSuccess());
    REQUIRE(stream.SetGain(0.5f, 0).IsSuccess());
    REQUIRE(stream.SetDataNotification(2, 480).IsSuccess());
    REQUIRE(stream.SetEngaged(true).IsSuccess());
    stream.Pump();
    REQUIRE(ReadCursor(context) == 1);
    REQUIRE(stream.GetCounters().accepted_frames == 2);
    REQUIRE(stream.GetCounters().nonzero_frames == 2);
    std::array<s16, 480> output{};
    context.sink.last->ProcessAudioOutAndRender(output, 240);
    REQUIRE(output[0] == 8192);
    REQUIRE(output[1] == -8192);
    REQUIRE(output[2] == 16383);
    REQUIRE(output[3] == -16384);
    REQUIRE(std::ranges::all_of(std::span{output}.subspan(4), [](s16 value) { return value == 0; }));
    REQUIRE(stream.GetDataEvent()->IsSignaled(context.system.Kernel()));
}

TEST_CASE("sysmix advertises only implemented stereo PCM formats", "[sysmix]") {
    REQUIRE(SysmixOutput::IsFormatSupported(2, 2, 48000));
    REQUIRE(SysmixOutput::IsFormatSupported(5, 2, 48000));
    for (const u32 format : {0U, 1U, 3U, 4U, 6U, 258U, 261U}) {
        REQUIRE_FALSE(SysmixOutput::IsFormatSupported(format, 2, 48000));
    }
    REQUIRE_FALSE(SysmixOutput::IsFormatSupported(2, 1, 48000));
    REQUIRE_FALSE(SysmixOutput::IsFormatSupported(2, 6, 48000));
    REQUIRE_FALSE(SysmixOutput::IsFormatSupported(2, 2, 44100));
}

TEST_CASE("sysmix mode2 event uses actual free frames even before engagement", "[sysmix]") {
    Context context;
    Cursor(context, 0, 600);
    SysmixOutput stream{context.system, *context.process, context.sink, 1, Context::Base,
                        Context::Size, 0x24};
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
    REQUIRE(stream.SetDataNotification(1, 1536).IsFailure());
    REQUIRE(stream.SetDataNotification(2, -1).IsFailure());
    REQUIRE(stream.SetDataNotification(2, 2049).IsFailure());
    REQUIRE(stream.SetDataNotification(2, 1536).IsSuccess());
    REQUIRE_FALSE(stream.GetDataEvent()->IsSignaled(context.system.Kernel()));
    stream.Pump();
    REQUIRE(ReadCursor(context) == 0);
    REQUIRE_FALSE(stream.GetDataEvent()->IsSignaled(context.system.Kernel()));
    REQUIRE(stream.SetEngaged(true).IsSuccess());
    stream.Pump();
    REQUIRE(ReadCursor(context) == 240);
    REQUIRE(stream.GetDataEvent()->IsSignaled(context.system.Kernel()));
    REQUIRE(stream.SetEngaged(false).IsSuccess());
    REQUIRE(stream.GetDataEvent()->Clear(context.system.Kernel()).IsSuccess());
    stream.Pump();
    REQUIRE(stream.GetDataEvent()->IsSignaled(context.system.Kernel()));
    REQUIRE(stream.SetDataNotification(0, 0).IsSuccess());
    REQUIRE(stream.GetDataEvent()->Clear(context.system.Kernel()).IsSuccess());
    stream.Pump();
    REQUIRE_FALSE(stream.GetDataEvent()->IsSignaled(context.system.Kernel()));
}

TEST_CASE("sysmix backpressure leaves guest ownership unchanged and resumes in order", "[sysmix]") {
    Context context;
    Cursor(context, 0, 2048);
    for (u32 i = 0; i < 2048; ++i) {
        Sample(context, i, 0.25f, -0.25f);
    }
    SysmixOutput stream{context.system, *context.process, context.sink, 1, Context::Base,
                        Context::Size, 0x24};
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
    REQUIRE(stream.SetEngaged(true).IsSuccess());
    for (u32 i = 0; i < 3; ++i) {
        stream.Pump();
    }
    REQUIRE(ReadCursor(context) == 720);
    stream.Pump();
    REQUIRE(ReadCursor(context) == 720);
    REQUIRE(stream.GetCounters().accepted_frames == 720);
    REQUIRE(stream.GetCounters().backpressure == 1);
    std::array<s16, 480> output{};
    context.sink.last->ProcessAudioOutAndRender(output, 240);
    stream.Pump();
    REQUIRE(ReadCursor(context) == 960);
    REQUIRE(context.sink.last->GetQueueSize() == 3);
}

TEST_CASE("sysmix waits for a whole block while the host still has audio queued", "[sysmix]") {
    Context context;
    Cursor(context, 0, 340);
    for (u32 i = 0; i < 340; ++i) {
        Sample(context, i, 0.25f, -0.25f);
    }
    SysmixOutput stream{context.system, *context.process, context.sink, 1, Context::Base,
                        Context::Size, 0x24};
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
    REQUIRE(stream.SetEngaged(true).IsSuccess());
    stream.Pump();
    REQUIRE(ReadCursor(context) == 240);
    REQUIRE(context.sink.last->GetQueueSize() == 1);

    // The host starts that block and plays 100 of its frames: nothing is queued any more, but
    // 140 frames are still to play.
    std::array<s16, 960> output{};
    context.sink.last->ProcessAudioOutAndRender(std::span{output}.first(200), 100);
    REQUIRE(context.sink.last->GetQueueSize() == 0);
    REQUIRE(context.sink.last->HasPendingAudio());

    // 100 frames are left in the ring: no zero-padded block while the host has audio to play.
    stream.Pump();
    REQUIRE(ReadCursor(context) == 240);
    REQUIRE(stream.GetCounters().deferred == 1);
    REQUIRE(stream.GetCounters().admitted_blocks == 1);

    // The guest tops the ring up: the next block plays straight on, with no silence between.
    Cursor(context, 240, 480);
    for (u32 i = 340; i < 480; ++i) {
        Sample(context, i, 0.25f, -0.25f);
    }
    stream.Pump();
    REQUIRE(ReadCursor(context) == 480);
    REQUIRE(stream.GetCounters().accepted_frames == 480);
    context.sink.last->ProcessAudioOutAndRender(std::span{output}.subspan(200), 380);
    REQUIRE(std::ranges::all_of(output, [](s16 value) { return value == 8192 || value == -8192; }));
    REQUIRE_FALSE(context.sink.last->HasPendingAudio());

    // Once the host has drained, a short block is a real underrun and is sent padded, so a guest
    // waiting on more free space than those frames leave (threshold 2040) is woken.
    REQUIRE(context.sink.last->GetQueueSize() == 0);
    Cursor(context, 480, 500);
    REQUIRE(stream.SetDataNotification(2, 2040).IsSuccess());
    REQUIRE_FALSE(stream.GetDataEvent()->IsSignaled(context.system.Kernel()));
    stream.Pump();
    REQUIRE(ReadCursor(context) == 500);
    REQUIRE(stream.GetDataEvent()->IsSignaled(context.system.Kernel()));
    REQUIRE(stream.GetCounters().accepted_frames == 500);
    REQUIRE(stream.GetCounters().deferred == 1);
}

TEST_CASE("sysmix detects unmapped cursor publication even though exclusive store reports success", "[sysmix]") {
    Context context;
    Cursor(context, 0, 240);
    SysmixOutput stream{context.system, *context.process, context.sink, 1, Context::Base,
                        Context::Size, 0x24};
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
    REQUIRE(stream.SetEngaged(true).IsSuccess());
    context.sink.last->on_admit = [&] { context.Unmap(Context::Base); };
    stream.Pump();
    REQUIRE(context.Memory().WriteExclusive32(Context::Base + 0x44, 240, 0));
    u32 read{};
    REQUIRE_FALSE(context.Memory().Read32Acquire(Context::Base + 0x44, read));
    REQUIRE(stream.GetCounters().accepted_frames == 0);
    REQUIRE(stream.GetCounters().faults == 1);
    REQUIRE(stream.GetCounters().admitted_blocks == 1);
    REQUIRE(stream.GetState() == 0);
    std::array<s16, 480> output{};
    output.fill(99);
    context.sink.last->ProcessAudioOutAndRender(output, 240);
    REQUIRE(std::ranges::all_of(output, [](s16 value) { return value == 0; }));
    stream.Pump();
    REQUIRE(stream.GetCounters().faults == 1);
}

TEST_CASE("sysmix rejects unmapped payload and tampered cursors without admission", "[sysmix]") {
    Context context;
    Cursor(context, 2048, 1);
    SysmixOutput stream{context.system, *context.process, context.sink, 1, Context::Base,
                        Context::Size, 0x24};
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
    REQUIRE(stream.SetEngaged(true).IsSuccess());
    SECTION("missing payload page") {
        context.Unmap(Context::Base + 0x5000);
    }
    SECTION("producer outside physical modulus") {
        Cursor(context, 2048, 2049);
    }
    SECTION("consumer changed by guest") {
        Cursor(context, 0, 1);
    }
    stream.Pump();
    REQUIRE(stream.GetCounters().faults == 1);
    REQUIRE(stream.GetCounters().accepted_frames == 0);
    REQUIRE(context.sink.last->GetQueueSize() == 0);
}

TEST_CASE("sysmix close releases host stream and allows reopening without stale samples", "[sysmix]") {
    Context context;
    Cursor(context, 0, 1);
    Sample(context, 0, 0.5f, 0.5f);
    {
        SysmixOutput stream{context.system, *context.process, context.sink, 1, Context::Base,
                            Context::Size, 0x24};
        REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
        REQUIRE(stream.SetEngaged(true).IsSuccess());
        stream.Pump();
        REQUIRE(stream.SetEngaged(false).IsSuccess());
        REQUIRE(context.sink.last->GetQueueSize() == 1);
        stream.Close();
        REQUIRE(context.sink.last == nullptr);
        const auto ticks = stream.GetCounters().ticks;
        stream.Pump();
        REQUIRE(stream.GetCounters().ticks == ticks);
        Cursor(context, 0, 0);
        REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
        REQUIRE(context.sink.last->GetQueueSize() == 0);
    }
    REQUIRE(context.sink.closed == 2);
    REQUIRE(context.process->GetReferenceCount() == 1);
}

TEST_CASE("sysmix requires initialized and successfully started host output", "[sysmix]") {
    Context context;
    context.sink.ready = false;
    SysmixOutput stream{context.system, *context.process, context.sink, 1, Context::Base,
                        Context::Size, 0x24};
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsFailure());
    REQUIRE(context.sink.last == nullptr);
    context.sink.ready = true;
    context.sink.fail_start = true;
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
    REQUIRE(stream.SetEngaged(true).IsFailure());
    stream.Pump();
    REQUIRE(stream.GetCounters().admitted_blocks == 0);
}

TEST_CASE("sysmix conversion handles gain clipping and nonfinite values", "[sysmix]") {
    Context context;
    Cursor(context, 0, 2);
    Sample(context, 0, 2.0f, -2.0f);
    Sample(context, 1, std::numeric_limits<f32>::quiet_NaN(),
           std::numeric_limits<f32>::infinity());
    SysmixOutput stream{context.system, *context.process, context.sink, 1, Context::Base,
                        Context::Size, 0x24};
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
    REQUIRE(stream.SetGain(0.25f, 1).IsFailure());
    REQUIRE(stream.SetGain(0.25f, 0).IsSuccess());
    REQUIRE(stream.SetEngaged(true).IsSuccess());
    stream.Pump();
    std::array<s16, 480> output{};
    context.sink.last->ProcessAudioOutAndRender(output, 240);
    REQUIRE(output[0] == 16384);
    REQUIRE(output[1] == -16384);
    REQUIRE(output[2] == 0);
    REQUIRE(output[3] == 0);
}

TEST_CASE("sysmix child lifetime retains registry after user session closes", "[sysmix]") {
    Context context;
    using namespace Service::Audio;
    auto registry = std::make_shared<Nx2SysmixStreams>(context.system);
    std::weak_ptr<Nx2SysmixStreams> weak = registry;
    auto output = std::make_shared<SysmixOutput>(context.system, *context.process, context.sink,
                                                1, Context::Base, Context::Size, 0x24);
    registry->streams.emplace(1, output);
    auto child = std::make_shared<INx2SysmixStreamSession>(context.system, registry, output, 1);
    auto user = std::make_shared<INx2SysmixUserSession>(context.system, registry, &*context.process);
    registry.reset();
    user.reset();
    REQUIRE_FALSE(weak.expired());
    REQUIRE(output->Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
    REQUIRE(weak.lock()->SetEngaged(1, true, nullptr).IsFailure());
    REQUIRE(weak.lock()->SetEngaged(1, true, &*context.process).IsSuccess());
    context.system.CoreTiming().cpu_ticks += Common::WallClock::CPUTickFreq / 200;
    context.system.CoreTiming().Advance();
    REQUIRE(output->GetCounters().admitted_blocks == 1);
    // Drain the silent block, so the empty ring underruns again instead of waiting on the host.
    std::array<s16, 480> drained{};
    context.sink.last->ProcessAudioOutAndRender(drained, 240);
    context.system.CoreTiming().cpu_ticks += Common::WallClock::CPUTickFreq / 10;
    context.system.CoreTiming().Advance();
    REQUIRE(output->GetCounters().ticks == 2);
    REQUIRE(output->GetCounters().admitted_blocks == 2);
    child.reset();
    REQUIRE(weak.expired());
    REQUIRE(output->GetState() == 0);
    REQUIRE(context.sink.last == nullptr);
    context.system.CoreTiming().cpu_ticks += Common::WallClock::CPUTickFreq / 200;
    context.system.CoreTiming().Advance();
    REQUIRE(output->GetCounters().ticks == 2);
}

TEST_CASE("sysmix does not report successful disengage after backend stop failure", "[sysmix]") {
    Context context;
    SysmixOutput stream{context.system, *context.process, context.sink, 1, Context::Base,
                        Context::Size, 0x24};
    REQUIRE(stream.Open(5, 2, 48000, 2048, 0, 0).IsSuccess());
    REQUIRE(stream.SetEngaged(true).IsSuccess());
    context.sink.last->fail_stop = true;
    REQUIRE(stream.SetEngaged(false).IsFailure());
    REQUIRE(stream.GetState() == 0);
    REQUIRE(stream.GetCounters().faults == 1);
    stream.Pump();
    REQUIRE(stream.GetCounters().admitted_blocks == 0);
}
