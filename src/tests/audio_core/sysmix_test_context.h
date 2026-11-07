// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "audio_core/sink/sink.h"
#include "common/page_table.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "core/memory.h"
#include "tests/core/system_test_context.h"

namespace SysmixTests {

class Stream final : public AudioCore::Sink::SinkStream {
public:
    Stream(Core::System& emu_system, AudioCore::Sink::StreamType stream_type, u32 channels,
           bool ready_)
        : SinkStream{emu_system, stream_type}, ready{ready_} {
        device_channels = channels;
    }
    bool IsReady() const override {
        return ready && !failed;
    }
    void Start(bool = false) override {
        if (IsReady() && !fail_start) {
            paused = false;
        }
    }
    void Stop() override {
        SignalPause();
        if (fail_stop) {
            MarkFaulted();
        }
    }
    bool TryAppendOutputBlock(AudioCore::Sink::SinkBuffer& buffer, std::span<s16> samples) override {
        const bool accepted = SinkStream::TryAppendOutputBlock(buffer, samples);
        if (accepted && on_admit) {
            on_admit();
        }
        return accepted;
    }
    bool ready;
    bool fail_start{};
    bool fail_stop{};
    std::function<void()> on_admit;
};

class Sink final : public AudioCore::Sink::Sink {
public:
    AudioCore::Sink::SinkStream* AcquireSinkStream(Core::System& system, u32 channels,
        const std::string&, AudioCore::Sink::StreamType type) override {
        auto stream = std::make_unique<Stream>(system, type, host_channels, ready);
        stream->SetSystemChannels(channels);
        stream->fail_start = fail_start;
        last = stream.get();
        streams.push_back(std::move(stream));
        return last;
    }
    void CloseStream(AudioCore::Sink::SinkStream* stream) override {
        std::erase_if(streams, [stream](const auto& value) { return value.get() == stream; });
        if (last == stream) {
            last = nullptr;
        }
        ++closed;
    }
    void CloseStreams() override {
        streams.clear();
        last = nullptr;
    }
    f32 GetDeviceVolume() const override { return 1.0f; }
    void SetDeviceVolume(f32) override {}
    void SetSystemVolume(f32) override {}
    Stream* last{};
    u32 host_channels{2};
    bool ready{true};
    bool fail_start{};
    u32 closed{};
    std::vector<std::unique_ptr<Stream>> streams;
};

class Context {
public:
    static constexpr u64 Base = 0x4000;
    static constexpr u64 Size = 0x6000;

    Context() {
        try {
            Settings::values.volume.SetValue(100);
            Settings::values.audio_muted.SetValue(false);
            system.Initialize();
            dummy_thread.emplace(system.Kernel());
            if (Kernel::KThread::InitializeDummyThread(system, &*dummy_thread, nullptr).IsFailure()) {
                throw std::runtime_error("Cannot initialize sysmix test thread");
            }
            system.Kernel().SetCurrentEmuThread(&*dummy_thread);
            system.Kernel().Initialize();
            kernel_initialized = true;
            process.emplace(system.Kernel());
            Kernel::KAutoObject::Create(&*process);
            auto& table = process->GetPageTable().GetImpl();
            table.Resize(20, 12);
            storage = std::make_unique<Backing>();
            backing = storage->bytes.data();
            for (u64 offset = 0; offset < Size; offset += 0x1000) {
                table.entries.GetAndFault((Base + offset) >> 12).Store(
                    false, Common::PageType::Memory, 1,
                    reinterpret_cast<uintptr_t>(backing + offset) - (Base + offset));
            }
            process->GetMemory().SetCurrentPageTable(*process);
        } catch (...) {
            Cleanup();
            throw;
        }
    }
    ~Context() {
        Cleanup();
    }
    Core::Memory::Memory& Memory() { return process->GetMemory(); }
    void Unmap(u64 address) {
        process->GetPageTable().GetImpl().entries.GetAndFault(address >> 12).Store(
            false, Common::PageType::Unmapped, 1, 0);
    }

    const u8 previous_volume = Settings::values.volume.GetValue();
    const bool previous_muted = Settings::values.audio_muted.GetValue();
    ScopedSystemTestContext environment;
    Core::System system;
    std::optional<Kernel::KProcess> process;
    Sink sink;
    u8* backing{};

private:
    void Cleanup() {
        sink.CloseStreams();
        if (process) {
            system.Kernel().UnregisterKernelObject(&*process);
            process.reset();
        }
        if (kernel_initialized) {
            system.CoreTiming().ClearPendingEvents();
            system.Kernel().Shutdown();
            kernel_initialized = false;
        }
        system.Kernel().SetCurrentEmuThread(nullptr);
        if (dummy_thread) {
            system.Kernel().UnregisterKernelObject(&*dummy_thread);
            dummy_thread.reset();
        }
        Settings::values.volume.SetValue(previous_volume);
        Settings::values.audio_muted.SetValue(previous_muted);
    }

    struct alignas(0x1000) Backing {
        std::array<u8, Size> bytes{};
    };
    std::unique_ptr<Backing> storage;
    std::optional<Kernel::KThread> dummy_thread;
    bool kernel_initialized{};
};

} // namespace SysmixTests
