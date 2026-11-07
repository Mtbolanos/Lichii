// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <optional>
#include <thread>
#include <catch2/catch_test_macros.hpp>
#include "common/scope_exit.h"
#include "core/hle/service/audio/nx2_audio_device_manager.h"
#include "core/hle/service/server_manager.h"
#include "tests/audio_core/sysmix_test_context.h"

TEST_CASE("Switch 2 microphone format query accepts input functionality", "[sysmix][microphone]") {
    SysmixTests::Context context;
    auto& kernel = context.system.Kernel();
    Kernel::KThread client{kernel};
    Kernel::KAutoObject::Create(&client);
    REQUIRE(Kernel::KThread::InitializeDummyThread(context.system, &client, &*context.process)
                .IsSuccess());
    SCOPE_EXIT {
        context.process->UnregisterThread(kernel, &client);
        context.process->Close(kernel);
        kernel.UnregisterKernelObject(&client);
    };
    std::optional<std::jthread> server_thread;
    Service::ServerManager server{context.system};
    server_thread.emplace(
        kernel.RunOnHostCoreProcess("microphone-ipc-test", [&] { server.LoopProcess(); }));
    auto manager = std::make_shared<Service::SessionRequestManager>(kernel, server);
    auto streams = std::make_shared<Service::Audio::Nx2SysmixStreams>(context.system);
    Service::Audio::INx2AudioVirtualDeviceManager service{context.system, streams};

    const auto query = [&](u32 channels, u32 functionality) {
        std::array<u32_le, IPC::COMMAND_BUFFER_LENGTH> words{};
        IPC::CommandHeader header{};
        header.type.Assign(IPC::CommandType::Request);
        header.data_size.Assign(12);
        std::memcpy(words.data(), &header, sizeof(header));
        words[4] = 0x49434653;
        words[6] = 300;
        words[8] = 2;
        words[9] = channels;
        words[10] = 48000;
        words[11] = functionality;
        Service::HLERequestContext request{kernel, context.Memory(), nullptr, &client};
        request.SetSessionRequestManager(manager);
        REQUIRE(request.PopulateFromIncomingCommandBuffer(words.data()).IsSuccess());
        service.InvokeRequest(request);
        REQUIRE(request.CommandBuffer()[4] == 0x4f434653);
        return request.CommandBuffer()[6];
    };
    REQUIRE(query(1, 2) == ResultSuccess.raw);
    REQUIRE(query(2, 2) == ResultSuccess.raw);
    REQUIRE(query(1, 1) == Service::Audio::ResultNotSupported.raw);
    REQUIRE(query(2, 3) == Service::Audio::ResultNotSupported.raw);
}
