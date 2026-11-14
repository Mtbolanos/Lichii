// SPDX-FileCopyrightText: Copyright 2026 Lichii contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <optional>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "common/scope_exit.h"
#include "core/hle/service/server_manager.h"
#include "core/hle/service/sockets/sfdnsres.h"
#include "core/hle/service/sockets/sockets.h"
#include "tests/audio_core/sysmix_test_context.h"

TEST_CASE("NameInfo13 HLE returns numeric outputs and rejects malformed requests atomically",
          "[nameinfo][ipc]") {
    // Reuse the existing mapped-memory/kernel test fixture; no ROM, frontend or host DNS.
    SysmixTests::Context context;
    const auto previous_auto_stub = Settings::values.use_auto_stub.GetValue();
    Settings::values.use_auto_stub.SetValue(true);
    SCOPE_EXIT {
        Settings::values.use_auto_stub.SetValue(previous_auto_stub);
    };
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
        kernel.RunOnHostCoreProcess("nameinfo-ipc-test", [&] { server.LoopProcess(); }));
    auto manager = std::make_shared<Service::SessionRequestManager>(kernel, server);
    Service::Sockets::SFDNSRES service{context.system};

    const auto query = [&](u32 raw_bytes, u32 address_bytes, u32 flags, u32 options_count,
                           u32 options_bytes, u32 host_capacity, u32 service_capacity,
                           bool map_options = false) {
        std::fill_n(context.backing, SysmixTests::Context::Size, u8{0xa5});
        const std::array<u8, 16> address{16, 2, 0x1e, 0x61, 192, 168, 1, 1};
        std::memcpy(context.backing, address.data(), address.size());

        std::array<u32_le, IPC::COMMAND_BUFFER_LENGTH> words{};
        IPC::CommandHeader header{};
        header.type.Assign(IPC::CommandType::Request);
        header.enable_handle_descriptor.Assign(1);
        header.num_buf_x_descriptors.Assign(1);
        header.num_buf_a_descriptors.Assign(2);
        header.num_buf_b_descriptors.Assign(2);
        // Descriptors end at word19; CMIF starts at word20, raw starts at24.
        header.data_size.Assign(5 + raw_bytes / 4);
        std::memcpy(words.data(), &header, sizeof(header));
        IPC::HandleDescriptorHeader handles{};
        handles.send_current_pid.Assign(1);
        std::memcpy(words.data() + 2, &handles, sizeof(handles));
        const auto ab = [&](std::size_t offset, u32 size, u32 address_) {
            IPC::BufferDescriptorABW buffer{};
            buffer.size_bits_0_31 = size;
            buffer.address_bits_0_31 = address_;
            std::memcpy(words.data() + offset, &buffer, sizeof(buffer));
        };
        // AutoSelect options: X0 (not X1), A1 placeholder. It must never be ignored.
        IPC::BufferDescriptorX options{};
        options.size.Assign(map_options ? 0 : options_bytes);
        options.address_bits_0_31 = SysmixTests::Context::Base + 0x300;
        std::memcpy(words.data() + 5, &options, sizeof(options));
        ab(7, address_bytes, SysmixTests::Context::Base);
        ab(10, map_options ? options_bytes : 0, SysmixTests::Context::Base + 0x300);
        ab(13, host_capacity, SysmixTests::Context::Base + 0x100);
        ab(16, service_capacity, SysmixTests::Context::Base + 0x200);
        words[20] = 0x49434653;
        words[22] = 13;
        words[24] = flags;
        words[25] = 0x12345678; // Not options_count; do not require this field to be zero.
        words[26] = options_count;

        Service::HLERequestContext request{kernel, context.Memory(), nullptr, &client};
        request.SetSessionRequestManager(manager);
        REQUIRE(request.PopulateFromIncomingCommandBuffer(words.data()).IsSuccess());
        service.InvokeRequest(request);
        REQUIRE(request.CommandBuffer()[4] == 0x4f434653);
        REQUIRE(request.CommandBuffer()[6] == ResultSuccess.raw);
        CHECK(request.CommandBuffer()[8] != 0xa5a5a5a5);
        CHECK(request.CommandBuffer()[9] == 0);  // h_errno
        CHECK(request.CommandBuffer()[10] == 0); // errno
        return static_cast<Service::Sockets::GetAddrInfoError>(request.CommandBuffer()[8]);
    };

    using Service::Sockets::GetAddrInfoError;
    SECTION("numeric host only and host plus service") {
        const auto host_result = query(24, 16, 2, 0, 0, 12, 0);
        REQUIRE(host_result == GetAddrInfoError::SUCCESS);
        CHECK(std::string{reinterpret_cast<const char*>(context.backing + 0x100), 11} ==
              "192.168.1.1");
        CHECK(context.backing[0x10b] == 0);
        CHECK(context.backing[0x10c] == 0xa5);
        CHECK(context.backing[0x200] == 0xa5);
        const auto service_result = query(24, 16, 2 | 8, 0, 0, 12, 5);
        REQUIRE(service_result == GetAddrInfoError::SUCCESS);
        CHECK(std::string{reinterpret_cast<const char*>(context.backing + 0x200), 4} == "7777");
        CHECK(context.backing[0x204] == 0);
        CHECK(context.backing[0x205] == 0xa5);
    }
    SECTION("rejections leave both output buffers untouched") {
        const auto rejected = [&](u32 raw, u32 address, u32 flags, u32 count, u32 options, u32 host,
                                  u32 port, GetAddrInfoError expected) {
            const auto result = query(raw, address, flags, count, options, host, port);
            REQUIRE(result == expected);
            CHECK(context.backing[0x100] == 0xa5);
            CHECK(context.backing[0x200] == 0xa5);
        };
        rejected(20, 16, 2, 0, 0, 12, 0, GetAddrInfoError::FAIL);
        rejected(24, 15, 2, 0, 0, 12, 0, GetAddrInfoError::FAIL);
        rejected(24, 257, 2, 0, 0, 12, 0, GetAddrInfoError::FAIL);
        rejected(24, 16, 2, 1, 20, 12, 0, GetAddrInfoError::FAIL);
        rejected(24, 16, 2, 0, 20, 12, 0, GetAddrInfoError::FAIL);
        rejected(24, 16, 2, 0, 0, 11, 0, GetAddrInfoError::OVERFLOW_);
        rejected(24, 16, 2 | 8, 0, 0, 12, 4, GetAddrInfoError::OVERFLOW_);
        rejected(24, 16, 0x20, 0, 0, 12, 0, GetAddrInfoError::BADFLAGS);
        const auto map_option_result = query(24, 16, 2, 0, 20, 12, 0, true);
        REQUIRE(map_option_result == GetAddrInfoError::FAIL);
        CHECK(context.backing[0x100] == 0xa5);
        CHECK(context.backing[0x200] == 0xa5);
    }
    SECTION("implemented command does not require auto stub") {
        Settings::values.use_auto_stub.SetValue(false);
        const auto result = query(24, 16, 2, 0, 0, 12, 0);
        REQUIRE(result == GetAddrInfoError::SUCCESS);
        CHECK(std::string{reinterpret_cast<const char*>(context.backing + 0x100), 11} ==
              "192.168.1.1");
        CHECK(context.backing[0x10b] == 0);
    }
}
