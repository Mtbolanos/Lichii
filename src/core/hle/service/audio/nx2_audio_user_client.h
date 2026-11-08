// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "core/core.h"
#include "core/hle/service/audio/audio_in_manager.h"
#include "core/hle/service/audio/audio_out_manager.h"
#include "core/hle/service/audio/audio_renderer_manager.h"
#include "core/hle/service/ipc_helpers.h"
#include "core/hle/service/service.h"
#include "core/hle/service/sm/sm.h"

namespace Service::Audio {

// Switch 2 "aud:u" (nn::audio::CreateUserClientSessionByHipc). The SDK reaches the Switch 1
// audio interfaces through it: command 2 returns an object used exactly like
// IAudioRendererManager (vtable slot 1 = GetWorkBufferSize with a Switch 1 parameter block), so
// it is backed by Eden's audren:u implementation. Command 1 is nn::audio::OpenAudioOut's
// manager, called like IAudioOutManager (slot 1 = OpenAudioOut), backed by audout:u. The nx2: name
// keeps unknown commands auto-stubbed and logged.
class INx2AudioUserClient final : public ServiceFramework<INx2AudioUserClient> {
public:
    explicit INx2AudioUserClient(Core::System& system_)
        : ServiceFramework{system_, "nx2:aud:u"} {
        // clang-format off
        static const FunctionInfo functions[] = {
            {0, &INx2AudioUserClient::GetAudioInManager, "GetAudioInManager"},
            {1, &INx2AudioUserClient::GetAudioOutManager, "GetAudioOutManager (guessed)"},
            {2, &INx2AudioUserClient::GetAudioRendererManager, "GetAudioRendererManager (guessed)"},
        };
        // clang-format on
        RegisterHandlers(functions);
    }

private:
    // The managers must be the registered audin:u / audout:u / audren:u instances. A fresh
    // IAudioOutManager owns its own AudioCore::AudioOut::Manager, but AudioManager accepts only
    // the first registered buffer-release callback, so sessions opened through a second manager
    // never release buffers: the title queues 3 silent buffers and its audio thread stalls.
    template <typename T>
    std::shared_ptr<T> SharedManager(const char* name) {
        auto manager = system.ServiceManager().GetService<T>(name);
        if (!manager) {
            LOG_ERROR(Service_Audio, "Switch 2 aud:u: {} not registered, using a private manager",
                      name);
            manager = std::make_shared<T>(system);
        }
        return manager;
    }

    void GetAudioInManager(HLERequestContext& ctx) {
        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(ResultSuccess);
        rb.PushIpcInterface<IAudioInManager>(ctx, SharedManager<IAudioInManager>("audin:u"));
    }

    void GetAudioOutManager(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const u32 arg = rp.Pop<u32>();
        LOG_INFO(Service_Audio, "Switch 2 aud:u 1 arg={}, returning shared audout:u", arg);
        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(ResultSuccess);
        rb.PushIpcInterface<IAudioOutManager>(ctx, SharedManager<IAudioOutManager>("audout:u"));
    }

    void GetAudioRendererManager(HLERequestContext& ctx) {
        IPC::RequestParser rp{ctx};
        const u32 arg = rp.Pop<u32>();
        LOG_INFO(Service_Audio, "Switch 2 aud:u 2 arg={}, returning shared audren:u", arg);
        IPC::ResponseBuilder rb{ctx, 2, 0, 1};
        rb.Push(ResultSuccess);
        rb.PushIpcInterface<IAudioRendererManager>(
            ctx, SharedManager<IAudioRendererManager>("audren:u"));
    }
};

} // namespace Service::Audio
