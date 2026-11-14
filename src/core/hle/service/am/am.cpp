// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/hle/service/am/am.h"
#include "core/hle/service/am/button_poller.h"
#include "core/hle/service/am/event_observer.h"
#include "core/hle/service/am/service/all_system_applet_proxies_service.h"
#include "core/hle/service/am/service/application_proxy_service.h"
#include "core/hle/service/am/window_system.h"
#include "core/hle/service/server_manager.h"

namespace Service::AM {

void LoopProcess(Core::System& system) {
    WindowSystem window_system(system);
    ButtonPoller button_poller(system, window_system);
    EventObserver event_observer(system, window_system);

    auto server_manager = std::make_unique<ServerManager>(system);

    server_manager->RegisterNamedService(
        "appletAE", std::make_shared<IAllSystemAppletProxiesService>(system, window_system));
    server_manager->RegisterNamedService(
        "appletOE", std::make_shared<IApplicationProxyService>(system, window_system));
    // The Switch 2 graphics system (nx2:gsv, from ISelfController) and every session it hands out
    // (vi2, vdecsvc, syncpt, gpu, nvsrv) are served here. vi2's PresentClientBuffer blocks until
    // the frame's fence and the next vblank (~14 ms of each 16.7 ms frame at 60 fps), and with one
    // thread every other request queued behind it: Ocarina of Time's movie thread got its decoder
    // images back at ~33 fps while presenting at 60, and the video fell ~10 s behind its audio.
    // Each service keeps its own lock, as with bsdsocket's extra threads.
    server_manager->StartAdditionalHostThreads("am", 2);
    ServerManager::RunServer(std::move(server_manager));
}

} // namespace Service::AM
