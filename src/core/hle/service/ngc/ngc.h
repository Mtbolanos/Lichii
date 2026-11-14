// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>

namespace Core {
class System;
}

namespace Service {
class ServiceFrameworkBase;

namespace NGC {

std::shared_ptr<ServiceFrameworkBase> CreateStreamPlayUserService(Core::System& system);

void LoopProcess(Core::System& system);

} // namespace NGC
} // namespace Service
