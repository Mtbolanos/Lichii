// SPDX-FileCopyrightText: Copyright 2026 Lichii Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <vector>

namespace Common::FS::ProfileMigration {

enum class Strategy { Copy, Move, Link };
enum class State { Unresolved, Pending, Complete, Fresh };

struct Root {
    std::filesystem::path source;
    std::filesystem::path destination;
    /// This root's strategy, else the plan's (Fritchii: copy most roots, link a big nand or
    /// sdmc when the copy would not fit).
    std::optional<Strategy> strategy{};
};

using LinkFunction =
    std::function<bool(const std::filesystem::path&, const std::filesystem::path&)>;

struct Plan {
    std::vector<Root> roots;
    Strategy strategy{Strategy::Copy};
    std::filesystem::path state_dir;
    LinkFunction create_link;
};

using Prepare = std::function<void(const std::vector<Root>&)>;

State GetState(const std::filesystem::path& state_dir);
bool NeedsMigration(const std::filesystem::path& state_dir);
void MarkFresh(const std::filesystem::path& state_dir);
void Recover(const std::filesystem::path& state_dir);
void RunImport(const Plan& plan, const Prepare& prepare = {});

} // namespace Common::FS::ProfileMigration
