// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <filesystem>
#include "common/fs/path_util.h"
#include "common/fs/profile_migration.h"
#include "frontend_common/profile_import.h"
#include "migration_worker.h"

MigrationWorker::MigrationWorker(const Emulator selected_emu_, const bool clear_shader_cache_,
                                 const MigrationStrategy strategy_)
    : QObject(), selected_emu(selected_emu_), clear_shader_cache(clear_shader_cache_),
      strategy(strategy_) {}

void MigrationWorker::process() {
    namespace fs = std::filesystem;
    namespace Migration = Common::FS::ProfileMigration;
    using namespace Common::FS;

    try {
        const auto source = GetLegacyPath(selected_emu.e_user_dir);
        const auto destination = GetEdenPath(EdenPath::EdenDir);
        Migration::Plan plan;
        plan.state_dir = FrontendCommon::ProfileImportStateDirectory();
        switch (strategy) {
        case MigrationStrategy::Copy:
            plan.strategy = Migration::Strategy::Copy;
            break;
        case MigrationStrategy::Move:
            plan.strategy = Migration::Strategy::Move;
            break;
        case MigrationStrategy::Link:
            plan.strategy = Migration::Strategy::Link;
            break;
        }
        plan.roots.push_back({source, destination});
        const auto add_root = [&](EmuPath legacy, EdenPath current) {
            const auto& target = GetEdenPath(current);
            for (const auto& root : plan.roots) {
                const auto relative = target.lexically_relative(root.destination);
                if (!relative.empty() && *relative.begin() != "..") {
                    return;
                }
            }
            const auto& old = GetLegacyPath(legacy);
            if (fs::is_directory(old)) {
                plan.roots.push_back({old, target});
            }
        };
        add_root(selected_emu.e_config_dir, EdenPath::ConfigDir);
        add_root(selected_emu.e_cache_dir, EdenPath::CacheDir);

        Migration::RunImport(plan, [&](const std::vector<Migration::Root>& staged) {
            const auto staged_path = [&](const fs::path& target) {
                for (std::size_t i = 0; i < plan.roots.size(); ++i) {
                    const auto relative = target.lexically_relative(plan.roots[i].destination);
                    if (!relative.empty() && *relative.begin() != "..") {
                        return staged[i].destination / relative;
                    }
                }
                return fs::path{};
            };
            const auto config = staged_path(GetEdenPath(EdenPath::ConfigDir));
            if (!config.empty()) {
                FrontendCommon::RemapImportedConfig(config, source, destination);
            }
            const auto shader = staged_path(GetEdenPath(EdenPath::ShaderDir));
            if (clear_shader_cache && !shader.empty()) {
                fs::remove_all(shader);
                fs::create_directories(shader);
            }
        });

        QString message = tr("Data was imported successfully. Existing Lichii data, if any, "
                             "was kept in backup directories beside the new profile.");
        if (strategy == MigrationStrategy::Link) {
            message += tr("\n\nData and settings are shared with %1. The source shader cache "
                          "was not cleared.")
                           .arg(selected_emu.name());
        } else if (strategy == MigrationStrategy::Move) {
            message += tr("\n\nThe old profile was retained in backup directories at its old "
                          "location. Keep these backups until you have checked your saves.");
        } else {
            message += tr("\n\nThe original %1 profile is unchanged. Keep it until you have "
                          "checked your saves.")
                           .arg(selected_emu.name());
        }
        emit finished(message, PathToUTF8String(source));
    } catch (const std::exception& exception) {
        emit error(tr("Import did not complete. Your source and previous destination data "
                      "have been kept. Close Lichii and launch it again to retry or recover.\n\n%1")
                       .arg(QString::fromUtf8(exception.what())));
    }
}
