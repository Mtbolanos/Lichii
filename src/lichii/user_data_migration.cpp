// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <QMessageBox>
#include <QPushButton>
#include <QString>
#include <QTranslator>
#include "common/fs/path_util.h"
#include "common/fs/profile_migration.h"
#include "common/fs/symlink.h"
#include "frontend_common/profile_import.h"
#include "qt_common/qt_string_lookup.h"
#include "user_data_migration.h"
#include "lichii/migration_dialog.h"

// Needs to be included at the end due to https://bugreports.qt.io/browse/QTBUG-73263
#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <QButtonGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QEventLoop>
#include <QGuiApplication>
#include <QProgressDialog>
#include <QRadioButton>
#include <QThread>

UserDataMigrator::UserDataMigrator(QMainWindow* main_window) {
    // NOTE: Logging is not initialized yet, do not produce logs here.

    if (FrontendCommon::NeedsLegacyProfileImport()) {
        namespace Migration = Common::FS::ProfileMigration;
        const auto state_dir = FrontendCommon::ProfileImportStateDirectory();
        if (Migration::GetState(state_dir) == Migration::State::Pending) {
            const auto answer = QMessageBox::question(
                main_window, QObject::tr("Recover unfinished import"),
                QObject::tr("An earlier import did not finish. Recover the previous profile "
                            "and retry? No source or backup data will be deleted."),
                QMessageBox::Retry | QMessageBox::Close, QMessageBox::Retry);
            if (answer != QMessageBox::Retry) {
                throw std::runtime_error(
                    "Import recovery was postponed. Relaunch Lichii to retry.");
            }
            Migration::Recover(state_dir);
        }
        ShowMigrationPrompt(main_window);
    }
}

void UserDataMigrator::ShowMigrationPrompt(QMainWindow* main_window) {
    namespace fs = std::filesystem;
    using namespace QtCommon::StringLookup;

    MigrationDialog migration_prompt;
    migration_prompt.setWindowTitle(QObject::tr("Migration"));

    // mutually exclusive
    QButtonGroup* group = new QButtonGroup(&migration_prompt);

    // MACRO MADNESS, indeed

#define BUTTON(clazz, name, text, tooltip, checkState)                                             \
    clazz* name = new clazz(&migration_prompt);                                                    \
    name->setText(text);                                                                           \
    name->setToolTip(Lookup(tooltip));                                                             \
    name->setChecked(checkState);                                                                  \
    migration_prompt.addBox(name);

    BUTTON(QCheckBox, clear_shaders, QObject::tr("Clear Shader Cache"), MigrationTooltipClearShader,
           true)

#define RADIO(name, text, tooltip, checkState, strategy)                                           \
    BUTTON(QRadioButton, name, text, tooltip, checkState)                                          \
    group->addButton(name, static_cast<int>(MigrationWorker::MigrationStrategy::strategy));

    RADIO(keep_old, QObject::tr("Keep Old Data"), MigrationTooltipKeepOld, true, Copy)
    RADIO(clear_old, QObject::tr("Move Old Data (retain backup)"), MigrationTooltipClearOld, false,
          Move)
    RADIO(link_old, QObject::tr("Link Old Directory"), MigrationTooltipLinkOld, false, Link)

#undef RADIO
#undef BUTTON

    std::vector<Emulator> found{};

    for (const Emulator& emu : legacy_emus)
        if (fs::is_directory(Common::FS::GetLegacyPath(emu.e_user_dir)))
            found.emplace_back(emu);

    if (found.empty()) {
        return;
    }

    auto* source = new QComboBox(&migration_prompt);
    for (const Emulator& emu : found) {
        source->addItem(emu.name());
    }
    migration_prompt.addBox(source);
    const auto update_cache_default = [&] {
        const Emulator& emu = found.at(source->currentIndex());
        clear_shaders->setChecked(emu.e_user_dir != Common::FS::LegacyEdenDir);
        clear_shaders->setEnabled(!link_old->isChecked());
        // A profile folder that is a link (e.g. a junction to another drive) can be copied or
        // linked, but Move would rename the real folder behind the link, so it is not offered.
        bool linked = false;
        for (const auto dir : {emu.e_user_dir, emu.e_config_dir, emu.e_cache_dir}) {
            linked = linked || Common::FS::IsSymlink(Common::FS::GetLegacyPath(dir));
        }
        if (linked && clear_old->isChecked()) {
            keep_old->setChecked(true);
        }
        clear_old->setEnabled(!linked);
    };
    QObject::connect(source, &QComboBox::currentIndexChanged, &migration_prompt,
                     update_cache_default);
    QObject::connect(link_old, &QRadioButton::toggled, &migration_prompt,
                     [=](bool linked) { clear_shaders->setEnabled(!linked); });
    update_cache_default();
    migration_prompt.setText(
        QObject::tr("Import an existing emulator profile into Lichii?\n"
                    "Copy keeps the original profile unchanged and is recommended.\n"
                    "Move retains a recovery backup and requires space for a full copy.\n"
                    "Link shares settings and saves with the source emulator.\n"
                    "Existing Lichii data is kept in backup directories, not merged or deleted.\n\n"
                    "Close the source emulator, then choose its profile below.\n"
                    "Close this window to decide on the next launch."));
    migration_prompt.addButton(QObject::tr("Import"));
    auto* fresh = migration_prompt.addButton(QObject::tr("Start fresh"), true);

    migration_prompt.exec();

    QAbstractButton* button = migration_prompt.clickedButton();

    if (button == nullptr) {
        throw std::runtime_error("Import was postponed. Relaunch Lichii to choose a profile.");
    }
    if (button == fresh) {
        Common::FS::ProfileMigration::MarkFresh(FrontendCommon::ProfileImportStateDirectory());
        return ShowMigrationCancelledMessage(main_window);
    }

    MigrationWorker::MigrationStrategy strategy =
        static_cast<MigrationWorker::MigrationStrategy>(group->checkedId());

    selected_emu = found.at(source->currentIndex());

    MigrateUserData(main_window, clear_shaders->isEnabled() && clear_shaders->isChecked(),
                    strategy);
    if (!migrated) {
        throw std::runtime_error("Import failed. Relaunch Lichii to retry or recover.");
    }
}

void UserDataMigrator::ShowMigrationCancelledMessage(QMainWindow* main_window) {
    QMessageBox::information(main_window, QObject::tr("Migration"),
                             QObject::tr("Lichii will use its own profile. Your old emulator "
                                         "data has not been changed."),
                             QMessageBox::Ok);
}

void UserDataMigrator::MigrateUserData(QMainWindow* main_window, const bool clear_shader_cache,
                                       const MigrationWorker::MigrationStrategy strategy) {
    // Create a dialog to let the user know it's migrating
    QProgressDialog* progress = new QProgressDialog(main_window);
    progress->setWindowTitle(QObject::tr("Migrating"));
    progress->setLabelText(QObject::tr("Migrating, this may take a while..."));
    progress->setRange(0, 0);
    progress->setCancelButton(nullptr);
    progress->setWindowFlag(Qt::WindowCloseButtonHint, false);
    progress->setWindowModality(Qt::WindowModality::ApplicationModal);

    QThread* thread = new QThread(main_window);
    MigrationWorker* worker = new MigrationWorker(selected_emu, clear_shader_cache, strategy);
    worker->moveToThread(thread);

    thread->connect(thread, &QThread::started, worker, &MigrationWorker::process);

    // The worker can't be stopped, so wait for it rather than for the dialog, and show the dialog
    // again if Esc closes it before the worker is done.
    QEventLoop loop;
    bool worker_done = false;
    QString failure;
    QObject::connect(progress, &QDialog::rejected, progress, [progress, &worker_done] {
        if (!worker_done) {
            progress->show();
        }
    });

    thread->connect(
        worker, &MigrationWorker::finished, progress,
        [=, this, &loop, &worker_done](const QString& success_text, const std::string& path) {
            worker_done = true;
            progress->close();
            QMessageBox::information(main_window, QObject::tr("Migration"), success_text,
                                     QMessageBox::Ok);

            migrated = true;
            thread->quit();
            loop.quit();
        });

    thread->connect(worker, &MigrationWorker::finished, worker, &QObject::deleteLater);
    thread->connect(worker, &MigrationWorker::error, progress,
                    [=, &loop, &worker_done, &failure](const QString& message) {
                        worker_done = true;
                        failure = message;
                        progress->close();
                        thread->quit();
                        loop.quit();
                    });
    thread->connect(worker, &MigrationWorker::error, worker, &QObject::deleteLater);
    thread->connect(thread, &QThread::finished, thread, &QObject::deleteLater);

    thread->start();
    progress->show();
    loop.exec();
    thread->wait();
    progress->deleteLater();
    if (!failure.isEmpty()) {
        throw std::runtime_error(failure.toStdString());
    }
}
