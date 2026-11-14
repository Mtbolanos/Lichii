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
#include "frontend_common/profile_import.h"
#include "qt_common/qt_string_lookup.h"
#include "user_data_migration.h"
#include "yuzu/migration_dialog.h"

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
#include <QDateTime>
#include <QDialog>
#include <QDialogButtonBox>
#include <QLabel>
#include <QLocale>
#include <QStringList>
#include <QVBoxLayout>
#include <sstream>
#include <QDesktopServices>
#include <QUrl>
#include <array>
#include <chrono>
#include <optional>
#include "common/fs/fs_paths.h"

UserDataMigrator::UserDataMigrator(QMainWindow* main_window) {
    // NOTE: Logging is not initialized yet, do not produce logs here.

    // Help > Copy Data from Eden or Lichii (v5.8.1) left a request and restarted Fritchii.
    if (const auto request = FrontendCommon::TakeProfileImportRequest(); !request.empty()) {
        if (const auto choice = ProfileImportChoice::Parse(request)) {
            AutoImport(main_window, choice);
        }
        return;
    }
    if (!FrontendCommon::NeedsLegacyProfileImport()) {
        return;
    }
    // Fritchii imports on its own; zortos's choice dialog stays behind a switch for support.
    if (qEnvironmentVariableIntValue("EDEN_NX2_IMPORT_PROMPT") == 0) {
        AutoImport(main_window);
        return;
    }
    {
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
        clear_shaders->setChecked(found.at(source->currentIndex()).e_user_dir !=
                                  Common::FS::LegacyEdenDir);
        clear_shaders->setEnabled(!link_old->isChecked());
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

namespace {
namespace fs = std::filesystem;
namespace Migration = Common::FS::ProfileMigration;

struct OldProfile {
    QString name;
    fs::path user;   ///< The old data folder (saves, keys, mods).
    fs::path config; ///< Its config folder (inside `user` on Windows, XDG_CONFIG_HOME elsewhere).
    bool fritchii{}; ///< A previous Fritchii's (it has frizti.ini).
    fs::file_time_type used{fs::file_time_type::min()};
};

fs::file_time_type LastUsed(const fs::path& config) {
    std::error_code ec;
    const auto time = fs::last_write_time(config / "qt-config.ini", ec);
    return ec ? fs::file_time_type::min() : time;
}

/// When the profile last saved a game: its newest save file (falling back to qt-config.ini).
/// v5.8.1: a player who moved from an earlier Fritchii (eden, with frizti.ini) to Lichii 150+
/// (lichii) got the older eden data, because frizti.ini won outright; the newest saves win now.
fs::file_time_type LastSaved(const fs::path& user, const fs::path& config) {
    auto newest = fs::file_time_type::min();
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(user / NAND_DIR / "user" / "save", ec);
         !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (it->is_regular_file(ec)) {
            newest = std::max(newest, it->last_write_time(ec));
        }
    }
    return newest != fs::file_time_type::min() ? newest : LastUsed(config);
}

/// Every old profile folder there is, the one to suggest first: the newer of Eden and Lichii
/// by their newest saves, then the newest of the older emulators.
std::vector<OldProfile> ListOldProfiles();

/// The suggested old profile, or the one in `only` ("eden", "lichii"...).
std::optional<OldProfile> PickOldProfile(const std::string& only = {}) {
    const auto all = ListOldProfiles();
    for (const auto& profile : all) {
        if (only.empty() ||
            Common::FS::PathToUTF8String(profile.user.filename()) == only) {
            return profile;
        }
    }
    return std::nullopt;
}

std::vector<OldProfile> ListOldProfiles() {
    using namespace Common::FS;
    struct Candidate {
        const char* name;
        EmuPath user;
        EmuPath config;
        int rank; ///< 0: Eden and Lichii, 1: the older emulators
    };
    constexpr std::array candidates{
        Candidate{QT_TR_NOOP("Eden"), LegacyEdenDir, LegacyEdenConfigDir, 0},
        Candidate{QT_TR_NOOP("Lichii"), LichiiDir, LichiiConfigDir, 0},
        Candidate{QT_TR_NOOP("Citron"), CitronDir, CitronConfigDir, 1},
        Candidate{QT_TR_NOOP("Sudachi"), SudachiDir, SudachiConfigDir, 1},
        Candidate{QT_TR_NOOP("Suyu"), SuyuDir, SuyuConfigDir, 1},
        Candidate{QT_TR_NOOP("Yuzu"), YuzuDir, YuzuConfigDir, 1},
    };
    std::vector<std::pair<int, OldProfile>> found;
    for (const auto& candidate : candidates) {
        const auto& user = GetLegacyPath(candidate.user);
        if (!fs::is_directory(user)) {
            continue;
        }
        const auto& config = GetLegacyPath(candidate.config);
        found.emplace_back(candidate.rank,
                           OldProfile{QObject::tr(candidate.name), user, config,
                                      fs::exists(config / "frizti.ini"), LastSaved(user, config)});
    }
    std::ranges::stable_sort(found, [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first < b.first : a.second.used > b.second.used;
    });
    std::vector<OldProfile> result;
    for (auto& [rank, profile] : found) {
        result.push_back(std::move(profile));
    }
    return result;
}

QString LastPlayedText(fs::file_time_type time) {
    if (time == fs::file_time_type::min()) {
        return {};
    }
    const auto system = std::chrono::clock_cast<std::chrono::system_clock>(time);
    const auto seconds =
        std::chrono::duration_cast<std::chrono::seconds>(system.time_since_epoch()).count();
    return QLocale().toString(QDateTime::fromSecsSinceEpoch(seconds).date(), QLocale::ShortFormat);
}

std::uintmax_t FolderSize(const fs::path& path) {
    std::uintmax_t total = 0;
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(path, ec);
         !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (it->is_regular_file(ec)) {
            total += it->file_size(ec);
        }
    }
    return total;
}

QString Gigabytes(std::uintmax_t bytes) {
    return QString::number(static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0), 'f', 1);
}

struct ImportPlan {
    Migration::Plan plan;
    std::vector<QString> linked; ///< Folders linked instead of copied (not enough free space)
    std::uintmax_t copy_bytes{};
};

/// Settings, keys, mods, saves and the small user folders are copied; caches, logs, dumps and
/// screenshots are not (shaders are prepared again on the first boot). nand and sdmc are linked
/// instead when copying them would not fit.
ImportPlan PlanImport(const OldProfile& old, const ProfileImportChoice& choice) {
    using namespace Common::FS;
    ImportPlan result;
    result.plan.strategy = Migration::Strategy::Copy;
    result.plan.state_dir = FrontendCommon::ProfileImportStateDirectory();
    const auto& destination = GetEdenPath(EdenPath::EdenDir);
    const auto add = [&](const fs::path& source, const fs::path& target,
                         std::optional<Migration::Strategy> strategy = {}) {
        if (!fs::is_directory(source)) {
            return;
        }
        std::error_code ec;
        fs::remove(target, ec); // an empty folder made before the import, not a backup
        result.plan.roots.push_back({source, target, strategy});
    };
    // v5.8.2: only the parts the player ticked (mods are off unless ticked).
    std::vector<const char*> folders;
    if (choice.settings) {
        add(old.config, GetEdenPath(EdenPath::ConfigDir));
        folders.insert(folders.end(), {TAS_DIR, ICONS_DIR, POST_SHADER_DIR, POST_PRESET_DIR,
                                       LOSSLESS_DIR, STREAMLINE_DIR, FIDELITYFX_DIR});
    }
    if (choice.saves) {
        folders.insert(folders.end(), {KEYS_DIR, AMIIBO_DIR, PLAY_TIME_DIR});
    }
    if (choice.mods) {
        folders.push_back(LOAD_DIR);
    }
    for (const char* name : folders) {
        if (old.user / name != old.config) {
            add(old.user / name, destination / name);
        }
    }
    for (const auto& root : result.plan.roots) {
        result.copy_bytes += FolderSize(root.source);
    }
    std::error_code ec;
    const auto space = fs::space(destination.parent_path(), ec);
    std::uintmax_t free_bytes = ec ? 0 : space.available;
    free_bytes = free_bytes > result.copy_bytes ? free_bytes - result.copy_bytes : 0;
    for (const char* name : {NAND_DIR, SDMC_DIR}) {
        const auto source = old.user / name;
        if (!choice.saves || !fs::is_directory(source)) {
            continue;
        }
        const auto size = FolderSize(source);
        if (size + size / 5 <= free_bytes) {
            free_bytes -= size;
            result.copy_bytes += size;
            add(source, destination / name);
        } else {
            result.linked.push_back(QString::fromLatin1(name));
            add(source, destination / name, Migration::Strategy::Link);
        }
    }
    return result;
}

} // namespace

std::string ProfileImportChoice::Serialize() const {
    return folder + "\nsaves=" + (saves ? "1" : "0") + "\nsettings=" + (settings ? "1" : "0") +
           "\nmods=" + (mods ? "1" : "0") + "\n";
}

std::optional<ProfileImportChoice> ProfileImportChoice::Parse(const std::string& text) {
    ProfileImportChoice choice;
    std::istringstream in{text};
    std::string line;
    if (!std::getline(in, choice.folder) || choice.folder.empty() ||
        choice.folder.find_first_of("/\\.") != std::string::npos) {
        return std::nullopt;
    }
    // A v5.8.1 request names the folder only: everything as then, mods included.
    choice.mods = true;
    while (std::getline(in, line)) {
        const auto value = line.size() > 2 && line.back() == '1';
        if (line.starts_with("saves=")) {
            choice.saves = value;
        } else if (line.starts_with("settings=")) {
            choice.settings = value;
        } else if (line.starts_with("mods=")) {
            choice.mods = value;
        }
    }
    return choice;
}

std::optional<ProfileImportChoice> UserDataMigrator::AskImport(QWidget* parent, bool from_menu) {
    const auto profiles = ListOldProfiles();
    if (profiles.empty()) {
        return std::nullopt;
    }
    QDialog dialog(parent);
    dialog.setWindowTitle(QStringLiteral("Fritchii"));
    dialog.setWindowFlag(Qt::WindowContextHelpButtonHint, false);
    auto* layout = new QVBoxLayout(&dialog);
    layout->setSpacing(12);
    QFont big = dialog.font();
    big.setPointSizeF(big.pointSizeF() * 1.15);
    auto* text = new QLabel(
        from_menu ? QObject::tr("Copy your data from another folder?\nFritchii restarts and "
                                "copies it over its own. What it has now is kept as a backup.")
                  : QObject::tr("Fritchii keeps its data in its own folder now.\nCopy your data "
                                "from another emulator folder?"),
        &dialog);
    text->setFont(big);
    layout->addWidget(text);
    auto* source = new QComboBox(&dialog);
    source->setFont(big);
    for (const auto& profile : profiles) {
        const QString name =
            QString::fromStdString(Common::FS::PathToUTF8String(profile.user.filename()));
        const QString played = LastPlayedText(profile.used);
        source->addItem(played.isEmpty() ? name
                                         : QObject::tr("%1 (last played %2)").arg(name, played),
                        name);
    }
    layout->addWidget(source);
    const auto check = [&](const QString& label, bool on) {
        auto* box = new QCheckBox(label, &dialog);
        box->setFont(big);
        box->setChecked(on);
        layout->addWidget(box);
        return box;
    };
    auto* saves = check(QObject::tr("Saves and keys"), true);
    auto* settings = check(QObject::tr("Settings"), true);
    auto* mods = check(QObject::tr("Mods"), false);
    auto* buttons = new QDialogButtonBox(&dialog);
    auto* copy = buttons->addButton(QObject::tr("Copy"), QDialogButtonBox::AcceptRole);
    buttons->addButton(from_menu ? QObject::tr("Cancel") : QObject::tr("Start fresh"),
                       QDialogButtonBox::RejectRole);
    copy->setDefault(true);
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    const auto update = [=] {
        copy->setEnabled(saves->isChecked() || settings->isChecked() || mods->isChecked());
    };
    for (auto* box : {saves, settings, mods}) {
        QObject::connect(box, &QCheckBox::toggled, &dialog, update);
    }
    layout->addWidget(buttons);
    if (dialog.exec() != QDialog::Accepted) {
        return std::nullopt;
    }
    return ProfileImportChoice{
        .folder = source->currentData().toString().toStdString(),
        .saves = saves->isChecked(),
        .settings = settings->isChecked(),
        .mods = mods->isChecked(),
    };
}

void UserDataMigrator::AutoImport(QMainWindow* main_window,
                                  std::optional<ProfileImportChoice> request) {
    using namespace Common::FS;
    const auto state_dir = FrontendCommon::ProfileImportStateDirectory();
    const auto& destination = GetEdenPath(EdenPath::EdenDir);
    const QString destination_text = QString::fromStdString(PathToUTF8String(destination));
    if (Migration::GetState(state_dir) == Migration::State::Pending) {
        Migration::Recover(state_dir); // an import a crash or power cut interrupted
    }
    // Fritchii data that is already there (copied by hand, or an earlier import) is kept as is,
    // unless the player asked for a copy (Help > Copy Data from Eden or Lichii): then RunImport
    // keeps it in backup folders beside the new copy.
    const bool requested = request.has_value();
    if (!requested && fs::exists(GetEdenPath(EdenPath::ConfigDir) / "qt-config.ini")) {
        Migration::MarkFresh(state_dir);
        return;
    }
    // v5.8.2: the first launch asks before copying anything.
    const auto choice = requested ? request : AskImport(main_window, false);
    if (!choice) {
        Migration::MarkFresh(state_dir); // Help > Copy Data can still copy later
        return;
    }
    const auto old = PickOldProfile(choice->folder);
    if (!old) {
        Migration::MarkFresh(state_dir);
        return;
    }
    const QString old_text = QString::fromStdString(PathToUTF8String(old->user));
    // Messages name the folder (eden, lichii, yuzu...): an eden folder can be an earlier
    // Fritchii's, Eden's or an old Lichii's.
    const QString folder = QString::fromStdString(PathToUTF8String(old->user.filename()));

    QString failure;
    bool busy = false;
    const auto run = [&](const ImportPlan& import) {
        failure.clear();
        // The other emulator still running on that profile would change files mid-copy.
        if (const auto used = LastUsed(old->config);
            used != fs::file_time_type::min() &&
            fs::file_time_type::clock::now() - used < std::chrono::seconds{3}) {
            busy = true;
            failure = QObject::tr("Another emulator seems to be using the %1 folder.").arg(folder);
            return;
        }
        auto* progress = new QProgressDialog(main_window);
        progress->setWindowTitle(QStringLiteral("Fritchii"));
        progress->setLabelText(
            QObject::tr("Copying your data from the %1 folder to Fritchii's own folder (%2 GB)...")
                .arg(folder, Gigabytes(import.copy_bytes)));
        progress->setRange(0, 0);
        progress->setCancelButton(nullptr);
        progress->setWindowFlag(Qt::WindowCloseButtonHint, false);
        progress->setWindowModality(Qt::ApplicationModal);
        QEventLoop loop;
        bool done = false;
        QObject::connect(progress, &QDialog::rejected, progress, [progress, &done] {
            if (!done) {
                progress->show();
            }
        });
        QThread* thread = QThread::create([&] {
            try {
                Migration::RunImport(import.plan, [&](const std::vector<Migration::Root>& staged) {
                    for (std::size_t i = 0; i < import.plan.roots.size(); ++i) {
                        if (import.plan.roots[i].destination == GetEdenPath(EdenPath::ConfigDir)) {
                            FrontendCommon::RemapImportedConfig(staged[i].destination, old->user,
                                                                destination);
                        }
                    }
                });
#ifdef _WIN32
                // The player's switch file sits beside the old profile's folders.
                const auto switches = old->user / "nx2_switches.txt";
                if (fs::exists(switches) && !fs::exists(destination / "nx2_switches.txt")) {
                    fs::copy_file(switches, destination / "nx2_switches.txt");
                }
#endif
            } catch (const std::exception& exception) {
                failure = QString::fromUtf8(exception.what());
                if (failure.isEmpty()) {
                    failure = QStringLiteral("?");
                }
            }
        });
        QObject::connect(thread, &QThread::finished, &loop, [&] {
            done = true;
            loop.quit();
        });
        thread->start();
        progress->show();
        loop.exec();
        thread->wait();
        delete thread;
        progress->close();
        progress->deleteLater();
    };

    const ImportPlan import = PlanImport(*old, *choice);
    if (import.plan.roots.empty()) {
        Migration::MarkFresh(state_dir);
        return;
    }
    run(import);
    if (!failure.isEmpty() && !busy) {
        // RunImport put everything back; one more try (a file another program held, say).
        if (Migration::GetState(state_dir) == Migration::State::Pending) {
            Migration::Recover(state_dir);
        }
        run(import);
    }

    if (failure.isEmpty()) {
        migrated = true;
        QStringList parts;
        if (choice->saves) {
            parts << QObject::tr("saves and keys");
        }
        if (choice->settings) {
            parts << QObject::tr("settings");
        }
        if (choice->mods) {
            parts << QObject::tr("mods");
        }
        QString what = parts.join(QStringLiteral(", "));
        what[0] = what[0].toUpper();
        QString text = QObject::tr("Fritchii now keeps its data in its own folder:\n%1\n\n"
                                   "%2 copied from the %3 folder.")
                           .arg(destination_text, what, folder);
        if (!import.linked.empty()) {
            text += QObject::tr("\n\n%1 was too big to copy, so it is shared with the old folder. "
                                "Keep that folder.")
                        .arg(import.linked.size() == 1 ? import.linked[0]
                                                       : QStringLiteral("nand and sdmc"));
        }
        if (requested) {
            text += QObject::tr("\n\nFritchii's previous data is kept next to it, in the folders "
                                "ending in \".backup\".");
        }
        text += QObject::tr("\n\nShaders get prepared again the first time you start a game.\n"
                            "Your old folder was not changed: %1")
                    .arg(old_text);
        QMessageBox box(QMessageBox::Information, QStringLiteral("Fritchii"), text,
                        QMessageBox::Ok, main_window);
        auto* open = box.addButton(QObject::tr("Open folder"), QMessageBox::ActionRole);
        box.exec();
        if (box.clickedButton() == open) {
            QDesktopServices::openUrl(QUrl::fromLocalFile(destination_text));
        }
        return;
    }

    if (Migration::GetState(state_dir) == Migration::State::Pending) {
        Migration::Recover(state_dir);
    }
    QMessageBox box(QMessageBox::Warning, QStringLiteral("Fritchii"),
                    QObject::tr("Your data in the %1 folder could not be copied to Fritchii's "
                                "folder. Nothing was changed.\n\n%2\n\nClose any emulator or "
                                "program using it, then start Fritchii again.")
                        .arg(folder, failure),
                    QMessageBox::NoButton, main_window);
    auto* later = box.addButton(QObject::tr("Close and try again"), QMessageBox::AcceptRole);
    box.addButton(QObject::tr("Start without it"), QMessageBox::RejectRole);
    box.exec();
    if (box.clickedButton() == later) {
        throw std::runtime_error("The import will be tried again on the next launch.");
    }
    Migration::MarkFresh(state_dir);
}
