// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2015 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <QLabel>
#include <QLineEdit>
#include <QHash>
#include <QList>
#include <QSet>
#include <QPushButton>
#include <QStandardItemModel>
#include <QString>
#include <QThreadPool>
#include <QVBoxLayout>
#include <QVector>
#include <atomic>
#include <memory>
#include <QWidget>

#include "common/common_types.h"
#include "core/core.h"
#include "frontend_common/play_time_manager.h"
#include "qt_common/config/uisettings.h"
#include "qt_common/game_list/model.h"
#include "qt_common/util/game.h"

class QVariantAnimation;

class GameCard;
class GameListModel;
class GameTree;
class GameGrid;
class GameListSearchField;
class ControllerNavigation;
class GameListWorker;
class GameListDir;
class MainWindow;
enum class AmLaunchType;
enum class StartGameType;

class GameCarousel;

class QAbstractItemView;
namespace Core {
class System;
}

namespace FileSys {
class ManualContentProvider;
class VfsFilesystem;
} // namespace FileSys

enum class GameListOpenTarget {
    SaveData,
    ModData,
};

enum class DumpRomFSTarget {
    Normal,
    SDMC,
};

class GameList : public QWidget {
    Q_OBJECT

public:
    explicit GameList(std::shared_ptr<FileSys::VfsFilesystem> vfs_,
                      FileSys::ManualContentProvider* provider_,
                      PlayTime::PlayTimeManager& play_time_manager_, Core::System& system_,
                      MainWindow* parent = nullptr);
    ~GameList() override;

    QString GetLastFilterResultItem() const;
    void ClearFilter();
    void SetFilterFocus();
    void SetFilterVisible(bool visibility);
    bool IsEmpty() const;

    void PopulateAsync(QVector<UISettings::GameDir>& game_dirs);
    void SetScanEnabled(bool enabled);
    void SetReportableProgramId(u64 program_id);

    void SaveInterfaceLayout();
    void LoadInterfaceLayout();

    QStandardItemModel* GetModel() const;

    /// Disables events from the emulated controller
    void UnloadController();

    void ResetViewMode();

public slots:
    void SetNupConversionRunning(bool running);
    void SetNxToolRunning(bool running);
    void RefreshGameDirectory();
    /// Shows a new name for every library entry of the program, without a rescan.
    void SetGameTitle(u64 program_id, const QString& title);
    /// The user saved or removed the program's custom artwork: a lookup still running for it
    /// does not replace that choice.
    void OnCustomArtworkChanged(u64 program_id);
    void RefreshExternalContent();
    void UpdateIconSizes();
    void OnPopulate();

signals:
    void ReportCompatibilityRequested(u64 program_id);
    void BootGame(const QString& game_path, StartGameType type);
    void GameChosen(const QString& game_path, const u64 title_id = 0);
    void OpenFolderRequested(u64 program_id, GameListOpenTarget target,
                             const std::string& game_path);
    void OpenTransferableShaderCacheRequested(u64 program_id);
    void PrepareShaderWarmRequested(u64 program_id, const std::string& game_path);
    void DeleteShaderWarmRequested(u64 program_id);
    void RemoveInstalledEntryRequested(u64 program_id, QtCommon::Game::InstalledEntryType type);
    void RemoveFileRequested(u64 program_id, QtCommon::Game::GameListRemoveTarget target,
                             const std::string& game_path);
    void RemovePlayTimeRequested(u64 program_id);
    void SetPlayTimeRequested(u64 program_id);
    void DumpRomFSRequested(u64 program_id, const std::string& game_path, DumpRomFSTarget target);
    void VerifyIntegrityRequested(const std::string& game_path);
    void ConvertToNupRequested(const std::string& game_path);
    void ExtractNupRequested(const std::string& game_path);
    void EditNupMetadataRequested(const std::string& game_path);
    void ConvertToNxRequested(const std::string& game_path);
    void AddToNxRequested(const std::string& game_path);
    void RemoveFromNxRequested(const std::string& game_path);
    void EditNxMetadataRequested(const std::string& game_path);
    void CopyTIDRequested(u64 program_id);
    void CreateShortcut(u64 program_id, const std::string& game_path,
                        const QtCommon::Game::ShortcutTarget target);
    void OpenPerGameGeneralRequested(const std::string& file);
    void LinkToRyujinxRequested(const u64& program_id);
    void OpenDirectory(const QString& directory);
    void AddDirectory();
    void ShowList(bool show);
    void PopulatingCompleted();
    void SaveConfig();

private slots:
    void OnTextChanged(const QString& new_text);
    void OnFilterCloseClicked();
    void OnPopulatingCompleted(const QStringList& watch_list);

private:
    void SetupViews();
    void SetupScrollAnimation();
    bool eventFilter(QObject* obj, QEvent* event) override;
    void changeEvent(QEvent*) override;

    void ValidateEntry(const QModelIndex& item);

    void ToggleFavorite(u64 program_id);

    void PopupContextMenu(const QPoint& menu_location);
    void AddGamePopup(QMenu& context_menu, u64 program_id, const std::string& path);
    void AddCustomDirPopup(QMenu& context_menu, QModelIndex selected);
    void AddPermDirPopup(QMenu& context_menu, QModelIndex selected);
    void AddFavoritesPopup(QMenu& context_menu);

    void RetranslateUI();

    /// Queues a SteamGridDB lookup for every game without artwork, once per game per session.
    void QueueArtworkLookups();
    /// Queues one game; forced lookups run again and replace existing custom artwork.
    void QueueArtworkLookup(u64 program_id, const QString& title, bool forced);
    void StartArtworkLookups();
    /// Shows the program's custom artwork in every library entry of it, without a rescan.
    void ShowCustomArtwork(u64 program_id);
    QString GameTitle(u64 program_id) const;

    friend class GameListSearchField;

    std::shared_ptr<FileSys::VfsFilesystem> vfs;
    FileSys::ManualContentProvider* provider;
    GameListSearchField* search_field = nullptr;
    MainWindow* main_window = nullptr;
    QVBoxLayout* layout = nullptr;

    GameTree* tree_view = nullptr;
    GameGrid* grid_view = nullptr;
    GameCarousel* carousel_view = nullptr;
    GameListModel* item_model = nullptr;
    Settings::GameListMode game_list_mode;

    ControllerNavigation* controller_navigation = nullptr;

    QVariantAnimation* vertical_scroll = nullptr;
    QVariantAnimation* horizontal_scroll = nullptr;
    int vertical_scroll_target = 0;
    int horizontal_scroll_target = 0;

    const PlayTime::PlayTimeManager& play_time_manager;
    Core::System& system;

    bool m_isTreeMode = true;
    bool m_populating = false;
    bool nup_conversion_running{};
    bool nx_tool_running{};
    u64 reportable_program_id{};
    QAbstractItemView* m_currentView = nullptr;

    struct ArtworkLookup {
        u64 program_id;
        QString title;
        bool forced{};
        /// Only the newest lookup of a program may save its result.
        u64 serial{};
        /// The API key the lookup ran with.
        std::string key;
    };
    QList<ArtworkLookup> artwork_queue;
    QSet<u64> artwork_attempted;
    QHash<u64, u64> artwork_newest;
    /// Programs whose running newest lookup was asked for by the user.
    QSet<u64> artwork_forced;
    u64 artwork_serial = 0;
    /// The API key the attempted games were looked up with.
    std::string artwork_key;
    int artwork_running = 0;
    bool artwork_failure_logged = false;
    /// Programs whose looked up artwork was saved since the last scan finished. A scan can add
    /// their rows with the default picture after the save, so the artwork is shown again.
    QSet<u64> artwork_saved;
    std::shared_ptr<std::atomic_bool> artwork_cancel = std::make_shared<std::atomic_bool>(false);
    /// Lookups wait on the network: their own threads keep them from holding up a library scan
    /// on the global pool.
    QThreadPool artwork_pool;
};

class GameListPlaceholder : public QWidget {
    Q_OBJECT
public:
    explicit GameListPlaceholder(MainWindow* parent = nullptr);
    ~GameListPlaceholder();

signals:
    void AddDirectory();

protected:
    void mouseDoubleClickEvent(QMouseEvent* event) override;

private:
    void changeEvent(QEvent* event) override;
    void RetranslateUI();

    QVBoxLayout* layout = nullptr;
    QLabel* image = nullptr;
    QLabel* text = nullptr;
};
