// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2015 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <filesystem>
#include <functional>
#include <utility>

#include <QAbstractItemView>
#include <QApplication>
#include <QDir>
#include <QFileInfo>
#include <QHeaderView>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QList>
#include <QListView>
#include <QMenu>
#include <QScrollBar>
#include <QScroller>
#include <QScrollerProperties>
#include <QToolButton>
#include <QFutureWatcher>
#include <QVariantAnimation>
#include <QtConcurrent/QtConcurrentRun>
#include <qlayoutitem.h>

#include "common/common_types.h"
#include "common/fs/fs_util.h"
#include "common/logging.h"
#include "core/core.h"
#include "core/file_sys/nup_writer.h"
#ifdef LICHII_NX_TOOLS
#include "lichii/nx_tools.h"
#endif
#include "core/file_sys/patch_manager.h"
#include "core/file_sys/registered_cache.h"
#include "qt_common/config/uisettings.h"
#include "qt_common/game_list/game_list_p.h"
#include "qt_common/game_list/model.h"
#include "qt_common/qt_common.h"
#include "qt_common/util/game.h"
#include "lichii/game/carousel.h"
#include "lichii/game/game_grid.h"
#include "lichii/game/game_list.h"
#include "lichii/game/game_tree.h"
#include "lichii/game/search_field.h"
#include "lichii/main_window.h"
#include "lichii/shader_warm_dialog.h"
#include "lichii/util/controller_navigation.h"
#include "lichii/util/icon_search.h"
#include "lichii/util/util.h"

namespace {
// A few at a time keeps SteamGridDB happy and the scan of a large library short.
constexpr int MaxParallelLookups = 2;
} // namespace

GameList::GameList(FileSys::VirtualFilesystem vfs_, FileSys::ManualContentProvider* provider_,
                   PlayTime::PlayTimeManager& play_time_manager_, Core::System& system_,
                   MainWindow* parent)
    : QWidget{parent}, vfs{std::move(vfs_)}, provider{provider_},
      play_time_manager{play_time_manager_}, system{system_} {

    this->main_window = parent;
    artwork_pool.setMaxThreadCount(MaxParallelLookups);
    layout = new QVBoxLayout;
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    setLayout(layout);

    item_model = new GameListModel(vfs, provider, play_time_manager, system, this);

    SetupViews();

    search_field = new GameListSearchField(this);
    layout->addWidget(search_field);

    controller_navigation = new ControllerNavigation(system.HIDCore(), this);

    SetupScrollAnimation();

    connect(tree_view, &QTreeView::activated, this, &GameList::ValidateEntry);
    connect(tree_view, &QTreeView::customContextMenuRequested, this, &GameList::PopupContextMenu);

    connect(grid_view, &QListView::activated, this, &GameList::ValidateEntry);
    connect(grid_view, &QListView::customContextMenuRequested, this, &GameList::PopupContextMenu);

    connect(carousel_view, &QListView::activated, this, &GameList::ValidateEntry);
    connect(carousel_view, &QListView::customContextMenuRequested, this, &GameList::PopupContextMenu);

    connect(controller_navigation, &ControllerNavigation::TriggerKeyboardEvent, this,
            [this](Qt::Key key) {
                if (system.IsPoweredOn()) {
                    return;
                }
                if (!this->isActiveWindow()) {
                    return;
                }
                QKeyEvent* event = new QKeyEvent(QEvent::KeyPress, key, Qt::NoModifier);
                QCoreApplication::postEvent(m_currentView, event);
            });

    connect(item_model, &GameListModel::PopulatingCompleted, this,
            &GameList::OnPopulatingCompleted);

    connect(item_model, &GameListModel::ShowList, this, &GameList::ShowList);
    connect(item_model, &GameListModel::SaveConfig, this, &GameList::SaveConfig);
    connect(item_model, &GameListModel::PopulatingStarted, this, &GameList::OnPopulate);

    // TODO: impl on grid/carousel
    connect(tree_view, &GameTree::FilterResultReady, search_field,
            [this](int visible, int total) { search_field->setFilterResult(visible, total); });

    qRegisterMetaType<QList<QStandardItem*>>("QList<QStandardItem*>");

    ResetViewMode();
}

void GameList::SetNupConversionRunning(bool running) {
    nup_conversion_running = running;
}

void GameList::SetNxToolRunning(bool running) {
    nx_tool_running = running;
}

GameList::~GameList() {
    // Running lookups see the flag and stop; their results are dropped with this widget. Lookups
    // that have not started are dropped; the pool then waits for the running ones.
    artwork_cancel->store(true);
    artwork_pool.clear();
    UnloadController();
}

void GameList::SetupViews() {
    tree_view = new GameTree(this);
    grid_view = new GameGrid(this);
    carousel_view = new GameCarousel(this);

    tree_view->SetModel(item_model);
    grid_view->SetModel(item_model);
    carousel_view->SetModel(item_model);
}

QString GameList::GetLastFilterResultItem() const {
    return tree_view->GetLastFilterResultItem();
}

void GameList::ClearFilter() {
    search_field->clear();
}

void GameList::SetFilterFocus() {
    if (item_model->rowCount() > 0) {
        search_field->setFocus();
    }
}

void GameList::SetFilterVisible(bool visibility) {
    search_field->setVisible(visibility);
}

bool GameList::IsEmpty() const {
    return item_model->IsEmpty();
}

void GameList::OnPopulate() {
    m_populating = true;
    m_currentView->setEnabled(false);

    switch (game_list_mode) {
    case Settings::GameListMode::TreeView:
        tree_view->UpdateColumnVisibility(item_model);
        break;
    case Settings::GameListMode::GridView:
        grid_view->UpdateIconSize();
        break;
    case Settings::GameListMode::CarouselView:
        carousel_view->UpdateIconSize();
        break;
    }
}

void GameList::PopulateAsync(QVector<UISettings::GameDir>& game_dirs) {
    item_model->PopulateAsync(game_dirs);
}

void GameList::SetScanEnabled(bool enabled) {
    item_model->SetScanEnabled(enabled);
}

void GameList::SaveInterfaceLayout() {
    tree_view->SaveInterfaceLayout();
}

void GameList::LoadInterfaceLayout() {
    tree_view->LoadInterfaceLayout();
}

QStandardItemModel* GameList::GetModel() const {
    return item_model;
}

void GameList::UnloadController() {
    controller_navigation->UnloadController();
}

void GameList::ResetViewMode() {
    const auto mode = UISettings::values.game_list_mode.GetValue();
    game_list_mode = mode;

    if (m_currentView)
        layout->removeWidget(m_currentView);

    bool newTreeMode = false;

    switch (mode) {
    case Settings::GameListMode::TreeView:
        m_currentView = tree_view;
        newTreeMode = true;

        break;
    case Settings::GameListMode::GridView:
        m_currentView = grid_view;
        newTreeMode = false;

        break;
    case Settings::GameListMode::CarouselView:
        m_currentView = carousel_view;
        newTreeMode = false;

        break;
    default:
        UNREACHABLE();
    }

    tree_view->setVisible(false);
    grid_view->setVisible(false);
    carousel_view->setVisible(false);

    tree_view->setEnabled(false);
    grid_view->setEnabled(false);
    carousel_view->setEnabled(false);

    m_currentView->setVisible(true);
    // MainWindow switches the view right after starting the first populate; keep it disabled
    // until that finishes, as OnPopulate does for every other populate.
    m_currentView->setEnabled(!m_populating);
    layout->insertWidget(0, m_currentView);

    auto view = m_currentView->viewport();
    view->installEventFilter(this);

    view->grabGesture(Qt::SwipeGesture);
    view->grabGesture(Qt::PanGesture);

    QScroller::grabGesture(view, QScroller::LeftMouseButtonGesture);

    auto scroller = QScroller::scroller(view);
    QScrollerProperties props;
    props.setScrollMetric(QScrollerProperties::HorizontalOvershootPolicy,
                          QScrollerProperties::OvershootAlwaysOff);
    props.setScrollMetric(QScrollerProperties::VerticalOvershootPolicy,
                          QScrollerProperties::OvershootAlwaysOff);
    scroller->setScrollerProperties(props);

    if (m_isTreeMode != newTreeMode) {
        m_isTreeMode = newTreeMode;
        item_model->SetFlat(!m_isTreeMode);
        if (!UISettings::values.game_dirs.empty()) {
            item_model->PopulateAsync(UISettings::values.game_dirs);
        }
    }
}

void GameList::OnTextChanged(const QString& new_text) {
    const QString edit_filter_text = new_text.toLower();

    if (m_isTreeMode) {
        tree_view->ApplyFilter(edit_filter_text, item_model);
    } else {
        grid_view->ApplyFilter(edit_filter_text, item_model);
    }
}

void GameList::OnFilterCloseClicked() {
    main_window->filterBarSetChecked(false);
}

void GameList::OnPopulatingCompleted(const QStringList& watch_list) {
    emit ShowList(!item_model->IsEmpty());

    // favorites row
    if (m_isTreeMode) {
        tree_view->setRowHidden(0, item_model->invisibleRootItem()->index(),
                                UISettings::values.favorited_ids.size() == 0);
        tree_view->setExpanded(item_model->invisibleRootItem()->child(0)->index(),
                               UISettings::values.favorites_expanded.GetValue());
    }

    // restore collapsed/expanded flags
    if (m_isTreeMode) {
        const auto* root = item_model->invisibleRootItem();
        for (int i = 1; i < root->rowCount() - 1; ++i) {
            const auto* dir_item = root->child(i);
            const auto type = dir_item->data(GameListItem::TypeRole).value<GameListItemType>();
            if (type == GameListItemType::CustomDir || type == GameListItemType::SdmcDir ||
                type == GameListItemType::UserNandDir || type == GameListItemType::SysNandDir) {
                const int dir_index = dir_item->data(GameListDir::GameDirRole).toInt();
                if (dir_index >= 0 && dir_index < UISettings::values.game_dirs.size()) {
                    tree_view->setExpanded(dir_item->index(),
                                           UISettings::values.game_dirs[dir_index].expanded);
                }
            }
        }
    }

    // Watcher updates
    auto* watcher = item_model->GetWatcher();
    auto current_watch_list = watcher->directories();

    constexpr qsizetype LIMIT_WATCH_DIRECTORIES = 5000;
    constexpr int SLICE_SIZE = 25;

    QStringList to_remove, to_add;

    const auto slice = [&](const QStringList& list, std::function<void(const QStringList&)> callback) {
        const int len = (std::min)(list.size(), LIMIT_WATCH_DIRECTORIES);
        for (int i = 0; i < len; i += SLICE_SIZE) {
            auto chunk = list.mid(i, SLICE_SIZE);
            if (!chunk.isEmpty()) {
                callback(chunk);
            }
        }
    };

    // remove any paths not in the new watch list
    for (const auto& path : std::as_const(current_watch_list)) {
        if (!watch_list.contains(path)) {
            to_remove.emplaceBack(path);
        }
    }

    slice(to_remove, [watcher](const QStringList& chunk) { watcher->removePaths(chunk); });

    // add any paths not in the old watch list
    for (const auto& path : std::as_const(watch_list)) {
        if (!current_watch_list.contains(path)) {
            to_add.emplaceBack(path);
        }
    }

    slice(to_add, [watcher](const QStringList& chunk) { watcher->addPaths(chunk); });

    m_populating = false;
    m_currentView->setEnabled(true);

    int children_total = 0;
    for (int i = 1; i < item_model->rowCount() - 1; ++i) {
        children_total += item_model->item(i, 0)->rowCount();
    }

    search_field->setFilterResult(children_total, children_total);
    if (children_total > 0) {
        search_field->setFocus();
    }

    // TODO: carousel/grid impl.
    item_model->sort(tree_view->header()->sortIndicatorSection(),
                     tree_view->header()->sortIndicatorOrder());

    // A repopulate (a directory removed, deep scan toggled) replaces the rows: apply the search
    // text the user still has typed to them.
    if (!search_field->filterText().isEmpty()) {
        OnTextChanged(search_field->filterText());
    }

    // Artwork a lookup saved while this scan ran can be missing from rows added after the save.
    for (const u64 program_id : std::as_const(artwork_saved)) {
        ShowCustomArtwork(program_id);
    }
    artwork_saved.clear();

    emit PopulatingCompleted();
    QueueArtworkLookups();
}

QString GameList::GameTitle(u64 program_id) const {
    QString title;
    const std::function<void(const QStandardItem*)> visit = [&](const QStandardItem* parent) {
        for (int row = 0; row < parent->rowCount() && title.isEmpty(); ++row) {
            const QStandardItem* item = parent->child(row);
            if (item->data(GameListItem::TypeRole).value<GameListItemType>() ==
                    GameListItemType::Game &&
                item->data(GameListItemPath::ProgramIdRole).toULongLong() == program_id) {
                title = item->data(GameListItemPath::TitleRole).toString();
            }
            visit(item);
        }
    };
    visit(item_model->invisibleRootItem());
    return title;
}

void GameList::QueueArtworkLookups() {
    if (UISettings::values.steamgriddb_api_key.GetValue().empty()) {
        return;
    }
    const std::function<void(const QStandardItem*)> visit = [&](const QStandardItem* parent) {
        for (int row = 0; row < parent->rowCount(); ++row) {
            const QStandardItem* item = parent->child(row);
            if (item->data(GameListItem::TypeRole).value<GameListItemType>() ==
                GameListItemType::Game) {
                // Games with an icon in the file or custom artwork already have a picture.
                if (!item->data(GameListItemPath::HasIconRole).toBool()) {
                    QueueArtworkLookup(item->data(GameListItemPath::ProgramIdRole).toULongLong(),
                                       item->data(GameListItemPath::TitleRole).toString(), false);
                }
            } else {
                visit(item);
            }
        }
    };
    visit(item_model->invisibleRootItem());
}

void GameList::QueueArtworkLookup(u64 program_id, const QString& title, bool forced) {
    const std::string& key = UISettings::values.steamgriddb_api_key.GetValue();
    if (program_id == 0 || title.trimmed().isEmpty() || key.empty()) {
        return;
    }
    // A new key gives the games looked up with the old one another try.
    if (key != artwork_key) {
        artwork_key = key;
        artwork_attempted.clear();
    }
    if (!forced && artwork_attempted.contains(program_id)) {
        return;
    }
    artwork_attempted.insert(program_id);
    for (auto& queued : artwork_queue) {
        if (queued.program_id == program_id) {
            // A forced lookup takes over the waiting one, with the name the game has now.
            if (forced) {
                queued.title = title.trimmed();
                queued.forced = true;
            }
            return;
        }
    }
    const u64 serial = ++artwork_serial;
    artwork_newest.insert(program_id, serial);
    artwork_queue.append({program_id, title.trimmed(), forced, serial});
    StartArtworkLookups();
}

void GameList::StartArtworkLookups() {
    while (artwork_running < MaxParallelLookups && !artwork_queue.isEmpty()) {
        ArtworkLookup lookup = artwork_queue.takeFirst();
        lookup.key = UISettings::values.steamgriddb_api_key.GetValue();
        // A waiting lookup is always the newest one of its game.
        if (lookup.forced) {
            artwork_forced.insert(lookup.program_id);
        } else {
            artwork_forced.remove(lookup.program_id);
        }
        ++artwork_running;
        auto* const watcher = new QFutureWatcher<ArtworkLookupResult>(this);
        connect(watcher, &QFutureWatcher<ArtworkLookupResult>::finished, this, [this, watcher,
                                                                                 lookup] {
            watcher->deleteLater();
            --artwork_running;
            const ArtworkLookupResult result = watcher->result();
            // A newer lookup of the same game wins, whichever finishes first.
            const bool newest = artwork_newest.value(lookup.program_id) == lookup.serial;
            if (newest) {
                artwork_newest.remove(lookup.program_id);
                artwork_forced.remove(lookup.program_id);
            }
            // An automatic lookup only fills a game without artwork: artwork the user set while
            // it ran stays.
            std::error_code ec;
            const bool artwork_added =
                !lookup.forced &&
                std::filesystem::exists(QtCommon::Game::CustomArtworkPath(lookup.program_id), ec);
            // A lookup whose key was cleared or replaced while it ran saves nothing.
            const bool key_current =
                lookup.key == UISettings::values.steamgriddb_api_key.GetValue();
            if (result.status == ArtworkLookupStatus::Found && newest && !artwork_added &&
                key_current) {
                if (QtCommon::Game::SaveCustomArtwork(lookup.program_id, result.image)) {
                    LOG_INFO(Frontend, "SteamGridDB artwork saved for {:016X}", lookup.program_id);
                    artwork_saved.insert(lookup.program_id);
                    ShowCustomArtwork(lookup.program_id);
                }
            } else if (newest && result.status != ArtworkLookupStatus::Found &&
                       result.status != ArtworkLookupStatus::Cancelled &&
                       !artwork_failure_logged) {
                // A found picture that was dropped is no failure, and neither is the result of a
                // lookup replaced by a newer one, so the one log stays free.
                // Logged once per session. The key itself is never logged.
                artwork_failure_logged = true;
                const char* reason = "it could not be reached";
                if (result.status == ArtworkLookupStatus::BadApiKey) {
                    reason = "the API key was rejected";
                } else if (result.status == ArtworkLookupStatus::NotFound) {
                    reason = "no match was found";
                }
                LOG_INFO(Frontend, "SteamGridDB artwork lookup for {:016X} failed: {} {}",
                         lookup.program_id, reason, result.detail);
            }
            if (result.status == ArtworkLookupStatus::BadApiKey &&
                lookup.key == UISettings::values.steamgriddb_api_key.GetValue()) {
                // Every other lookup would fail the same way. A late failure with a key that was
                // replaced since leaves the lookups of the new key alone.
                artwork_queue.clear();
            }
            StartArtworkLookups();
        });
        // Switch 2 program IDs start with 04. Their titles are often remakes that share the name
        // of an older game, so the newest release of that name is the right one.
        const bool prefer_newest = (lookup.program_id >> 56) == 0x04;
        watcher->setFuture(QtConcurrent::run(
            &artwork_pool, [title = lookup.title.toStdString(), key = lookup.key,
                            cancel = artwork_cancel, prefer_newest] {
                return LookUpSteamGridDbArtwork(title, key, *cancel, prefer_newest);
            }));
    }
}

void GameList::ShowCustomArtwork(u64 program_id) {
    // Updates the shown pictures in place. A rescan here would stop and wait for a scan that is
    // still running on the UI thread, and rebuild the whole library for one picture.
    const std::vector<u8> artwork = QtCommon::Game::ReadCustomArtwork(program_id);
    if (artwork.empty()) {
        return;
    }
    const std::function<void(QStandardItem*)> visit = [&](QStandardItem* parent) {
        for (int row = 0; row < parent->rowCount(); ++row) {
            QStandardItem* item = parent->child(row);
            if (item->data(GameListItem::TypeRole).value<GameListItemType>() ==
                GameListItemType::Game) {
                if (item->data(GameListItemPath::ProgramIdRole).toULongLong() == program_id) {
                    GameListItemPath::SetPicture(item, artwork);
                }
            } else {
                visit(item);
            }
        }
    };
    visit(item_model->invisibleRootItem());
}

void GameList::SetGameTitle(u64 program_id, const QString& title) {
    if (program_id == 0 || title.isEmpty()) {
        return;
    }
    // Favorites are plain copies of a library entry that store their shown text, so they take it
    // from an updated library entry afterwards.
    QList<QStandardItem*> copies;
    QVariant display;
    QVariant sort;
    const std::function<void(QStandardItem*)> visit = [&](QStandardItem* parent) {
        for (int row = 0; row < parent->rowCount(); ++row) {
            QStandardItem* item = parent->child(row);
            if (item->data(GameListItem::TypeRole).value<GameListItemType>() ==
                GameListItemType::Game) {
                if (item->data(GameListItemPath::ProgramIdRole).toULongLong() == program_id) {
                    item->setData(title, GameListItemPath::TitleRole);
                    // A copy is a plain item: only library entries report the game type.
                    if (item->type() != static_cast<int>(GameListItemType::Game)) {
                        copies.append(item);
                    } else if (!display.isValid()) {
                        display = item->data(Qt::DisplayRole);
                        sort = item->data(GameListItem::SortRole);
                    }
                }
            } else {
                visit(item);
            }
        }
    };
    visit(item_model->invisibleRootItem());
    if (display.isValid()) {
        for (QStandardItem* copy : copies) {
            copy->setData(display, Qt::DisplayRole);
            copy->setData(sort, GameListItem::SortRole);
        }
    }
    // A list sorted by name puts the renamed game in its new place. This only reorders the rows
    // already shown: no rescan.
    if (display.isValid() || !copies.isEmpty()) {
        item_model->sort(tree_view->header()->sortIndicatorSection(),
                         tree_view->header()->sortIndicatorOrder());
    }
    // A new name can match the search text, or stop matching it.
    if (!search_field->filterText().isEmpty()) {
        OnTextChanged(search_field->filterText());
    }
    // A waiting artwork lookup searches the new name. A running one searched the old name, so
    // its result is dropped and the new name is looked up instead.
    const QString search_title = title.trimmed();
    for (auto& queued : artwork_queue) {
        if (queued.program_id == program_id) {
            if (!search_title.isEmpty()) {
                queued.title = search_title;
            }
            return;
        }
    }
    if (!artwork_newest.contains(program_id)) {
        return;
    }
    const bool forced = artwork_forced.contains(program_id);
    artwork_newest.remove(program_id);
    artwork_forced.remove(program_id);
    // An automatic lookup only fills a game that still has no artwork.
    std::error_code ec;
    if (search_title.isEmpty() || UISettings::values.steamgriddb_api_key.GetValue().empty() ||
        (!forced && std::filesystem::exists(QtCommon::Game::CustomArtworkPath(program_id), ec))) {
        return;
    }
    const u64 serial = ++artwork_serial;
    artwork_newest.insert(program_id, serial);
    artwork_queue.append({program_id, search_title, forced, serial});
    StartArtworkLookups();
}

void GameList::OnCustomArtworkChanged(u64 program_id) {
    // A running lookup's result is dropped and a waiting one is not started.
    artwork_newest.remove(program_id);
    artwork_forced.remove(program_id);
    artwork_queue.removeIf(
        [program_id](const ArtworkLookup& queued) { return queued.program_id == program_id; });
}

void GameList::RefreshGameDirectory() {
    item_model->RefreshGameDirectory();
}

void GameList::RefreshExternalContent() {
    item_model->RefreshExternalContent();
}

void GameList::UpdateIconSizes() {
    switch (game_list_mode) {
    case Settings::GameListMode::GridView:
        grid_view->UpdateIconSize();
        break;
    case Settings::GameListMode::CarouselView:
        carousel_view->UpdateIconSize();
        break;
    case Settings::GameListMode::TreeView:
        break;
    }
}

void GameList::ValidateEntry(const QModelIndex& item) {
    const auto selected = item.sibling(item.row(), 0);

    switch (selected.data(GameListItem::TypeRole).value<GameListItemType>()) {
    case GameListItemType::Game: {
        const QString file_path = selected.data(GameListItemPath::FullPathRole).toString();
        if (file_path.isEmpty())
            return;
        const QFileInfo file_info(file_path);
        if (!file_info.exists())
            return;

        if (file_info.isDir()) {
            const QDir dir{file_path};
            const QStringList matching_main = dir.entryList({QStringLiteral("main")}, QDir::Files);
            if (matching_main.size() == 1) {
                emit GameChosen(dir.path() + QDir::separator() + matching_main[0]);
            }
            return;
        }

        const auto title_id = selected.data(GameListItemPath::ProgramIdRole).toULongLong();

        search_field->clear();
        emit GameChosen(file_path, title_id);
        break;
    }
    case GameListItemType::AddDir:
        emit AddDirectory();
        break;
    default:
        break;
    }
}

void GameList::ToggleFavorite(u64 program_id) {
    item_model->ToggleFavorite(program_id);

    if (UISettings::values.favorited_ids.contains(program_id)) {
        tree_view->setRowHidden(0, item_model->invisibleRootItem()->index(),
                                !search_field->filterText().isEmpty());
        item_model->sort(tree_view->header()->sortIndicatorSection(),
                         tree_view->header()->sortIndicatorOrder());
    } else {
        if (UISettings::values.favorited_ids.size() == 0) {
            tree_view->setRowHidden(0, item_model->invisibleRootItem()->index(), true);
        }
    }
}

void GameList::PopupContextMenu(const QPoint& menu_location) {
    QModelIndex item = m_currentView->indexAt(menu_location);
    if (!item.isValid()) {
        if (m_isTreeMode)
            return;

        QMenu blank_menu;
        QAction* addGameDirAction = blank_menu.addAction(tr("&Add New Game Directory"));
        addGameDirAction->setIcon(MenuIcon("plus"));

        connect(addGameDirAction, &QAction::triggered, this, &GameList::AddDirectory);
        blank_menu.exec(m_currentView->viewport()->mapToGlobal(menu_location));
        return;
    }

    const auto selected = item.sibling(item.row(), 0);
    QMenu context_menu;
    switch (selected.data(GameListItem::TypeRole).value<GameListItemType>()) {
    case GameListItemType::Game:
        AddGamePopup(context_menu, selected.data(GameListItemPath::ProgramIdRole).toULongLong(),
                     selected.data(GameListItemPath::FullPathRole).toString().toStdString());
        break;
    case GameListItemType::CustomDir:
        AddPermDirPopup(context_menu, selected);
        AddCustomDirPopup(context_menu, selected);
        break;
    case GameListItemType::SdmcDir:
    case GameListItemType::UserNandDir:
    case GameListItemType::SysNandDir:
        AddPermDirPopup(context_menu, selected);
        break;
    case GameListItemType::Favorites:
        AddFavoritesPopup(context_menu);
        break;
    default:
        break;
    }
    context_menu.exec(m_currentView->viewport()->mapToGlobal(menu_location));
}

void GameList::SetReportableProgramId(u64 program_id) {
    reportable_program_id = program_id;
}

void GameList::AddGamePopup(QMenu& context_menu, u64 program_id, const std::string& path) {
    // TODO(crueter): Refactor this and make it less bad
    QAction* favorite = context_menu.addAction(tr("Favorite"));
    context_menu.addSeparator();
    QAction* start_game = context_menu.addAction(tr("Start Game"));
    QAction* start_game_global =
        context_menu.addAction(tr("Start Game without Custom Configuration"));
    context_menu.addSeparator();
    QAction* open_save_location = context_menu.addAction(tr("Open Save Data Location"));
    QAction* open_mod_location = context_menu.addAction(tr("Open Mod Data Location"));
    QAction* open_transferable_shader_cache =
        context_menu.addAction(tr("Open Transferable Pipeline Cache"));
    const bool has_shader_warm = ShaderWarmFilesExist(program_id);
    QAction* prepare_shaders = context_menu.addAction(
        has_shader_warm ? tr("Rebuild Shader Precache") : tr("Build Shader Precache"));
    QAction* delete_shader_warm = nullptr;
    if (has_shader_warm) {
        delete_shader_warm = context_menu.addAction(tr("Delete Shader Precache"));
    }
    QAction* ryujinx = context_menu.addAction(tr("Link to Ryujinx"));
    context_menu.addSeparator();
    QMenu* remove_menu = context_menu.addMenu(tr("Remove"));
    QAction* remove_update = remove_menu->addAction(tr("Remove Installed Update"));
    QAction* remove_dlc = remove_menu->addAction(tr("Remove All Installed DLC"));
    QAction* remove_custom_config = remove_menu->addAction(tr("Remove Custom Configuration"));
    QAction* remove_cache_storage = remove_menu->addAction(tr("Remove Cache Storage"));
    QAction* remove_gl_shader_cache = remove_menu->addAction(tr("Remove OpenGL Pipeline Cache"));
    QAction* remove_vk_shader_cache = remove_menu->addAction(tr("Remove Vulkan Pipeline Cache"));
    remove_menu->addSeparator();
    QAction* remove_shader_cache = remove_menu->addAction(tr("Remove All Pipeline Caches"));
    QAction* remove_all_content = remove_menu->addAction(tr("Remove All Installed Contents"));
    QMenu* play_time_menu = context_menu.addMenu(tr("Manage Play Time"));
    QAction* set_play_time = play_time_menu->addAction(tr("Edit Play Time Data"));
    QAction* remove_play_time_data = play_time_menu->addAction(tr("Remove Play Time Data"));
    const bool is_loose_dump =
        FileSys::Nup::DumpRootFromMain(std::filesystem::path{Common::FS::ToU8String(path)})
            .has_value();
    const bool is_nup_archive =
        !is_loose_dump &&
        QString::fromStdString(path).endsWith(QStringLiteral(".nup"), Qt::CaseInsensitive);
    QAction* extract_nup = nullptr;
    if (is_nup_archive) {
        extract_nup = context_menu.addAction(tr("Dump NUP..."));
        extract_nup->setEnabled(!nup_conversion_running);
    }
    QMenu* dump_romfs_menu = context_menu.addMenu(tr("Dump RomFS"));
    QAction* dump_romfs = dump_romfs_menu->addAction(tr("Dump RomFS"));
    QAction* dump_romfs_sdmc = dump_romfs_menu->addAction(tr("Dump RomFS to SDMC"));
    QAction* verify_integrity = context_menu.addAction(tr("Verify Integrity"));
    QAction* copy_tid = context_menu.addAction(tr("Copy Title ID to Clipboard"));
    QAction* convert_nup = nullptr;
    if (is_loose_dump) {
        convert_nup = context_menu.addAction(tr("Convert to NUP..."));
        convert_nup->setEnabled(!nup_conversion_running);
    }
    QAction* convert_nx = nullptr;
    QAction* add_to_nx = nullptr;
    QAction* remove_from_nx = nullptr;
    QAction* edit_nx_metadata = nullptr;
#ifdef LICHII_NX_TOOLS
    if (NxTools::CanConvert(path)) {
        convert_nx = context_menu.addAction(tr("Convert to NX..."));
        convert_nx->setEnabled(!nx_tool_running);
    }
    if (NxTools::IsNxArchive(path)) {
        QMenu* nx_menu = context_menu.addMenu(tr("Manage NX Archive"));
        add_to_nx = nx_menu->addAction(tr("Add Update or DLC..."));
        remove_from_nx = nx_menu->addAction(tr("Remove Update or DLC..."));
        nx_menu->addSeparator();
        edit_nx_metadata = nx_menu->addAction(tr("Edit Title and Icon..."));
        nx_menu->setEnabled(!nx_tool_running);
        nx_menu->menuAction()->setIcon(MenuIcon("archive"));
    }
#endif
#if !defined(__APPLE__)
    QMenu* shortcut_menu = context_menu.addMenu(tr("Create Shortcut"));
    shortcut_menu->menuAction()->setIcon(MenuIcon("share"));
    QAction* create_desktop_shortcut = shortcut_menu->addAction(tr("Add to Desktop"));
    QAction* create_applications_menu_shortcut =
        shortcut_menu->addAction(tr("Add to Applications Menu"));
#endif
    context_menu.addSeparator();
    QAction* edit_nup = nullptr;
    if (is_nup_archive) {
        edit_nup = context_menu.addAction(tr("Edit NUP Metadata..."));
        edit_nup->setEnabled(!nup_conversion_running);
    }
    QAction* lookup_artwork = context_menu.addAction(tr("Look up artwork on SteamGridDB"));
    lookup_artwork->setVisible(program_id != 0);
    lookup_artwork->setEnabled(!UISettings::values.steamgriddb_api_key.GetValue().empty());
    QAction* properties = context_menu.addAction(tr("Configure Game"));
    if (program_id != 0 && program_id == reportable_program_id) {
        auto* report = context_menu.addAction(tr("Submit compatibility report…"));
        connect(report, &QAction::triggered, this,
                [this, program_id] { emit ReportCompatibilityRequested(program_id); });
    }

    const std::array<std::pair<QAction*, const char*>, 20> icons{{
        {favorite, "star"},
        {start_game, "play"},
        {start_game_global, "play"},
        {open_save_location, "folder"},
        {open_mod_location, "puzzle"},
        {open_transferable_shader_cache, "layers"},
        {prepare_shaders, "cpu"},
        {delete_shader_warm, "trash"},
        {ryujinx, "link"},
        {remove_menu->menuAction(), "trash"},
        {play_time_menu->menuAction(), "clock"},
        {extract_nup, "archive"},
        {dump_romfs_menu->menuAction(), "hdd"},
        {verify_integrity, "check"},
        {copy_tid, "copy"},
        {convert_nup, "archive"},
        {convert_nx, "archive"},
        {edit_nx_metadata, "edit"},
        {edit_nup, "edit"},
        {properties, "sliders"},
    }};
    for (const auto& [action, name] : icons) {
        if (action != nullptr) {
            action->setIcon(MenuIcon(name));
        }
    }

    favorite->setVisible(program_id != 0);
    favorite->setCheckable(true);
    favorite->setChecked(UISettings::values.favorited_ids.contains(program_id));
    open_save_location->setVisible(program_id != 0);
    open_mod_location->setVisible(program_id != 0);
    open_transferable_shader_cache->setVisible(program_id != 0);
    prepare_shaders->setVisible(program_id != 0);
    remove_update->setVisible(program_id != 0);
    remove_dlc->setVisible(program_id != 0);
    remove_gl_shader_cache->setVisible(program_id != 0);
    remove_vk_shader_cache->setVisible(program_id != 0);
    remove_shader_cache->setVisible(program_id != 0);
    remove_all_content->setVisible(program_id != 0);

    connect(favorite, &QAction::triggered, this,
            [this, program_id]() { ToggleFavorite(program_id); });
    connect(open_save_location, &QAction::triggered, this, [this, program_id, path]() {
        emit OpenFolderRequested(program_id, GameListOpenTarget::SaveData, path);
    });
    connect(start_game, &QAction::triggered, this,
            [this, path]() { emit BootGame(QString::fromStdString(path), StartGameType::Normal); });
    connect(start_game_global, &QAction::triggered, this,
            [this, path]() { emit BootGame(QString::fromStdString(path), StartGameType::Global); });
    connect(open_mod_location, &QAction::triggered, this, [this, program_id, path]() {
        emit OpenFolderRequested(program_id, GameListOpenTarget::ModData, path);
    });
    connect(open_transferable_shader_cache, &QAction::triggered, this,
            [this, program_id]() { emit OpenTransferableShaderCacheRequested(program_id); });
    connect(prepare_shaders, &QAction::triggered, this,
            [this, program_id, path]() { emit PrepareShaderWarmRequested(program_id, path); });
    if (delete_shader_warm) {
        connect(delete_shader_warm, &QAction::triggered, this,
                [this, program_id]() { emit DeleteShaderWarmRequested(program_id); });
    }
    connect(remove_all_content, &QAction::triggered, this, [this, program_id]() {
        emit RemoveInstalledEntryRequested(program_id, QtCommon::Game::InstalledEntryType::Game);
    });
    connect(remove_update, &QAction::triggered, this, [this, program_id]() {
        emit RemoveInstalledEntryRequested(program_id, QtCommon::Game::InstalledEntryType::Update);
    });
    connect(remove_dlc, &QAction::triggered, this, [this, program_id]() {
        emit RemoveInstalledEntryRequested(program_id,
                                           QtCommon::Game::InstalledEntryType::AddOnContent);
    });
    connect(remove_gl_shader_cache, &QAction::triggered, this, [this, program_id, path]() {
        emit RemoveFileRequested(program_id, QtCommon::Game::GameListRemoveTarget::GlShaderCache,
                                 path);
    });
    connect(remove_vk_shader_cache, &QAction::triggered, this, [this, program_id, path]() {
        emit RemoveFileRequested(program_id, QtCommon::Game::GameListRemoveTarget::VkShaderCache,
                                 path);
    });
    connect(remove_shader_cache, &QAction::triggered, this, [this, program_id, path]() {
        emit RemoveFileRequested(program_id, QtCommon::Game::GameListRemoveTarget::AllShaderCache,
                                 path);
    });
    connect(remove_custom_config, &QAction::triggered, this, [this, program_id, path]() {
        emit RemoveFileRequested(program_id,
                                 QtCommon::Game::GameListRemoveTarget::CustomConfiguration, path);
    });
    connect(set_play_time, &QAction::triggered, this,
            [this, program_id]() { emit SetPlayTimeRequested(program_id); });
    connect(remove_play_time_data, &QAction::triggered, this,
            [this, program_id]() { emit RemovePlayTimeRequested(program_id); });
    connect(remove_cache_storage, &QAction::triggered, this, [this, program_id, path] {
        emit RemoveFileRequested(program_id, QtCommon::Game::GameListRemoveTarget::CacheStorage,
                                 path);
    });
    connect(dump_romfs, &QAction::triggered, this, [this, program_id, path]() {
        emit DumpRomFSRequested(program_id, path, DumpRomFSTarget::Normal);
    });
    connect(dump_romfs_sdmc, &QAction::triggered, this, [this, program_id, path]() {
        emit DumpRomFSRequested(program_id, path, DumpRomFSTarget::SDMC);
    });
    connect(verify_integrity, &QAction::triggered, this,
            [this, path]() { emit VerifyIntegrityRequested(path); });
    connect(copy_tid, &QAction::triggered, this,
            [this, program_id]() { emit CopyTIDRequested(program_id); });
    if (convert_nup) {
        connect(convert_nup, &QAction::triggered, this,
                [this, path]() { emit ConvertToNupRequested(path); });
    }
    if (convert_nx) {
        connect(convert_nx, &QAction::triggered, this,
                [this, path]() { emit ConvertToNxRequested(path); });
    }
    if (add_to_nx) {
        connect(add_to_nx, &QAction::triggered, this,
                [this, path]() { emit AddToNxRequested(path); });
        connect(remove_from_nx, &QAction::triggered, this,
                [this, path]() { emit RemoveFromNxRequested(path); });
        connect(edit_nx_metadata, &QAction::triggered, this,
                [this, path]() { emit EditNxMetadataRequested(path); });
    }
    if (extract_nup) {
        connect(extract_nup, &QAction::triggered, this,
                [this, path]() { emit ExtractNupRequested(path); });
    }
    if (edit_nup) {
        connect(edit_nup, &QAction::triggered, this,
                [this, path]() { emit EditNupMetadataRequested(path); });
    }
#if !defined(__APPLE__)
    connect(create_desktop_shortcut, &QAction::triggered, this, [this, program_id, path]() {
        emit CreateShortcut(program_id, path, QtCommon::Game::ShortcutTarget::Desktop);
    });
    connect(create_applications_menu_shortcut, &QAction::triggered, this,
            [this, program_id, path]() {
                emit CreateShortcut(program_id, path, QtCommon::Game::ShortcutTarget::Applications);
            });
#endif
    connect(properties, &QAction::triggered, this,
            [this, path]() { emit OpenPerGameGeneralRequested(path); });
    connect(lookup_artwork, &QAction::triggered, this, [this, program_id] {
        QueueArtworkLookup(program_id, GameTitle(program_id), true);
    });

    connect(ryujinx, &QAction::triggered, this,
            [this, program_id]() { emit LinkToRyujinxRequested(program_id); });
};

void GameList::AddCustomDirPopup(QMenu& context_menu, QModelIndex selected) {
    UISettings::GameDir& game_dir =
        UISettings::values.game_dirs[selected.data(GameListDir::GameDirRole).toInt()];

    QAction* deep_scan = context_menu.addAction(tr("Scan Subfolders"));
    QAction* delete_dir = context_menu.addAction(tr("Remove Game Directory"));
    delete_dir->setIcon(MenuIcon("x"));

    deep_scan->setCheckable(true);
    deep_scan->setChecked(game_dir.deep_scan);

    connect(deep_scan, &QAction::triggered, this, [this, &game_dir] {
        game_dir.deep_scan = !game_dir.deep_scan;
        PopulateAsync(UISettings::values.game_dirs);
    });
    connect(delete_dir, &QAction::triggered, this, [this, &game_dir] {
        UISettings::values.game_dirs.removeOne(game_dir);
        // Repopulate instead of removing the row: a running worker adds games under the
        // directory rows it created, and the other rows' GameDirRole indices would shift.
        PopulateAsync(UISettings::values.game_dirs);
    });
}

void GameList::AddPermDirPopup(QMenu& context_menu, QModelIndex selected) {
    const int game_dir_index = selected.data(GameListDir::GameDirRole).toInt();

    QAction* move_up = context_menu.addAction(tr("\u25B2 Move Up"));
    QAction* move_down = context_menu.addAction(tr("\u25bc Move Down"));
    QAction* open_directory_location = context_menu.addAction(tr("Open Directory Location"));
    open_directory_location->setIcon(MenuIcon("folder"));

    const int row = selected.row();

    move_up->setEnabled(row > 1);
    move_down->setEnabled(row < item_model->rowCount() - 2);

    connect(move_up, &QAction::triggered, this, [this, selected, row, game_dir_index] {
        const int other_index = selected.sibling(row - 1, 0).data(GameListDir::GameDirRole).toInt();
        std::swap(UISettings::values.game_dirs[game_dir_index],
                  UISettings::values.game_dirs[other_index]);
        item_model->setData(selected, QVariant(other_index), GameListDir::GameDirRole);
        item_model->setData(selected.sibling(row - 1, 0), QVariant(game_dir_index),
                            GameListDir::GameDirRole);
        QList<QStandardItem*> items = item_model->takeRow(row);
        item_model->invisibleRootItem()->insertRow(row - 1, items);
        tree_view->setExpanded(selected.sibling(row - 1, 0),
                               UISettings::values.game_dirs[other_index].expanded);
    });

    connect(move_down, &QAction::triggered, this, [this, selected, row, game_dir_index] {
        const int other_index = selected.sibling(row + 1, 0).data(GameListDir::GameDirRole).toInt();
        std::swap(UISettings::values.game_dirs[game_dir_index],
                  UISettings::values.game_dirs[other_index]);
        item_model->setData(selected, QVariant(other_index), GameListDir::GameDirRole);
        item_model->setData(selected.sibling(row + 1, 0), QVariant(game_dir_index),
                            GameListDir::GameDirRole);
        const QList<QStandardItem*> items = item_model->takeRow(row);
        item_model->invisibleRootItem()->insertRow(row + 1, items);
        tree_view->setExpanded(selected.sibling(row + 1, 0),
                               UISettings::values.game_dirs[other_index].expanded);
    });

    connect(open_directory_location, &QAction::triggered, this, [this, game_dir_index] {
        emit OpenDirectory(
            QString::fromStdString(UISettings::values.game_dirs[game_dir_index].path));
    });
}

void GameList::AddFavoritesPopup(QMenu& context_menu) {
    QAction* clear = context_menu.addAction(tr("Clear"));

    connect(clear, &QAction::triggered, this, [this] {
        UISettings::values.favorited_ids.clear();
        item_model->invisibleRootItem()->child(0)->removeRows(
            0, item_model->invisibleRootItem()->child(0)->rowCount());
        tree_view->setRowHidden(0, item_model->invisibleRootItem()->index(), true);
    });
}

void GameList::changeEvent(QEvent* event) {
    if (event->type() == QEvent::LanguageChange) {
        RetranslateUI();
    }

    QWidget::changeEvent(event);
}

void GameList::RetranslateUI() {
    item_model->RetranslateUI();
}

void GameList::SetupScrollAnimation() {
    auto setup = [this](QVariantAnimation* anim, QScrollBar* bar) {
        anim->setEasingCurve(QEasingCurve::OutCubic);
        anim->setDuration(200);
        connect(anim, &QVariantAnimation::valueChanged, this,
                [bar](const QVariant& value) { bar->setValue(value.toInt()); });
    };

    vertical_scroll = new QVariantAnimation(this);
    horizontal_scroll = new QVariantAnimation(this);

    setup(vertical_scroll, tree_view->verticalScrollBar());
    setup(horizontal_scroll, tree_view->horizontalScrollBar());

    setup(vertical_scroll, grid_view->verticalScrollBar());
    setup(horizontal_scroll, grid_view->horizontalScrollBar());
}

bool GameList::eventFilter(QObject* obj, QEvent* event) {
    if (obj == m_currentView->viewport() && event->type() == QEvent::Wheel) {
        QWheelEvent* wheelEvent = static_cast<QWheelEvent*>(event);

        bool horizontal = wheelEvent->modifiers() & Qt::ShiftModifier;

        int deltaX = wheelEvent->angleDelta().x();
        int deltaY = wheelEvent->angleDelta().y();

        if (horizontal && deltaY != 0 && deltaX == 0) {
            deltaX = deltaY;
            deltaY = 0;
        }

        if (deltaY != 0) {
            if (vertical_scroll->state() == QAbstractAnimation::Stopped)
                vertical_scroll_target = m_currentView->verticalScrollBar()->value();

            vertical_scroll_target -= deltaY;
            vertical_scroll_target =
                qBound(0, vertical_scroll_target, m_currentView->verticalScrollBar()->maximum());

            vertical_scroll->stop();
            vertical_scroll->setStartValue(m_currentView->verticalScrollBar()->value());
            vertical_scroll->setEndValue(vertical_scroll_target);
            vertical_scroll->start();
        }

        if (deltaX != 0) {
            if (horizontal_scroll->state() == QAbstractAnimation::Stopped)
                horizontal_scroll_target = m_currentView->horizontalScrollBar()->value();

            horizontal_scroll_target -= deltaX;
            horizontal_scroll_target = qBound(0, horizontal_scroll_target,
                                              m_currentView->horizontalScrollBar()->maximum());

            horizontal_scroll->stop();
            horizontal_scroll->setStartValue(m_currentView->horizontalScrollBar()->value());
            horizontal_scroll->setEndValue(horizontal_scroll_target);
            horizontal_scroll->start();
        }

        return true;
    }

    if (obj == m_currentView->viewport() && event->type() == QEvent::MouseButtonPress) {
        QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);

        QModelIndex index = m_currentView->indexAt(mouseEvent->pos());
        if (!index.isValid()) {
            m_currentView->selectionModel()->clearSelection();
            m_currentView->setCurrentIndex(QModelIndex());
        }
    }

    if (obj == grid_view->viewport() && event->type() == QEvent::Resize) {
        grid_view->UpdateIconSize();
        return true;
    }

    if (obj == carousel_view->viewport() && event->type() == QEvent::Resize) {
        carousel_view->UpdateIconSize();
        return true;
    }

    return QWidget::eventFilter(obj, event);
}

GameListPlaceholder::GameListPlaceholder(MainWindow* parent) : QWidget{parent} {
    layout = new QVBoxLayout;
    image = new QLabel;
    text = new QLabel;
    layout->setAlignment(Qt::AlignCenter);
    image->setPixmap(QIcon::fromTheme(QStringLiteral("plus_folder")).pixmap(200));

    RetranslateUI();
    QFont font = text->font();
    font.setPointSize(20);
    text->setFont(font);
    text->setAlignment(Qt::AlignHCenter);
    image->setAlignment(Qt::AlignHCenter);

    layout->addWidget(image);
    layout->addWidget(text);
    setLayout(layout);
}

GameListPlaceholder::~GameListPlaceholder() = default;

void GameListPlaceholder::mouseDoubleClickEvent(QMouseEvent* event) {
    emit GameListPlaceholder::AddDirectory();
}

void GameListPlaceholder::changeEvent(QEvent* event) {
    if (event->type() == QEvent::LanguageChange) {
        RetranslateUI();
    }

    QWidget::changeEvent(event);
}

void GameListPlaceholder::RetranslateUI() {
    text->setText(tr("Double-click to add a new folder to the game list"));
}
