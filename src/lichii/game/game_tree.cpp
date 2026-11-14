// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>

#include <QApplication>
#include <QHeaderView>
#include <QPainter>
#include <QScroller>
#include <QScrollerProperties>
#include <QStyledItemDelegate>

#include "qt_common/config/uisettings.h"
#include "qt_common/game_list/game_list_p.h"
#include "qt_common/game_list/model.h"
#include "lichii/game/common.h"
#include "lichii/game/game_tree.h"
#include "lichii/util/util.h"

namespace {

// Draws the name column the way Lichii Dark lays it out: a game's title over a muted
// monospace second line with a star for favourites, and a folder's game count after its path.
class LichiiNameDelegate final : public QStyledItemDelegate {
public:
    using QStyledItemDelegate::QStyledItemDelegate;

    void paint(QPainter* painter, const QStyleOptionViewItem& option,
               const QModelIndex& index) const override {
        const auto type = index.data(GameListItem::TypeRole).value<GameListItemType>();
        if (!IsLichiiTheme() || type == GameListItemType::AddDir) {
            QStyledItemDelegate::paint(painter, option, index);
            return;
        }

        QStyleOptionViewItem item = option;
        initStyleOption(&item, index);
        const QStringList lines =
            item.text.replace(QChar::LineSeparator, QLatin1Char('\n')).split(QLatin1Char('\n'));
        item.text.clear();
        if (const QPixmap picture = index.data(Qt::DecorationRole).value<QPixmap>();
            !picture.isNull()) {
            QIcon icon(picture);
            icon.addPixmap(picture, QIcon::Selected);
            item.icon = icon;
        }
        const QWidget* widget = item.widget;
        QStyle* style = widget != nullptr ? widget->style() : QApplication::style();
        style->drawControl(QStyle::CE_ItemViewItem, &item, painter, widget);
        const int text_left =
            style->subElementRect(QStyle::SE_ItemViewItemText, &item, widget).left() + 4;
        const QRect rect(text_left, item.rect.top(), item.rect.right() - text_left - 4,
                         item.rect.height());

        const bool selected = item.state & QStyle::State_Selected;
        QFont title_font = item.font;
        title_font.setWeight(QFont::DemiBold);

        painter->save();
        painter->setClipRect(item.rect);
        painter->setFont(title_font);
        const QFontMetrics title_metrics = painter->fontMetrics();
        const QString title =
            title_metrics.elidedText(lines.value(0), Qt::ElideRight, rect.width());
        painter->setPen(selected ? QColor(Qt::white) : QColor(0xec, 0xec, 0xef));

        if (type != GameListItemType::Game) {
            painter->drawText(rect, Qt::AlignVCenter | Qt::AlignLeft, title);
            const int games = index.model()->rowCount(index);
            const int x = rect.left() + title_metrics.horizontalAdvance(title) + 10;
            painter->setFont(item.font);
            painter->setPen(QColor(0x6f, 0x71, 0x78));
            painter->drawText(QRect(x, rect.top(), std::max(0, rect.right() - x), rect.height()),
                              Qt::AlignVCenter | Qt::AlignLeft,
                              games == 1
                                  ? QCoreApplication::translate("GameList", "1 game")
                                  : QCoreApplication::translate("GameList", "%1 games").arg(games));
            painter->restore();
            return;
        }

        QFont id_font = GetMonospaceFont();
        if (item.font.pixelSize() > 0) {
            id_font.setPixelSize(std::max(1, item.font.pixelSize() * 17 / 20));
        } else {
            id_font.setPointSizeF(item.font.pointSizeF() * 0.85);
        }
        const bool two_lines = lines.size() > 1;
        const int middle = rect.top() + rect.height() / 2;
        const int flags = Qt::AlignLeft | Qt::TextDontClip;

        QRect title_rect;
        painter->drawText(
            two_lines ? QRect(rect.left(), rect.top(), rect.width(), middle - rect.top() - 1)
                      : rect,
            flags | (two_lines ? Qt::AlignBottom : Qt::AlignVCenter), title, &title_rect);

        const u64 program_id = index.data(GameListItemPath::ProgramIdRole).toULongLong();
        if (program_id != 0 && UISettings::values.favorited_ids.contains(program_id)) {
            const int star_x = title_rect.right() + 8;
            const int star_y = title_rect.center().y() - 7;
            if (star_x + 14 <= rect.right()) {
                painter->drawPixmap(QRect(star_x, star_y, 14, 14),
                                    QPixmap(QStringLiteral(":/lichii/icons/star_accent@2x.png")));
            }
        }

        if (two_lines) {
            painter->setFont(id_font);
            painter->setPen(selected ? QColor(0xc4, 0xc5, 0xca) : QColor(0x6f, 0x71, 0x78));
            painter->drawText(
                QRect(rect.left(), middle + 2, rect.width(), rect.bottom() - middle - 2),
                flags | Qt::AlignTop,
                painter->fontMetrics().elidedText(lines.at(1).trimmed(), Qt::ElideRight,
                                                  rect.width()));
        }
        painter->restore();
    }

    QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const override {
        QSize size = QStyledItemDelegate::sizeHint(option, index);
        if (!IsLichiiTheme()) {
            return size;
        }
        const auto type = index.data(GameListItem::TypeRole).value<GameListItemType>();
        if (type == GameListItemType::Game) {
            const int icon = static_cast<int>(UISettings::values.game_icon_size.GetValue());
            size.setHeight(std::max(size.height(), icon + 16));
        } else {
            size.setHeight(std::max(size.height(), 34));
        }
        return size;
    }
};

} // namespace

GameTree::GameTree(QWidget* parent) : QTreeView{parent} {
    setItemDelegateForColumn(GameListModel::COLUMN_NAME, new LichiiNameDelegate(this));
    setAlternatingRowColors(true);
    setSelectionMode(QHeaderView::SingleSelection);
    setSelectionBehavior(QHeaderView::SelectRows);
    setVerticalScrollMode(QHeaderView::ScrollPerPixel);
    setHorizontalScrollMode(QHeaderView::ScrollPerPixel);
    setSortingEnabled(true);
    setEditTriggers(QHeaderView::NoEditTriggers);
    setContextMenuPolicy(Qt::CustomContextMenu);
    setAttribute(Qt::WA_AcceptTouchEvents, true);
    setStyleSheet(QStringLiteral("QTreeView{ border: none; }"));

    connect(this, &QTreeView::expanded, this, &GameTree::OnItemExpanded);
    connect(this, &QTreeView::collapsed, this, &GameTree::OnItemExpanded);
}

void GameTree::SetModel(GameListModel* model) {
    QTreeView::setModel(model);
    LoadInterfaceLayout();
    UpdateColumnVisibility(model);
}

void GameTree::OnItemExpanded(const QModelIndex& item) {
    const auto type = item.data(GameListItem::TypeRole).value<GameListItemType>();
    const bool is_dir = type == GameListItemType::CustomDir || type == GameListItemType::SdmcDir ||
                        type == GameListItemType::UserNandDir ||
                        type == GameListItemType::SysNandDir;
    const bool is_fave = type == GameListItemType::Favorites;
    if (!is_dir && !is_fave) {
        return;
    }
    const bool is_expanded = isExpanded(item);
    if (is_fave) {
        UISettings::values.favorites_expanded = is_expanded;
        return;
    }
    const int item_dir_index = item.data(GameListDir::GameDirRole).toInt();
    UISettings::values.game_dirs[item_dir_index].expanded = is_expanded;
}

void GameTree::SaveInterfaceLayout() {
    UISettings::values.gamelist_header_state = header()->saveState();
}

void GameTree::LoadInterfaceLayout() {
    auto* hdr = header();

    if (hdr->restoreState(UISettings::values.gamelist_header_state))
        return;

    hdr->resizeSection(GameListModel::COLUMN_NAME, 840);
}

void GameTree::UpdateColumnVisibility(GameListModel* model) {
    Q_UNUSED(model)
    setColumnHidden(GameListModel::COLUMN_ADD_ONS, !UISettings::values.show_add_ons);
    setColumnHidden(GameListModel::COLUMN_FILE_TYPE, !UISettings::values.show_types);
    setColumnHidden(GameListModel::COLUMN_SIZE, !UISettings::values.show_size);
    setColumnHidden(GameListModel::COLUMN_PLAY_TIME, !UISettings::values.show_play_time);
}

QString GameTree::GetLastFilterResultItem() const {
    QString file_path;

    auto* model = qobject_cast<GameListModel*>(QTreeView::model());
    if (!model)
        return {};

    for (int i = 1; i < model->rowCount() - 1; ++i) {
        const QStandardItem* folder = model->item(i, 0);
        const QModelIndex folder_index = folder->index();
        const int children_count = folder->rowCount();

        for (int j = 0; j < children_count; ++j) {
            if (isRowHidden(j, folder_index)) {
                continue;
            }

            const QStandardItem* child = folder->child(j, 0);
            file_path = child->data(GameListItemPath::FullPathRole).toString();
        }
    }

    return file_path;
}

int GameTree::FilterClosedResultCount(GameListModel* model) {
    int children_total = 0;

    auto hide_favorites_row = UISettings::values.favorited_ids.size() == 0;
    setRowHidden(0, model->invisibleRootItem()->index(), hide_favorites_row);

    for (int i = 1; i < model->rowCount() - 1; ++i) {
        auto* folder = model->item(i, 0);
        const QModelIndex folder_index = folder->index();
        const int children_count = folder->rowCount();
        for (int j = 0; j < children_count; ++j) {
            ++children_total;
            setRowHidden(j, folder_index, false);
        }
    }

    return children_total;
}

void GameTree::ApplyFilter(const QString& edit_filter_text, GameListModel* model) {
    int children_total = 0;
    int result_count = 0;

    if (edit_filter_text.isEmpty()) {
        children_total = FilterClosedResultCount(model);
        emit FilterResultReady(children_total, children_total);
        return;
    }

    setRowHidden(0, model->invisibleRootItem()->index(), true);

    for (int i = 1; i < model->rowCount() - 1; ++i) {
        auto* folder = model->item(i, 0);
        const QModelIndex folder_index = folder->index();
        const int children_count = folder->rowCount();

        for (int j = 0; j < children_count; ++j) {
            ++children_total;

            const QStandardItem* child = folder->child(j, 0);

            if (Lichii::FilterMatches(edit_filter_text, child)) {
                setRowHidden(j, folder_index, false);
                ++result_count;
            } else {
                setRowHidden(j, folder_index, true);
            }
        }
    }

    emit FilterResultReady(result_count, children_total);
}
