// SPDX-FileCopyrightText: 2016 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <memory>
#include <type_traits>
#include <vector>
#include <QCheckBox>
#include <QScreen>
#include <QScrollArea>
#include "lichii/configuration/configuration_shared.h"

namespace ConfigurationShared {

QIcon PageIcon(const QString& name) {
    QIcon icon;
    if (name.isEmpty()) {
        return icon;
    }
    const QString normal = QStringLiteral(":/lichii/icons/%1.png").arg(name);
    const QString active = QStringLiteral(":/lichii/icons/%1_active.png").arg(name);
    icon.addFile(normal, {}, QIcon::Normal, QIcon::Off);
    icon.addFile(active, {}, QIcon::Normal, QIcon::On);
    icon.addFile(active, {}, QIcon::Selected, QIcon::Off);
    icon.addFile(active, {}, QIcon::Selected, QIcon::On);
    return icon;
}

QScrollArea* ScrollTab(QWidget* tab) {
    auto* const area = new QScrollArea;
    area->setFrameShape(QFrame::NoFrame);
    area->setWidgetResizable(true);
    area->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    area->setWidget(tab);
    area->setAccessibleName(tab->accessibleName());
    return area;
}

void UseToggles(QWidget* root) {
    for (auto* const checkbox : root->findChildren<QCheckBox*>()) {
        checkbox->setProperty("toggle", true);
    }
}

void FitToScreen(QWidget* dialog, const QSize& preferred) {
    const QScreen* screen = dialog->screen();
    if (dialog->parentWidget() != nullptr) {
        screen = dialog->parentWidget()->screen();
    }
    const QSize available = screen->availableGeometry().size() * 9 / 10;
    dialog->resize(std::min(preferred.width(), available.width()),
                   std::min(preferred.height(), available.height()));
}

Tab::Tab(std::shared_ptr<std::vector<Tab*>> group, QWidget* parent) : QWidget(parent) {
    if (group != nullptr) {
        group->push_back(this);
    }
}

Tab::~Tab() = default;

} // namespace ConfigurationShared
