// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <QIcon>
#include <fmt/ranges.h>
#include "common/scm_rev.h"
#include "ui_aboutdialog.h"
#include "lichii/about_dialog.h"

AboutDialog::AboutDialog(QWidget* parent)
    : QDialog(parent), ui{std::make_unique<Ui::AboutDialog>()} {
    static const std::string build_id = std::string{Common::g_build_id};
    // Published builds show only their tag (build-N), so add the commit a report came from.
    static const std::string version =
        Common::g_is_dev_build
            ? std::string{Common::g_build_version}
            : fmt::format("{} ({:.10})", Common::g_build_version, Common::g_scm_rev);
    static const std::string lichii_build =
        fmt::format("{} | {} | {}", std::string{Common::g_build_name}, version,
                    std::string{Common::g_compiler_id});

    const auto override_build =
        fmt::format(fmt::runtime(std::string(Common::g_title_bar_format_idle)), build_id);
    const auto lichii_build_version = override_build.empty() ? lichii_build : override_build;

    ui->setupUi(this);
    // Try and request the icon from Qt theme (Linux?)
    const QIcon lichii_logo = QIcon::fromTheme(QStringLiteral("io.github.zortos293.lichii"),
                                           QIcon(QStringLiteral(":/img/lichii.ico")));
    if (!lichii_logo.isNull()) {
        ui->labelLogo->setPixmap(lichii_logo.pixmap(112));
    }
    ui->labelLichii->setPixmap(QIcon(QStringLiteral(":/img/lichii_wordmark.png")).pixmap(113, 48));
    ui->labelLichii->setAccessibleName(QStringLiteral("Lichii"));
    ui->verticalLayout_3->setContentsMargins(32, 28, 32, 20);
    ui->horizontalLayout->setSpacing(32);
    ui->verticalLayout->setSpacing(10);
    ui->labelBuildInfo->setText(
        ui->labelBuildInfo->text().arg(QString::fromStdString(lichii_build_version),
                                       QString::fromUtf8(Common::g_build_date).left(10)));
}

AboutDialog::~AboutDialog() = default;
