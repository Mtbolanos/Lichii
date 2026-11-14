// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <filesystem>
#include <QDialog>
#include "common/net/net.h"

class QRadioButton;
namespace Ui {
class UpdateDialog;
}

class UpdateDialog : public QDialog {
    Q_OBJECT

public:
    // exec() returns QDialog::Accepted to download, QDialog::Rejected to be reminded later.
    static constexpr int SkipVersion = 2;

    explicit UpdateDialog(const Common::Net::Release& release, QWidget* parent = nullptr);
    ~UpdateDialog();

    // Start `target` again once this process has exited, then close `window`.
    static void Restart(QWidget* window, const QString& target);

#ifdef _WIN32
    // Start the installer helper on the files under `staging`, then close `window`.
    static bool LaunchInstaller(QWidget* window, const QString& staging);
#endif

private slots:
    void Download();

private:
    // Download the asset to `filename`, verified against the published size and SHA-256.
    bool DownloadTo(const QString& filename);
    // Replace the running AppImage with the new one, keeping the old one for a rollback.
    void InstallInPlace(const std::filesystem::path& target);
#ifdef _WIN32
    // Download and unpack the new version beside the running one; a helper installs it on exit.
    void InstallOnRestart(const std::filesystem::path& exe);
#endif

    Ui::UpdateDialog* ui;
    QList<QRadioButton*> m_buttons;
    Common::Net::Asset m_asset;
};
