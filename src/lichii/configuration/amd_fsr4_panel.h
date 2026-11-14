// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Shown under the FSR 4 setting. On Windows: whether AMD's FidelityFX runtime is downloaded, and a
// button that fetches it from AMD's FidelityFX SDK release on GitHub. On Linux: whether the fsr4-vk
// port is in the package.

#pragma once

#include <QWidget>

class QLabel;
class QPushButton;

class AmdFsr4Panel : public QWidget {
public:
    explicit AmdFsr4Panel(QWidget* parent);

    void Refresh();

protected:
    void showEvent(QShowEvent* event) override;

private:
    void Download();

    QLabel* icon_label;
    QLabel* status_label;
    QPushButton* download_button;
    QPushButton* folder_button;
};
