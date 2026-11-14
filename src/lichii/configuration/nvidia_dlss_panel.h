// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Shown under the NVIDIA DLSS setting: which of NVIDIA's files were found, and
// a button that fetches missing ones from NVIDIA's own Streamline release.

#pragma once

#ifdef EGLANTINE_STREAMLINE

#include <QWidget>

class QLabel;
class QPushButton;

class NvidiaDlssPanel : public QWidget {
public:
    explicit NvidiaDlssPanel(QWidget* parent);

    void Refresh();

    /// Whether DLSS Frame Generation is the chosen frame generation type: the panel then lists
    /// its files and downloads them with the rest.
    void SetFrameGenWanted(bool wanted);

protected:
    void showEvent(QShowEvent* event) override;

private:
    void Download();

    QLabel* icon_label;
    QLabel* status_label;
    QPushButton* download_button;
    QPushButton* folder_button;
    bool frame_gen_wanted{};
};

#endif
