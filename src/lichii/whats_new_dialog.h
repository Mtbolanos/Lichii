// SPDX-FileCopyrightText: Copyright 2026 Lichii Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <QDialog>
#include "common/common_types.h"

class QCheckBox;

/// The quick start shown once after an update that changes the interface, and from Help.
class WhatsNewDialog : public QDialog {
    Q_OBJECT

public:
    /// Raise this when the dialog's content changes, so it shows once more after the update.
    static constexpr u32 Version = 1;

    explicit WhatsNewDialog(QWidget* parent);
    ~WhatsNewDialog() override;

    [[nodiscard]] bool OpenConfigureRequested() const {
        return open_configure;
    }

private:
    void done(int result) override;

    QCheckBox* dont_show_again{};
    bool open_configure = false;
};
