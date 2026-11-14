// SPDX-FileCopyrightText: Copyright 2026 Lichii Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <memory>
#include <QDialog>
#include <QFuture>
#include <QFutureWatcher>
#include <QString>

#include "core/debugger/nx2_bug_bundle.h"

class QDialogButtonBox;
class QLabel;
class QPlainTextEdit;
class QPushButton;

namespace Core {
class System;
}

/// Help → Report a bug and its hotkey: captures the moment the player asked, then asks for a
/// description and writes the bug-report bundle (docs/nx2/bug-bundle.md).
class BugReportDialog : public QDialog {
    Q_OBJECT

public:
    BugReportDialog(Core::System& system, QWidget* parent);
    ~BugReportDialog() override;

    /// Ends the capture before the emulation shuts down. Blocks until it has.
    void StopCapture();

private:
    struct Written {
        Core::Nx2Debug::BugBundle::Result result;
        QString error;
    };

    void Save();
    void ShowResult(const Written& written);
    void reject() override;

    std::shared_ptr<std::atomic<bool>> stop;
    std::shared_ptr<Core::Nx2Debug::BugBundle> bundle;
    /// The capture's error, empty when it ran.
    QFuture<QString> capture;
    QFutureWatcher<Written> writer;
    bool writing = false;

    QLabel* status{};
    QPlainTextEdit* description{};
    QDialogButtonBox* buttons{};
    QPushButton* save{};
    QPushButton* open_folder{};
};
