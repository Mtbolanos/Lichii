// SPDX-FileCopyrightText: Copyright 2026 Lichii Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <QDesktopServices>
#include <QDialogButtonBox>
#include <QLabel>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QUrl>
#include <QVBoxLayout>
#include <QtConcurrentRun>

#include "lichii/bug_report_dialog.h"

BugReportDialog::BugReportDialog(Core::System& system, QWidget* parent)
    : QDialog(parent), stop{std::make_shared<std::atomic<bool>>(false)},
      bundle{std::make_shared<Core::Nx2Debug::BugBundle>(
          system, Core::Nx2Debug::BugBundle::Options{.trigger = "ui"})} {
    setWindowTitle(tr("Report a Bug"));
    setMinimumWidth(520);

    capture = QtConcurrent::run([bundle = bundle, stop = stop] {
        try {
            bundle->Capture(*stop);
            return QString{};
        } catch (const std::exception& failure) {
            return QString::fromStdString(failure.what());
        }
    });

    auto* const layout = new QVBoxLayout(this);
    auto* const intro = new QLabel(
        tr("Lichii is saving this moment: the log, your settings, a screenshot and the next "
           "frame with its shaders. Describe what went wrong and what you expected."));
    intro->setWordWrap(true);
    layout->addWidget(intro);

    description = new QPlainTextEdit;
    description->setPlaceholderText(
        tr("For example: the sky turns white after the first cutscene in world 2."));
    layout->addWidget(description, 1);

    auto* const privacy = new QLabel(
        tr("<b>Send this report privately to the Lichii maintainers</b> (the log bot's /bug "
           "command or a direct message), not to a public channel or issue. It holds the "
           "game's textures and shaders from one frame. Your user name is removed from paths "
           "and online service settings are left out."));
    privacy->setWordWrap(true);
    layout->addWidget(privacy);

    status = new QLabel;
    status->setWordWrap(true);
    status->setTextInteractionFlags(Qt::TextSelectableByMouse);
    status->hide();
    layout->addWidget(status);

    buttons = new QDialogButtonBox;
    save = buttons->addButton(tr("Save Report"), QDialogButtonBox::AcceptRole);
    open_folder = buttons->addButton(tr("Open Folder"), QDialogButtonBox::ActionRole);
    open_folder->hide();
    buttons->addButton(QDialogButtonBox::Cancel);
    layout->addWidget(buttons);

    connect(save, &QPushButton::clicked, this, &BugReportDialog::Save);
    connect(buttons, &QDialogButtonBox::rejected, this, &BugReportDialog::reject);
    connect(open_folder, &QPushButton::clicked, this, [] {
        QDesktopServices::openUrl(QUrl::fromLocalFile(QString::fromStdString(
            Core::Nx2Debug::BugBundle::DefaultDirectory().generic_string())));
    });
    connect(&writer, &QFutureWatcher<Written>::finished, this,
            [this] { ShowResult(writer.result()); });
}

BugReportDialog::~BugReportDialog() {
    StopCapture();
    writer.waitForFinished();
}

void BugReportDialog::StopCapture() {
    stop->store(true);
    capture.waitForFinished();
}

void BugReportDialog::Save() {
    writing = true;
    save->setEnabled(false);
    description->setReadOnly(true);
    buttons->button(QDialogButtonBox::Cancel)->setEnabled(false);
    status->setText(tr("Writing the report. The frame capture can take a minute in a busy scene."));
    status->show();
    writer.setFuture(QtConcurrent::run([bundle = bundle, capture = capture,
                                        text = description->toPlainText().toStdString()] {
        Written written;
        QFuture<QString> pending = capture;
        written.error = pending.result();
        if (!written.error.isEmpty()) {
            return written;
        }
        try {
            written.result = bundle->Write(text);
        } catch (const std::exception& failure) {
            written.error = QString::fromStdString(failure.what());
        }
        return written;
    }));
}

void BugReportDialog::ShowResult(const Written& written) {
    writing = false;
    auto* const close = buttons->button(QDialogButtonBox::Cancel);
    close->setEnabled(true);
    close->setText(tr("Close"));
    save->hide();
    if (!written.error.isEmpty()) {
        status->setText(tr("The report could not be written: %1").arg(written.error));
        return;
    }
    QString text = tr("Saved %1 (%2 MiB).")
                       .arg(QString::fromStdString(written.result.path.generic_string()))
                       .arg(static_cast<double>(written.result.bytes) / (1 << 20), 0, 'f', 1);
    if (!written.result.frame_error.empty()) {
        text += QStringLiteral("\n") + tr("No frame capture: %1")
                                            .arg(QString::fromStdString(written.result.frame_error));
    }
    if (!written.result.dropped.empty()) {
        text += QStringLiteral("\n") + tr("Left out to fit the size limit: %1")
                                            .arg(QString::number(written.result.dropped.size()));
    }
    status->setText(text);
    open_folder->show();
}

void BugReportDialog::reject() {
    if (!writing) {
        QDialog::reject();
    }
}
