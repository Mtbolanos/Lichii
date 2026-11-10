// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/httplib.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <system_error>

#include <QApplication>
#include <QCryptographicHash>
#include <QDesktopServices>
#include <QDialog>
#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProgressDialog>
#include <QPushButton>
#include <QUrl>
#include <QVBoxLayout>
#include <fmt/format.h>

#include "common/common_types.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "video_core/vulkan_common/ffx_files.h"
#include "yuzu/configuration/amd_fsr4_panel.h"

#ifdef YUZU_BUNDLED_OPENSSL
#include <openssl/cert.h>
#endif

namespace {

namespace FfxRuntime = Vulkan::FfxRuntime;

QString Tr(const char* text) {
    return QCoreApplication::translate("AmdFsr4Panel", text);
}

QString PathText(const std::filesystem::path& path) {
    return QString::fromStdWString(path.wstring());
}

u64 DownloadSize() {
    u64 total = 0;
    for (const FfxRuntime::File& file : FfxRuntime::FILES) {
        total += file.size;
    }
    return total;
}

/// GETs one file of the SDK repository at the pinned commit. Empty with `error` set on failure.
std::optional<std::string> Fetch(std::string_view entry, std::string& error,
                                 const std::function<bool(u64)>& progress = {}) {
    httplib::Client client{std::string{FfxRuntime::DOWNLOAD_HOST}};
    client.set_connection_timeout(30);
    client.set_read_timeout(30);
#ifdef YUZU_BUNDLED_OPENSSL
    client.load_ca_cert_store(kCert, sizeof(kCert));
#endif
    std::string body;
    int status = 0;
    const auto result = client.Get(
        fmt::format("{}{}", FfxRuntime::DOWNLOAD_PATH, entry),
        [&status](const httplib::Response& response) {
            status = response.status;
            return response.status == 200;
        },
        [&](const char* data, size_t length) {
            body.append(data, length);
            return !progress || progress(body.size());
        });
    if (!result || status != 200) {
        error = result ? fmt::format("HTTP {}", status) : httplib::to_string(result.error());
        return std::nullopt;
    }
    return body;
}

} // Anonymous namespace

AmdFsr4Panel::AmdFsr4Panel(QWidget* parent) : QWidget(parent) {
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->setContentsMargins(24, 0, 0, 8);
    layout->setSpacing(6);

    QHBoxLayout* status_row = new QHBoxLayout;
    status_row->setSpacing(8);
    icon_label = new QLabel(this);
    icon_label->setTextFormat(Qt::RichText);
    icon_label->setAlignment(Qt::AlignTop | Qt::AlignHCenter);
    icon_label->setFixedWidth(16);
    status_label = new QLabel(this);
    status_label->setTextFormat(Qt::RichText);
    status_label->setWordWrap(true);
    status_row->addWidget(icon_label);
    status_row->addWidget(status_label, 1);

    QHBoxLayout* button_row = new QHBoxLayout;
    button_row->setContentsMargins(24, 0, 0, 0);
    download_button = new QPushButton(this);
    folder_button = new QPushButton(Tr("Open Folder"), this);
    QLabel* about = new QLabel(
        QStringLiteral("<a href=\"%1\">%2</a>")
            .arg(QString::fromUtf8(FfxRuntime::RELEASE_PAGE.data(),
                                   static_cast<qsizetype>(FfxRuntime::RELEASE_PAGE.size())),
                 Tr("Where do these files come from?")),
        this);
    about->setOpenExternalLinks(true);
    about->setToolTip(Tr("AMD's FidelityFX SDK release on GitHub. The files are in "
                         "Kits\\FidelityFX\\signedbin."));
    button_row->addWidget(download_button);
    button_row->addWidget(folder_button);
    button_row->addStretch(1);
    button_row->addWidget(about);

    layout->addLayout(status_row);
    layout->addLayout(button_row);

    connect(download_button, &QPushButton::clicked, this, [this] { Download(); });
    connect(folder_button, &QPushButton::clicked, this, [] {
        const std::filesystem::path folder = FfxRuntime::Folder();
        std::error_code ec;
        std::filesystem::create_directories(folder, ec);
        QDesktopServices::openUrl(QUrl::fromLocalFile(PathText(folder)));
    });
    Refresh();
}

void AmdFsr4Panel::showEvent(QShowEvent* event) {
    Refresh();
    QWidget::showEvent(event);
}

void AmdFsr4Panel::Refresh() {
    const bool ready = FfxRuntime::FilesPresent();
    const QString version = QString::fromUtf8(FfxRuntime::SDK_VERSION.data(),
                                              static_cast<qsizetype>(FfxRuntime::SDK_VERSION.size()));
    icon_label->setText(ready ? QStringLiteral("<span style=\"color:#2ea043\">&#10004;</span>")
                              : QStringLiteral("<span style=\"color:#d29922\">&#9888;</span>"));
    status_label->setText(
        ready ? Tr("<b>Ready.</b> Found AMD's FidelityFX runtime from FidelityFX SDK %1. FSR 4 runs "
                   "if AMD's driver offers it for this GPU; otherwise FSR 3.1 does.")
                    .arg(version)
              : Tr("<b>AMD's files needed:</b> the FidelityFX runtime from AMD's FidelityFX SDK %1 "
                   "(amd_fidelityfx_loader_vk.dll and amd_fidelityfx_upscaler_vk.dll). Until then, "
                   "FSR 3.1 is used.")
                    .arg(version));
    status_label->setToolTip(Tr("Folder: %1").arg(PathText(FfxRuntime::Folder())));
    download_button->setText(
        Tr("Download from AMD (%1 MB)").arg((DownloadSize() + (1 << 20) - 1) >> 20));
    download_button->setVisible(!ready);
}

void AmdFsr4Panel::Download() {
    const std::filesystem::path folder = FfxRuntime::Folder();
    const auto fail = [this](const std::string& error) {
        LOG_ERROR(Frontend, "AMD FSR 4 download failed: {}", error);
        QMessageBox::warning(this, Tr("AMD FSR 4"),
                             Tr("Could not download AMD's files: %1. FSR 3.1 is used.")
                                 .arg(QString::fromStdString(error)));
        Refresh();
    };

    std::string error;
    std::optional<std::string> licence;
    {
        QProgressDialog busy(Tr("Contacting AMD's download..."), QString{}, 0, 0, this);
        busy.setWindowTitle(Tr("AMD FSR 4"));
        busy.setWindowModality(Qt::WindowModal);
        busy.setMinimumDuration(0);
        busy.show();
        QApplication::processEvents();
        licence = Fetch(FfxRuntime::LICENSE_ENTRY, error);
    }
    if (!licence || licence->empty()) {
        fail(error.empty() ? "could not retrieve AMD's licence text" : error);
        return;
    }

    QDialog dialog(this);
    dialog.setWindowTitle(Tr("Download AMD FSR 4"));
    QVBoxLayout* layout = new QVBoxLayout(&dialog);
    QLabel* intro = new QLabel(
        Tr("Lichii will download amd_fidelityfx_loader_vk.dll and amd_fidelityfx_upscaler_vk.dll "
           "from AMD's FidelityFX SDK %1 release on GitHub into %2.<br>These files are AMD's, under "
           "AMD's licence terms below. Lichii checks AMD's signature before using them.")
            .arg(QString::fromUtf8(FfxRuntime::SDK_VERSION.data(),
                                   static_cast<qsizetype>(FfxRuntime::SDK_VERSION.size())),
                 PathText(folder)),
        &dialog);
    intro->setWordWrap(true);
    QPlainTextEdit* licence_text = new QPlainTextEdit(QString::fromStdString(*licence), &dialog);
    licence_text->setReadOnly(true);
    licence_text->setMinimumSize(640, 380);
    QDialogButtonBox* buttons = new QDialogButtonBox(&dialog);
    buttons->addButton(Tr("Accept and Download"), QDialogButtonBox::AcceptRole);
    buttons->addButton(QDialogButtonBox::Cancel);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    layout->addWidget(intro);
    layout->addWidget(licence_text, 1);
    layout->addWidget(buttons);
    if (dialog.exec() != QDialog::Accepted) {
        return;
    }

    QProgressDialog progress(Tr("Downloading AMD's FSR 4 files..."), Tr("Cancel"), 0, 1000, this);
    progress.setWindowTitle(Tr("AMD FSR 4"));
    progress.setWindowModality(Qt::WindowModal);
    progress.setMinimumDuration(0);
    progress.setValue(0);
    QApplication::processEvents();
    std::error_code ec;
    std::filesystem::create_directories(folder, ec);
    const u64 total = DownloadSize();
    u64 done = 0;
    for (const FfxRuntime::File& file : FfxRuntime::FILES) {
        const std::optional<std::string> contents = Fetch(file.entry, error, [&](u64 received) {
            progress.setValue(static_cast<int>((done + received) * 1000 / total));
            QApplication::processEvents();
            return !progress.wasCanceled();
        });
        if (progress.wasCanceled()) {
            return;
        }
        if (!contents) {
            progress.close();
            fail(error);
            return;
        }
        done += file.size;
        const QByteArray sha256 =
            QCryptographicHash::hash(QByteArrayView{contents->data(),
                                                    static_cast<qsizetype>(contents->size())},
                                     QCryptographicHash::Sha256)
                .toHex();
        if (contents->size() != file.size ||
            sha256 != QByteArray{file.sha256.data(), static_cast<qsizetype>(file.sha256.size())}) {
            progress.close();
            fail(fmt::format("{} doesn't match the FidelityFX SDK {} file", file.name,
                             FfxRuntime::SDK_VERSION));
            return;
        }
        const std::filesystem::path target = folder / file.name;
        std::filesystem::path partial = target;
        partial += L".download";
        {
            std::ofstream stream(partial, std::ios::binary | std::ios::trunc);
            stream.write(contents->data(), static_cast<std::streamsize>(contents->size()));
            if (!stream) {
                progress.close();
                fail("could not write " + Common::FS::PathToUTF8String(partial));
                return;
            }
        }
        if (!FfxRuntime::IsSignedByAmd(partial)) {
            std::filesystem::remove(partial, ec);
            progress.close();
            fail(fmt::format("{} is not signed by AMD", file.name));
            return;
        }
        std::filesystem::rename(partial, target, ec);
        if (ec) {
            std::filesystem::remove(partial, ec);
            progress.close();
            fail("could not save " + Common::FS::PathToUTF8String(target));
            return;
        }
        LOG_INFO(Frontend, "Downloaded {} from AMD's FidelityFX SDK {}",
                 Common::FS::PathToUTF8String(target), FfxRuntime::SDK_VERSION);
    }
    {
        std::ofstream stream(folder / FfxRuntime::LICENSE_FILE, std::ios::binary | std::ios::trunc);
        stream.write(licence->data(), static_cast<std::streamsize>(licence->size()));
    }
    progress.setValue(1000);
    progress.close();
    Refresh();
    QMessageBox::information(this, Tr("AMD FSR 4"),
                             Tr("AMD's FSR 4 runtime is ready. It is used the next time a game "
                                "starts."));
}
