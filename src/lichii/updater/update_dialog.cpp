// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <filesystem>
#include <QTimer>
#include <QEventLoop>
#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QProcess>
#include <QProcessEnvironment>
#include <QPushButton>
#include <QRadioButton>
#include <QSaveFile>
#include <QStandardPaths>
#include <qdesktopservices.h>
#include "common/logging.h"
#include "common/scm_rev.h"
#include "frontend_common/update_installer.h"
#include "qt_common/abstract/frontend.h"
#include "qt_common/abstract/progress.h"
#ifdef _WIN32
#include "qt_common/util/compress.h"
#endif
#include "ui_update_dialog.h"
#include "update_dialog.h"

#include "common/httplib.h"

#ifdef LICHII_BUNDLED_OPENSSL
#include <openssl/cert.h>
#endif

#include <QDesktopServices>

#undef GetSaveFileName

UpdateDialog::UpdateDialog(const Common::Net::Release& release, QWidget* parent)
    : QDialog(parent), ui(new Ui::UpdateDialog) {
    ui->setupUi(this);

    ui->version->setText(
        tr("%1 is available for download.").arg(QString::fromStdString(release.title)));
    ui->url->setText(
        tr("<a href=\"%1\">View on GitHub</a>").arg(QString::fromStdString(release.html_url)));

    std::string text{release.body};
    if (auto pos = text.find("# Packages"); pos != std::string::npos) {
        text = text.substr(0, pos);
    }

    ui->body->setMarkdown(QString::fromStdString(text));

    ui->buttonBox->button(QDialogButtonBox::No)->setText(tr("Remind Me Later"));
    auto* skip = ui->buttonBox->addButton(tr("Skip This Version"), QDialogButtonBox::ActionRole);
    connect(skip, &QPushButton::clicked, this, [this] { done(SkipVersion); });

    // TODO(crueter): Find a way to set default
    const auto assets = release.GetPlatformAssets();

    if (assets.empty()) {
        ui->groupBox->setHidden(true);
        connect(this, &QDialog::accepted, this, [release]() {
            QDesktopServices::openUrl(QUrl{QString::fromStdString(release.html_url)});
        });
    } else if (assets.size() == 1) {
        ui->groupBox->setHidden(true);
        m_asset = assets[0];

        if (m_asset.name == "Standard" &&
            std::string_view{Common::g_build_release_platform}.ends_with("-pgo")) {
            ui->version->setText(
                ui->version->text() +
                tr(" This release has no PGO build, so the standard build will be downloaded."));
        }

        connect(this, &QDialog::accepted, this, &UpdateDialog::Download);
    } else {
        u32 i = 0;
        for (const Common::Net::Asset& a : assets) {
            QRadioButton* r = new QRadioButton(tr(a.name.c_str()), this);
            connect(r, &QRadioButton::toggled, this, [a, this](bool checked) {
                if (checked)
                    m_asset = a;
            });

            if (i == 0)
                r->setChecked(true);
            ++i;

            ui->radioButtons->addWidget(r);
        }

        connect(this, &QDialog::accepted, this, &UpdateDialog::Download);
    }
}

UpdateDialog::~UpdateDialog() {
    delete ui;
}

void UpdateDialog::Restart(QWidget* window, const QString& target) {
    auto environment = QProcessEnvironment::systemEnvironment();
    const auto app_dir = environment.value(QStringLiteral("APPDIR"));
    if (!app_dir.isEmpty()) {
        for (const auto& key : environment.keys()) {
            const auto value = environment.value(key);
            if (!value.contains(app_dir)) {
                continue;
            }
            const auto stripped = UpdateInstaller::StripPathListEntries(value.toStdString(),
                                                                        app_dir.toStdString());
            if (stripped.empty()) {
                environment.remove(key);
            } else {
                environment.insert(key, QString::fromStdString(stripped));
            }
        }
    }
    for (const auto& key : {"APPDIR", "APPIMAGE", "ARGV0", "OWD"}) {
        environment.remove(QString::fromLatin1(key));
    }

    // Wait for this process to exit, so the new one reads the settings this one writes on close.
    QProcess process;
    process.setProgram(QStringLiteral("/bin/sh"));
    process.setArguments(
        {QStringLiteral("-c"),
         QStringLiteral("i=0; while kill -0 \"$1\" 2>/dev/null && [ \"$i\" -lt 1500 ]; do "
                        "sleep 0.2; i=$((i+1)); done; kill -0 \"$1\" 2>/dev/null && exit 0; "
                        "exec \"$2\""),
         QStringLiteral("sh"), QString::number(QCoreApplication::applicationPid()), target});
    process.setProcessEnvironment(environment);
    if (!process.startDetached()) {
        LOG_ERROR(Frontend, "Could not start the restart helper for {}", target.toStdString());
        QtCommon::Frontend::Critical(tr("Failed to restart"),
                                     tr("Please start %1 again yourself.").arg(target));
        return;
    }
    if (window) {
        QMetaObject::invokeMethod(window, &QWidget::close, Qt::QueuedConnection);
    }
}

#ifdef _WIN32
bool UpdateDialog::LaunchInstaller(QWidget* window, const QString& staging) {
    // The helper runs from a copy, because the installed one is replaced along with everything else.
    const QString helper =
        QCoreApplication::applicationDirPath() + QStringLiteral("/lichii-updater.exe");
    const QString copy = QDir::temp().filePath(
        QStringLiteral("lichii-updater-%1.exe").arg(QCoreApplication::applicationPid()));
    QFile::remove(copy);
    if (!QFile::copy(helper, copy)) {
        LOG_ERROR(Frontend, "Could not copy the installer helper {}", helper.toStdString());
        QtCommon::Frontend::Critical(tr("Failed to install the update"),
                                     tr("Could not start the update installer."));
        return false;
    }

    const auto install_dir = QCoreApplication::applicationDirPath();
    const auto native = [](const QString& path) { return QDir::toNativeSeparators(path); };
    if (!QProcess::startDetached(
            copy, {QString::number(QCoreApplication::applicationPid()), native(staging),
                   native(install_dir), native(QCoreApplication::applicationFilePath())})) {
        LOG_ERROR(Frontend, "Could not start the installer helper {}", copy.toStdString());
        QtCommon::Frontend::Critical(tr("Failed to install the update"),
                                     tr("Could not start the update installer."));
        return false;
    }
    if (window) {
        QMetaObject::invokeMethod(window, &QWidget::close, Qt::QueuedConnection);
    }
    return true;
}
#endif

void UpdateDialog::Download() {
#ifdef _WIN32
    if (m_asset.filename.ends_with(".zip")) {
        const std::filesystem::path exe{QCoreApplication::applicationFilePath().toStdWString()};
        if (UpdateInstaller::CanReplace(exe) &&
            QFile::exists(QCoreApplication::applicationDirPath() +
                          QStringLiteral("/lichii-updater.exe"))) {
            InstallOnRestart(exe);
            return;
        }
    }
#endif
#ifdef __linux__
    if (Common::Net::CurrentLinuxPackage() == Common::Net::LinuxPackage::AppImage &&
        m_asset.filename.ends_with(".AppImage")) {
        const std::filesystem::path target{qEnvironmentVariable("APPIMAGE").toStdString()};
        if (UpdateInstaller::CanReplace(target)) {
            InstallInPlace(target);
            return;
        }
    }
#endif

    // The application folder is read-only inside an AppImage and is the app bundle on macOS.
    auto default_dir = QStandardPaths::writableLocation(QStandardPaths::DownloadLocation);
    if (default_dir.isEmpty()) {
        default_dir = QDir::homePath();
    }

    const auto filename = QtCommon::Frontend::GetSaveFileName(
        tr("New Version Location"),
        default_dir % QStringLiteral("/") % QString::fromStdString(m_asset.filename),
        tr("All Files (*.*)"));

    if (filename.isEmpty() || !DownloadTo(filename))
        return;

    // Download is complete. User may choose to open in the file manager.
    auto button =
        QtCommon::Frontend::Question(tr("Download Complete"),
                                     tr("Successfully downloaded %1. Would you like to open it?")
                                         .arg(QString::fromStdString(m_asset.filename)),
                                     QtCommon::Frontend::Yes | QtCommon::Frontend::No);

    if (button == QtCommon::Frontend::Yes) {
        QDesktopServices::openUrl(QUrl::fromLocalFile(filename));
    }
}

void UpdateDialog::InstallInPlace(const std::filesystem::path& target) {
    const UpdateInstaller::Lock lock{UpdateInstaller::LockPath(target)};
    if (!lock.Held()) {
        QtCommon::Frontend::Critical(tr("Failed to install the update"),
                                     tr("Another Lichii window is already updating this "
                                        "installation."));
        return;
    }

    const auto staged = UpdateInstaller::StagedPath(target);
    if (!DownloadTo(QString::fromStdString(staged.string()))) {
        return;
    }

    std::error_code ec;
    if (!UpdateInstaller::Swap(staged, target, ec)) {
        LOG_ERROR(Frontend, "Could not replace {}: {}", target.string(), ec.message());
        std::filesystem::remove(staged, ec);
        QtCommon::Frontend::Critical(
            tr("Failed to install the update"),
            tr("Could not replace %1. Your installation was not changed.")
                .arg(QString::fromStdString(target.string())));
        return;
    }

    LOG_INFO(Frontend, "Installed {} over {}", m_asset.filename, target.string());
    const auto button = QtCommon::Frontend::Question(
        tr("Update Installed"),
        tr("The new version is installed. The previous version is kept as %1.\n\nRestart now?")
            .arg(QString::fromStdString(UpdateInstaller::PreviousPath(target).string())),
        QtCommon::Frontend::Yes | QtCommon::Frontend::No);
    if (button == QtCommon::Frontend::Yes) {
        Restart(parentWidget(), QString::fromStdString(target.string()));
    }
}

#ifdef _WIN32
void UpdateDialog::InstallOnRestart(const std::filesystem::path& exe) {
    const auto install_dir = exe.parent_path();
    const auto staging = install_dir / ".update-staging";
    const auto archive = install_dir / ".update-download.zip";
    const auto cleanup = [&] {
        std::error_code ec;
        std::filesystem::remove_all(staging, ec);
        std::filesystem::remove(archive, ec);
    };
    const auto fail = [&](const QString& text) {
        cleanup();
        QtCommon::Frontend::Critical(tr("Failed to install the update"), text);
    };

    cleanup();
    std::error_code ec;
    std::filesystem::create_directories(staging, ec);
    if (ec) {
        fail(tr("Could not create a folder for the update."));
        return;
    }
    if (!DownloadTo(QString::fromStdWString(archive.wstring()))) {
        cleanup();
        return;
    }

    auto progress = QtCommon::Frontend::newProgressDialog(tr("Unpacking..."), tr("Cancel"), 0, 100);
    progress->show();
    const auto unpacked = QtCommon::Compress::extractDir(
        QString::fromStdWString(archive.wstring()), QString::fromStdWString(staging.wstring()),
        [&](std::size_t total, std::size_t done) {
            QGuiApplication::processEvents();
            if (total != 0) {
                progress->setValue(static_cast<int>((done * 100) / total));
            }
            return !progress->wasCanceled();
        });
    progress->close();
    if (unpacked.isEmpty() || !std::filesystem::exists(staging / exe.filename(), ec)) {
        fail(tr("The downloaded update could not be unpacked."));
        return;
    }
    std::filesystem::remove(archive, ec);

    const auto button = QtCommon::Frontend::Question(
        tr("Update Ready"),
        tr("Lichii will close, install the update and start again. The previous version is "
           "kept, and your settings and data are not changed.\n\nRestart now?"),
        QtCommon::Frontend::Yes | QtCommon::Frontend::No);
    if (button != QtCommon::Frontend::Yes ||
        !LaunchInstaller(parentWidget(), QString::fromStdWString(staging.wstring()))) {
        cleanup();
    }
}
#endif

// TODO: migrate to a net.cpp wrapper
bool UpdateDialog::DownloadTo(const QString& filename) {
    QSaveFile file(filename);
    if (!file.open(QIODevice::Truncate | QIODevice::WriteOnly)) {
        LOG_WARNING(Frontend, "Could not open file {}", filename.toStdString());
        QtCommon::Frontend::Critical(tr("Failed to save file"),
                                     tr("Could not open file %1 for writing.").arg(filename));
        return false;
    }

    // TODO(crueter): Move to net.cpp
    constexpr std::size_t timeout_seconds = 15;

    std::unique_ptr<httplib::Client> client = std::make_unique<httplib::Client>(m_asset.url);
    client->set_connection_timeout(timeout_seconds);
    client->set_read_timeout(timeout_seconds);
    client->set_write_timeout(timeout_seconds);
    client->set_follow_location(true);

#ifdef LICHII_BUNDLED_OPENSSL
    client->load_ca_cert_store(kCert, sizeof(kCert));
#endif

    if (client == nullptr) {
        LOG_ERROR(Frontend, "Invalid URL {}{}", m_asset.url, m_asset.path);
        return false;
    }

    auto progress =
        QtCommon::Frontend::newProgressDialog(tr("Downloading..."), tr("Cancel"), 0, 100);
    progress->show();

    QGuiApplication::processEvents();

    // Progress dialog.
    auto progress_callback = [&](size_t processed_size, size_t total_size) {
        QGuiApplication::processEvents();
        if (total_size != 0) {
            progress->setValue(static_cast<int>((processed_size * 100) / total_size));
        }
        return !progress->wasCanceled();
    };

    QCryptographicHash hash(QCryptographicHash::Sha256);
    u64 received = 0;

    // Write file in chunks.
    auto content_receiver = [&file, &hash, &received, filename](const char* t_data,
                                                                size_t data_length) -> bool {
        hash.addData(QByteArrayView(t_data, static_cast<qsizetype>(data_length)));
        received += data_length;
        if (file.write(t_data, data_length) == -1) {
            LOG_WARNING(Frontend, "Could not write {} bytes to file {}", data_length,
                        filename.toStdString());
            QtCommon::Frontend::Critical(tr("Failed to save file"),
                                         tr("Could not write to file %1.").arg(filename));
            return false;
        }

        return true;
    };

    // Now send off request. GitHub's file server sometimes answers 502-504 or drops the
    // connection for a moment (v5.2's release hour): try up to three times, a few seconds apart,
    // starting the file over each time. The size and SHA-256 checks below still decide.
    constexpr int attempts = 3;
    httplib::Result result;
    for (int attempt = 1; attempt <= attempts; ++attempt) {
        if (attempt > 1) {
            file.seek(0);
            file.resize(0);
            hash.reset();
            received = 0;
            progress->setValue(0);
            progress->setLabelText(tr("Downloading... (try %1 of %2)").arg(attempt).arg(attempts));
            QEventLoop wait;
            QTimer::singleShot(2000 * (attempt - 1), &wait, &QEventLoop::quit);
            wait.exec();
            if (progress->wasCanceled()) {
                break;
            }
        }
        result = client->Get(m_asset.path, content_receiver, progress_callback);
        const bool canceled = !result && result.error() == httplib::Error::Canceled;
        const bool transient = (!result && !canceled) ||
                               (result && (result->status >= 500 || result->status == 429));
        if (!transient || attempt == attempts) {
            break;
        }
        LOG_WARNING(Frontend, "GET to {}{} failed ({}), trying again", m_asset.url, m_asset.path,
                    result ? std::to_string(result->status) : httplib::to_string(result.error()));
    }
    progress->close();

    if (!result) {
        LOG_ERROR(Frontend, "GET to {}{} returned null", m_asset.url, m_asset.path);
        return false;
    }

    const auto& response = result.value();
    if (response.status >= 400) {
        LOG_ERROR(Frontend, "GET to {}{} returned error status code: {}", m_asset.url, m_asset.path,
                  response.status);
        QtCommon::Frontend::Critical(tr("Failed to download file"),
                                     tr("Could not download from %1%2\nError code: %3")
                                         .arg(QString::fromStdString(m_asset.url),
                                              QString::fromStdString(m_asset.path),
                                              QString::number(response.status)));
        return false;
    }
    if (!response.has_header("content-type")) {
        LOG_ERROR(Frontend, "GET to {}{} returned no content", m_asset.url, m_asset.path);
        return false;
    }

    const bool wrong_size = m_asset.size != 0 && received != m_asset.size;
    const bool wrong_hash =
        !m_asset.sha256.empty() &&
        hash.result().toHex() != QByteArray::fromStdString(m_asset.sha256);
    if (wrong_size || wrong_hash) {
        LOG_ERROR(Frontend, "Downloaded {} does not match the published size or SHA-256",
                  m_asset.filename);
        file.cancelWriting();
        QtCommon::Frontend::Critical(
            tr("Download failed verification"),
            tr("The downloaded file does not match the checksum GitHub published for it, so it "
               "was discarded. Try again later."));
        return false;
    }

    if (!file.commit()) {
        LOG_WARNING(Frontend, "Could not commit to file {}", filename.toStdString());
        QtCommon::Frontend::Critical(tr("Failed to save file"),
                                     tr("Could not commit to file %1.").arg(filename));
        return false;
    }

    return true;
}
