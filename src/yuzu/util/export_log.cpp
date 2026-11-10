// SPDX-License-Identifier: GPL-3.0-or-later

#include "yuzu/util/export_log.h"

#include <thread>
#include <utility>

#include <QApplication>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QGuiApplication>
#include <QMap>
#include <QMessageBox>
#include <QProcess>
#include <QProcessEnvironment>
#include <QRegularExpression>
#include <QScreen>
#include <QSet>
#include <QStandardPaths>
#include <QSysInfo>
#include <QUrl>
#include <quazip.h>
#include <quazipfile.h>
#include <quazipnewinfo.h>
#include <zlib.h>

#include "common/cpu_features.h"
#include "common/frizti_settings.h"
#include "common/fs/fs_paths.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/memory_detect.h"
#include "common/scm_rev.h"
#include "common/settings.h"
#include "qt_common/qt_common.h"
#include "qt_common/util/vk.h"

namespace ExportLog {
namespace {

// A log can reach 100 MiB (1 GiB with extended logging). Past this size the zip keeps the start
// (boot, PC and settings) and the end (where the problem usually is) and cuts the middle.
constexpr qint64 MaxLogBytes = 64LL * 1024 * 1024;
constexpr qint64 HeadBytes = 8LL * 1024 * 1024;

/// Replaces the Windows user name and the PC name wherever they show up as a whole word, so
/// "C:\Users\Name\AppData" becomes "C:\Users\user\AppData" but a word that merely contains the
/// name (a user called "Frizt" and the build name "Frizti") stays as it is.
class Scrubber {
public:
    Scrubber() {
        const auto env = QProcessEnvironment::systemEnvironment();
        Add(env.value(QStringLiteral("USERNAME")), QStringLiteral("user"));
        Add(QDir(QDir::homePath()).dirName(), QStringLiteral("user"));
        Add(env.value(QStringLiteral("COMPUTERNAME")), QStringLiteral("PC"));
        Add(QSysInfo::machineHostName(), QStringLiteral("PC"));
    }

    QString Apply(QString text) const {
        for (const auto& [pattern, replacement] : rules) {
            text.replace(pattern, replacement);
        }
        return text;
    }

private:
    void Add(const QString& word, const QString& replacement) {
        // A one-letter name would garble the whole log.
        if (word.size() < 2) {
            return;
        }
        for (const auto& [pattern, existing] : rules) {
            if (pattern.match(word).hasMatch()) {
                return;
            }
        }
        rules.emplace_back(QRegularExpression(QStringLiteral("(?<![A-Za-z0-9])%1(?![A-Za-z0-9])")
                                                  .arg(QRegularExpression::escape(word)),
                                              QRegularExpression::CaseInsensitiveOption),
                           replacement);
    }

    std::vector<std::pair<QRegularExpression, QString>> rules;
};

QString ReadLog(const QString& path, qint64* size_out) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        return {};
    }
    const qint64 size = file.size();
    *size_out = size;
    QByteArray bytes;
    if (size <= MaxLogBytes) {
        bytes = file.readAll();
    } else {
        bytes = file.read(HeadBytes);
        bytes += QStringLiteral("\n\n[... %1 MB cut from the middle of this log to keep the zip "
                                "small ...]\n\n")
                     .arg((size - MaxLogBytes) / (1024 * 1024))
                     .toUtf8();
        file.seek(size - (MaxLogBytes - HeadBytes));
        bytes += file.readAll();
    }
    return QString::fromUtf8(bytes);
}

QString Line(const QString& name, const QString& value) {
    return QStringLiteral("%1: %2\n").arg(name, value);
}

QString Gigabytes(u64 bytes) {
    return QStringLiteral("%1 GB").arg(static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0), 0,
                                      'f', 1);
}

QString SystemSummary(const std::vector<VkDeviceInfo::Record>& vk_device_records,
                      const QStringList& included) {
    QString text;
    text += QStringLiteral("Fritchii log export, made %1\n\n")
                .arg(QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd HH:mm")));

    text += QStringLiteral("== Fritchii ==\n");
    text += Line(QStringLiteral("Version"), QString::fromStdString(QtCommon::BuildVersionString()));
    text += Line(QStringLiteral("Build"), QStringLiteral("%1, %2, %3")
                                              .arg(QString::fromUtf8(Common::g_build_fullname),
                                                   QString::fromUtf8(Common::g_build_date),
                                                   QString::fromUtf8(Common::g_compiler_id)));
    text += Line(QStringLiteral("Tuned settings revision"),
                 QStringLiteral("%1 (applied: %2)")
                     .arg(Settings::Frizti::Revision)
                     .arg(Settings::values.frizti_settings_revision.GetValue()));
    // A portable copy keeps its data in a "user" folder next to the exe.
    auto data_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::EdenDir);
    if (!data_dir.has_filename()) {
        data_dir = data_dir.parent_path();
    }
    text += Line(QStringLiteral("Portable"),
                 data_dir.filename() == "user" ? QStringLiteral("yes") : QStringLiteral("no"));

    text += QStringLiteral("\n== PC ==\n");
    text += Line(QStringLiteral("OS"), QtCommon::PrettyProductName());
#ifdef ARCHITECTURE_x86_64
    const auto& caps = Common::g_cpu_caps;
    QString cpu = QString::fromUtf8(caps.cpu_string).trimmed();
    cpu += caps.avx512f ? QStringLiteral(" | AVX-512")
           : caps.avx2  ? QStringLiteral(" | AVX2")
           : caps.avx   ? QStringLiteral(" | AVX")
                        : QString{};
    text += Line(QStringLiteral("CPU"), cpu);
#endif
    const auto cores = Common::GetProcessorCount();
    text += Line(QStringLiteral("CPU cores / threads"),
                 QStringLiteral("%1 / %2")
                     .arg(cores ? QString::number(*cores) : QStringLiteral("?"))
                     .arg(std::thread::hardware_concurrency()));
    const auto& memory = Common::GetMemInfo();
    text += Line(QStringLiteral("RAM"), QStringLiteral("%1 (page file %2)")
                                            .arg(Gigabytes(memory.TotalPhysicalMemory),
                                                 Gigabytes(memory.TotalSwapMemory)));
    const u32 selected_gpu = Settings::values.vulkan_device.GetValue();
    for (size_t i = 0; i < vk_device_records.size(); ++i) {
        const auto& record = vk_device_records[i];
        QString gpu = QString::fromStdString(record.name);
        if (!record.driver_version.empty()) {
            gpu += QStringLiteral(", driver %1").arg(QString::fromStdString(record.driver_version));
        }
        text += Line(i == selected_gpu ? QStringLiteral("GPU (used)") : QStringLiteral("GPU"), gpu);
    }
    if (vk_device_records.empty()) {
        text += Line(QStringLiteral("GPU"), QStringLiteral("no Vulkan device found"));
    }
    const auto screens = QGuiApplication::screens();
    for (const QScreen* screen : screens) {
        const QSize pixels = screen->size() * screen->devicePixelRatio();
        text += Line(screen == QGuiApplication::primaryScreen() ? QStringLiteral("Screen (main)")
                                                                : QStringLiteral("Screen"),
                     QStringLiteral("%1x%2 at %3 Hz, scale %4%")
                         .arg(pixels.width())
                         .arg(pixels.height())
                         .arg(screen->refreshRate(), 0, 'f', 0)
                         .arg(qRound(screen->devicePixelRatio() * 100)));
    }

    text += QStringLiteral("\n== Settings that matter for support ==\n");
    auto& values = Settings::values;
    std::vector<Settings::BasicSetting*> settings{
        &values.renderer_backend, &values.vulkan_device,   &values.oot_30fps,
        &values.use_speed_limit,  &values.speed_limit,     &values.accelerate_astc,
        &values.astc_recompression, &values.oot_high_refresh, &values.oot_high_refresh_cap,
    };
    for (auto* setting : Settings::Frizti::TunedSettings(values)) {
        settings.push_back(setting);
    }
    for (const auto* setting : settings) {
        text += Line(QString::fromStdString(setting->GetLabel()),
                     QString::fromStdString(setting->Canonicalize()));
    }

    // Ocarina's game-specific settings (right-click > Configure Game, a value with a clear button):
    // they beat the global ones, the usual reason "my settings didn't apply" (2026-11-09).
    text += QStringLiteral("\n== OoT game-specific settings (these beat the global ones) ==\n");
    {
        const auto game_ini = Common::FS::GetEdenPath(Common::FS::EdenPath::ConfigDir) / "custom" /
                              "0400486024DEC000.ini";
        QFile file(QString::fromStdString(Common::FS::PathToUTF8String(game_ini)));
        QStringList overridden;
        if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            QString section;
            QSet<QString> own;
            QMap<QString, QString> values_by_key;
            while (!file.atEnd()) {
                const QString line = QString::fromUtf8(file.readLine()).trimmed();
                if (line.startsWith(QLatin1Char('[')) && line.endsWith(QLatin1Char(']'))) {
                    section = line.mid(1, line.size() - 2);
                    continue;
                }
                const qsizetype eq = line.indexOf(QLatin1Char('='));
                if (eq <= 0) {
                    continue;
                }
                const QString key = line.left(eq);
                const QString value = line.mid(eq + 1);
                if (key.endsWith(QStringLiteral("\\use_global"))) {
                    if (value == QStringLiteral("false")) {
                        own.insert(section + QLatin1Char('/') + key.chopped(11));
                    }
                } else if (!key.contains(QLatin1Char('\\'))) {
                    values_by_key.insert(section + QLatin1Char('/') + key, value);
                }
            }
            for (const QString& key : own) {
                overridden << QStringLiteral("%1: %2").arg(key, values_by_key.value(key));
            }
            overridden.sort();
        }
        text += overridden.isEmpty() ? QStringLiteral("none") : overridden.join(QLatin1Char('\n'));
        text += QLatin1Char('\n');
    }

    QStringList switches;
    const auto env = QProcessEnvironment::systemEnvironment();
    for (const QString& key : env.keys()) {
        if (key.startsWith(QStringLiteral("EDEN_")) || key.startsWith(QStringLiteral("LICHII_"))) {
            switches << QStringLiteral("%1=%2").arg(key, env.value(key));
        }
    }
    switches.sort();
    text += QStringLiteral("\n== Switches (environment) ==\n");
    text += switches.isEmpty() ? QStringLiteral("none") : switches.join(QLatin1Char('\n'));
    text += QLatin1Char('\n');

    text += QStringLiteral("\n== In this zip ==\n");
    text += included.join(QLatin1Char('\n'));
    text += QLatin1Char('\n');
    text += QStringLiteral("\nThe Windows user name and PC name were replaced with \"user\" and "
                           "\"PC\" everywhere in this zip.\n");
    return text;
}

bool AddFile(QuaZip& zip, const QString& name, const QByteArray& data) {
    QuaZipFile out(&zip);
    if (!out.open(QIODevice::WriteOnly, QuaZipNewInfo(name), nullptr, 0, Z_DEFLATED,
                  Z_DEFAULT_COMPRESSION)) {
        return false;
    }
    const bool written = out.write(data) == data.size();
    out.close();
    return written && out.getZipError() == ZIP_OK;
}

void ShowInExplorer(const QString& path) {
#ifdef _WIN32
    if (QProcess::startDetached(QStringLiteral("explorer.exe"),
                                {QStringLiteral("/select,"), QDir::toNativeSeparators(path)})) {
        return;
    }
#endif
    QDesktopServices::openUrl(QUrl::fromLocalFile(QFileInfo(path).absolutePath()));
}

} // namespace

void Export(QWidget* parent, const std::vector<VkDeviceInfo::Record>& vk_device_records) {
    const QString desktop = QStandardPaths::writableLocation(QStandardPaths::DesktopLocation);
    const QString default_name =
        QStringLiteral("Fritchii-log-%1.zip")
            .arg(QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd-HHmm")));
    QString path = QFileDialog::getSaveFileName(
        parent, QObject::tr("Export Log"), QDir(desktop).filePath(default_name),
        QObject::tr("Zip file (*.zip)"));
    if (path.isEmpty()) {
        return;
    }
    if (!path.endsWith(QStringLiteral(".zip"), Qt::CaseInsensitive)) {
        path += QStringLiteral(".zip");
    }

    QApplication::setOverrideCursor(Qt::WaitCursor);

    // Logging writes through a buffer: flush it so the zip has everything up to now.
    Common::Log::Stop();

    const QString log_dir = QString::fromStdString(
        Common::FS::PathToUTF8String(Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir)));
    const QString usual_path = QDir(log_dir).filePath(QStringLiteral(LOG_FILE));
    const QString previous_path = usual_path + QStringLiteral(".old.txt");
    // The file this session really writes: another Fritchii that never finished closing keeps
    // the usual one locked, and that one then holds the stuck process's old log.
    bool usual_locked = false;
    const auto session_file = Common::Log::FilePath(&usual_locked);
    const QString current_path =
        session_file.empty()
            ? usual_path
            : QString::fromStdString(Common::FS::PathToUTF8String(session_file));

    const Scrubber scrubber;
    QStringList included;
    std::vector<std::pair<QString, QByteArray>> files;
    const auto add_log = [&](const QString& source, const QString& name, const QString& what) {
        qint64 size = 0;
        const QString text = ReadLog(source, &size);
        if (size == 0) {
            return;
        }
        files.emplace_back(name, scrubber.Apply(text).toUtf8());
        included << QStringLiteral("%1: %2 (%3 KB)").arg(name, what).arg((size + 1023) / 1024);
    };
    add_log(current_path, QStringLiteral("log-this-session.txt"),
            QObject::tr("the log of this session"));
    if (usual_locked) {
        add_log(usual_path, QStringLiteral("log-stuck-process.txt"),
                QObject::tr("the log of a Fritchii that was still running when this one started"));
        included << QStringLiteral(
            "WARNING: another lichii.exe was still running when this session started. End it in "
            "Task Manager (or restart the PC), then start Fritchii again.");
    } else {
        add_log(previous_path, QStringLiteral("log-previous-session.txt"),
                QObject::tr("the log of the session before"));
    }
    included.prepend(QStringLiteral("system-info.txt: this summary"));
    files.emplace(files.begin(), QStringLiteral("system-info.txt"),
                  scrubber.Apply(SystemSummary(vk_device_records, included)).toUtf8());

    bool ok = false;
    {
        QuaZip zip(path);
        if (zip.open(QuaZip::mdCreate)) {
            ok = true;
            for (const auto& [name, data] : files) {
                ok = ok && AddFile(zip, name, data);
            }
            zip.close();
            ok = ok && zip.getZipError() == ZIP_OK;
        }
    }

    QApplication::restoreOverrideCursor();

    if (!ok) {
        QFile::remove(path);
        QMessageBox::warning(parent, QObject::tr("Export Log"),
                             QObject::tr("Couldn't save the log to %1.\nTry another folder.")
                                 .arg(QDir::toNativeSeparators(path)));
        return;
    }

    ShowInExplorer(path);

    QMessageBox box(parent);
    box.setIcon(QMessageBox::Information);
    box.setWindowTitle(QObject::tr("Log Exported"));
    box.setText(QObject::tr("<b>Saved %1</b>").arg(QFileInfo(path).fileName().toHtmlEscaped()));
    box.setInformativeText(QObject::tr(
        "Drag this file into your post in the Fritchii Discord issues, or send it to Frizti."
        "<br><br>What's inside:"
        "<ul style='margin-left:-20px'>"
        "<li>Your log from this session and the one before</li>"
        "<li>Your Fritchii version and the settings that matter for support</li>"
        "<li>A short summary of your PC: Windows version, CPU, graphics card and driver, RAM "
        "and screen</li>"
        "</ul>"
        "Your Windows user name and PC name are removed. Nothing else from your PC is in it."));
    box.setStandardButtons(QMessageBox::Ok);
    box.exec();
}

} // namespace ExportLog
