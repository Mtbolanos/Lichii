// SPDX-License-Identifier: GPL-3.0-or-later

#include "yuzu/fritchii/fritchii_ui.h"

#include <map>
#include <string_view>

#include <QAction>
#include <QActionGroup>
#include <QApplication>
#include <QBuffer>
#include <QDateTime>
#include <QDesktopServices>
#include <QDir>
#include <QFile>
#include <QFont>
#include <QFontDatabase>
#include <QIcon>
#include <QImageReader>
#include <QLocale>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QPalette>
#include <QPixmap>
#include <QProxyStyle>
#include <QPushButton>
#include <QUrl>
#include <QScrollArea>
#include <QStyleFactory>

#include "common/fs/fs_paths.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/scm_rev.h"
#include "common/settings.h"
#include "core/core.h"
#include "qt_common/config/shared_translation.h"
#include "qt_common/qt_common.h"
#include "ui_main.h"
#ifdef HAS_RESHADE
#include "video_core/post_processing/fx_preset.h"
#endif

namespace FritchiiUi {
namespace {

// Graphite
const QColor Background{0x11, 0x11, 0x13};
const QColor Raised{0x1b, 0x1b, 0x1f};
const QColor Field{0x18, 0x18, 0x1b};
const QColor Button{0x1f, 0x1f, 0x24};
const QColor Text{0xec, 0xec, 0xf1};
const QColor TextSecondary{0x9a, 0x9a, 0xa8};
const QColor TextDisabled{0x55, 0x55, 0x5f};
const QColor IconIdle{0x6b, 0x6b, 0x78};
const QColor Accent{0xa7, 0x8b, 0xfa};
const QColor OnAccent{0x1c, 0x14, 0x33};

QPalette GraphitePalette() {
    QPalette p;
    p.setColor(QPalette::Window, Background);
    p.setColor(QPalette::WindowText, Text);
    p.setColor(QPalette::Base, Field);
    p.setColor(QPalette::AlternateBase, Raised);
    p.setColor(QPalette::Text, Text);
    p.setColor(QPalette::Button, Button);
    p.setColor(QPalette::ButtonText, Text);
    p.setColor(QPalette::BrightText, Text);
    p.setColor(QPalette::ToolTipBase, Button);
    p.setColor(QPalette::ToolTipText, Text);
    p.setColor(QPalette::PlaceholderText, IconIdle);
    p.setColor(QPalette::Highlight, Accent);
    p.setColor(QPalette::HighlightedText, OnAccent);
    p.setColor(QPalette::Link, Accent);
    p.setColor(QPalette::LinkVisited, Accent);
    p.setColor(QPalette::Light, QColor(0x2e, 0x2e, 0x35));
    p.setColor(QPalette::Midlight, QColor(0x26, 0x26, 0x2c));
    p.setColor(QPalette::Mid, QColor(0x22, 0x22, 0x27));
    p.setColor(QPalette::Dark, QColor(0x0c, 0x0c, 0x0e));
    p.setColor(QPalette::Shadow, QColor(0x05, 0x05, 0x06));
    for (const auto role : {QPalette::WindowText, QPalette::Text, QPalette::ButtonText}) {
        p.setColor(QPalette::Disabled, role, TextDisabled);
    }
    return p;
}

/// Fusion without the always-on mnemonic underlines.
class GraphiteStyle final : public QProxyStyle {
public:
    GraphiteStyle() : QProxyStyle(QStyleFactory::create(QStringLiteral("Fusion"))) {}
    int styleHint(StyleHint hint, const QStyleOption* option, const QWidget* widget,
                  QStyleHintReturn* data) const override {
        if (hint == SH_UnderlineShortcut) {
            return 0;
        }
        return QProxyStyle::styleHint(hint, option, widget, data);
    }
};

QPixmap RenderSvg(QByteArray svg, const QColor& color, int px) {
    svg.replace("currentColor", color.name().toLatin1());
    QBuffer buffer(&svg);
    buffer.open(QIODevice::ReadOnly);
    QImageReader reader(&buffer, "svg");
    const qreal ratio = qApp->devicePixelRatio();
    reader.setScaledSize(QSize(px, px) * ratio);
    QPixmap pixmap = QPixmap::fromImage(reader.read());
    pixmap.setDevicePixelRatio(ratio);
    return pixmap;
}

void Rename(QAction* action, const QString& text) {
    if (action) {
        action->setText(text);
    }
}

} // namespace

bool Enabled() {
    static const bool enabled = [] {
        if (qEnvironmentVariableIntValue("FRITCHII_CLASSIC_UI") != 0) {
            return false;
        }
        return std::string_view{Common::g_build_name} == "Fritchii";
    }();
    return enabled;
}

bool ApplyTheme() {
    if (!Enabled()) {
        return false;
    }
    static bool once = false;
    if (!once) {
        once = true;
        QFontDatabase::addApplicationFont(QStringLiteral(":/fritchii/fonts/Inter-Regular.ttf"));
        QFontDatabase::addApplicationFont(QStringLiteral(":/fritchii/fonts/Inter-Medium.ttf"));
        QApplication::setStyle(new GraphiteStyle);
    }
    QFont font(QStringLiteral("Inter"));
    font.setPixelSize(14);
    QApplication::setFont(font);
    QApplication::setPalette(GraphitePalette());
    // Folder and refresh icons come from the dark icon set.
    QIcon::setThemeName(QStringLiteral("colorful_dark"));

    QFile qss(QStringLiteral(":/fritchii/style.qss"));
    if (qss.open(QFile::ReadOnly | QFile::Text)) {
        qApp->setStyleSheet(QString::fromUtf8(qss.readAll()));
    }
    return true;
}

bool HidesSetting(const std::string& label) {
    // No frame generation in Fritchii (real frames only), and its options are dev tools.
    return Enabled() && label.starts_with("frame_gen");
}

QIcon Icon(const QString& name, int px) {
    QFile file(QStringLiteral(":/fritchii/icons/%1.svg").arg(name));
    if (!file.open(QFile::ReadOnly)) {
        return {};
    }
    const QByteArray svg = file.readAll();
    QIcon icon;
    const QPixmap idle = RenderSvg(svg, IconIdle, px);
    const QPixmap hover = RenderSvg(svg, Text, px);
    const QPixmap on = RenderSvg(svg, Accent, px);
    icon.addPixmap(idle, QIcon::Normal, QIcon::Off);
    icon.addPixmap(hover, QIcon::Active, QIcon::Off);
    icon.addPixmap(on, QIcon::Normal, QIcon::On);
    icon.addPixmap(on, QIcon::Active, QIcon::On);
    return icon;
}

QIcon Glyph(const QString& name, unsigned rgb, int px) {
    QFile file(QStringLiteral(":/fritchii/icons/%1.svg").arg(name));
    if (!file.open(QFile::ReadOnly)) {
        return {};
    }
    return QIcon(RenderSvg(file.readAll(), QColor::fromRgb(rgb), px));
}

void RestructureMenus(Ui::MainWindow& ui, QMainWindow& window, const MenuHooks& hooks) {
    const auto tr = [](const char* text) { return QObject::tr(text); };

    // File: open, install, folders, settings, exit.
    ui.menuOpen_Eden_Folders->removeAction(ui.action_Log_Folder);
    QMenu* file = ui.menu_File;
    file->clear();
    file->addAction(ui.action_Load_File);
    file->addAction(ui.action_Load_Folder);
    file->addMenu(ui.menu_recent_files);
    file->addSeparator();
    file->addAction(ui.action_Install_File_NAND);
    file->addSeparator();
    file->addMenu(ui.menuOpen_Eden_Folders);
    file->addSeparator();
    file->addAction(ui.action_Configure);
    file->addSeparator();
    file->addAction(ui.action_Exit);

    // Game (was Emulation): pause, stop, restart, shaders, game settings.
    ui.menu_Emulation->setTitle(tr("&Game"));
    ui.menu_Emulation->removeAction(ui.action_Configure);

    // View: window, bars, game list, window size, effects, debugging.
    QMenu* view = ui.menu_View;
    view->clear();
    view->addAction(ui.action_Fullscreen);
    view->addAction(ui.action_Single_Window_Mode);
    view->addSeparator();
    view->addAction(ui.action_Show_Performance_Overlay);
    view->addAction(ui.action_Show_Status_Bar);
    view->addAction(ui.action_Show_Filter_Bar);
    view->addSeparator();
    ui.menu_Game_List_Mode->addSeparator();
    ui.menu_Game_List_Mode->addMenu(ui.menuGame_Icon_Size);
    ui.menu_Game_List_Mode->addAction(ui.action_Show_Game_Name);
    view->addMenu(ui.menu_Game_List_Mode);
    view->addMenu(ui.menu_Reset_Window_Size);
    view->addSeparator();
    view->addAction(ui.action_Post_Processing_Shaders);
    view->addMenu(ui.menu_View_Debugging);

    // Tools: setup tools first; Amiibo; rarely used things under Advanced.
    QMenu* tools = ui.menu_Tools;
    tools->clear();
    tools->addAction(ui.action_Install_Keys);
    tools->addMenu(ui.menuInstall_Firmware);
    tools->addAction(ui.action_Verify_installed_contents);
    tools->addAction(ui.action_Data_Manager);
    tools->addSeparator();
    tools->addAction(ui.action_Capture_Screenshot);
    tools->addMenu(ui.menu_Create_Shortcuts);
    tools->addSeparator();
    QMenu* amiibo = ui.menu_cabinet_applet;
    QAction* first_cabinet = amiibo->actions().value(0);
    amiibo->insertAction(first_cabinet, ui.action_Load_Amiibo);
    amiibo->insertSeparator(first_cabinet);
    tools->addMenu(amiibo);
    QMenu* advanced = new QMenu(tr("Ad&vanced"), tools);
    advanced->addMenu(ui.menu_Applets);
    advanced->addMenu(ui.menuTAS);
    advanced->addMenu(ui.menu_Multiplayer);
    advanced->addSeparator();
    advanced->addAction(ui.action_Enable_Overlay_Applet);
    tools->addMenu(advanced);
    ui.menubar->removeAction(ui.menu_Multiplayer->menuAction());

    // Help: the log first; Eden's mods page leaves the menu.
    auto* export_log = new QAction(tr("&Export Log..."), &window);
    QObject::connect(export_log, &QAction::triggered, &window, hooks.export_log);
    ui.menu_Help->removeAction(ui.action_Open_Mods_Page);
    QAction* first_help = ui.menu_Help->actions().value(0);
    auto* whats_new = new QAction(tr("What's &New"), &window);
    QObject::connect(whats_new, &QAction::triggered, &window, hooks.whats_new);
    auto* delete_cache = new QAction(tr("&Delete Shader Cache..."), &window);
    QObject::connect(delete_cache, &QAction::triggered, &window, hooks.delete_shader_cache);
    ui.menu_Help->insertAction(first_help, export_log);
    ui.menu_Help->insertAction(first_help, ui.action_Log_Folder);
    ui.menu_Help->insertAction(first_help, delete_cache);
    ui.menu_Help->insertSeparator(first_help);
    ui.menu_Help->insertAction(first_help, whats_new);
    auto* discord = new QAction(Icon(QStringLiteral("community"), 16), tr("Fritchii &Discord"), &window);
    QObject::connect(discord, &QAction::triggered, &window, [] { OpenDiscord(); });
    auto* donate = new QAction(Icon(QStringLiteral("heart"), 16), tr("D&onate"), &window);
    QObject::connect(donate, &QAction::triggered, &window, [] { OpenDonate(); });
    ui.menu_Help->insertAction(first_help, discord);
    ui.menu_Help->insertAction(first_help, donate);

    // Plain names.
    ui.menu_recent_files->setTitle(tr("Open &Recent"));
    ui.menuOpen_Eden_Folders->setTitle(tr("Open &Folder"));
    ui.menu_Game_List_Mode->setTitle(tr("Game &List"));
    ui.menuGame_Icon_Size->setTitle(tr("&Icon Size"));
    ui.menu_Reset_Window_Size->setTitle(tr("&Window Size"));
    ui.menu_Create_Shortcuts->setTitle(tr("Create &Shortcut"));
    Rename(ui.action_Load_File, tr("&Open Game..."));
    Rename(ui.action_Load_Folder, tr("Open &Extracted Game..."));
    Rename(ui.action_Install_File_NAND, tr("&Install Update or DLC..."));
    Rename(ui.action_Configure, tr("&Settings..."));
    Rename(ui.action_Configure_Current_Game, tr("Game &Settings..."));
    Rename(ui.action_Root_Data_Folder, tr("&Data"));
    Rename(ui.action_NAND_Folder, tr("&NAND"));
    Rename(ui.action_SDMC_Folder, tr("&SD Card"));
    Rename(ui.action_Mod_Folder, tr("&Mods"));
    Rename(ui.action_Log_Folder, tr("Open &Log Folder"));
    Rename(ui.action_Reset_Window_Size_720, tr("&720p"));
    Rename(ui.action_Reset_Window_Size_900, tr("&900p"));
    Rename(ui.action_Reset_Window_Size_1080, tr("&1080p"));
    Rename(ui.action_Tree_View, tr("&List"));
    Rename(ui.action_Grid_View, tr("&Grid"));
    Rename(ui.action_Carousel_View, tr("&Carousel"));
    Rename(ui.action_Show_Game_Name, tr("Show Game &Names"));
    Rename(ui.action_Show_Performance_Overlay, tr("&Performance Overlay"));
    Rename(ui.action_Show_Status_Bar, tr("&Status Bar"));
    Rename(ui.action_Show_Filter_Bar, tr("Search &Bar"));
    Rename(ui.action_Single_Window_Mode, tr("Single &Window"));
    Rename(ui.action_Enable_Overlay_Applet, tr("&Overlay Display Applet"));
    Rename(ui.action_Data_Manager, tr("&Data Manager..."));
    Rename(ui.action_Capture_Screenshot, tr("Take &Screenshot"));
    Rename(ui.action_Load_Amiibo, tr("&Load or Remove Amiibo..."));
    Rename(ui.action_Firmware_From_Folder, tr("From &Folder..."));
    Rename(ui.action_Firmware_From_ZIP, tr("From &ZIP..."));
    Rename(ui.action_About, tr("&About Fritchii"));
    Rename(ui.action_Eden_Dependencies, tr("&Open-Source Components"));
}

QPushButton* MakeGraphicsButton(QWidget* parent, GraphicsHooks hooks) {
    auto* button = new QPushButton(QObject::tr("GRAPHICS"), parent);
    button->setObjectName(QStringLiteral("TogglableStatusBarButton"));
    button->setFocusPolicy(Qt::NoFocus);

    QObject::connect(button, &QPushButton::clicked, button, [button, hooks] {
        QMenu menu(button);
        const auto add_choices = [&menu, &hooks](const QString& title, const auto& texts,
                                                 auto current, auto apply, bool enabled) {
            QMenu* sub = menu.addMenu(title);
            sub->setEnabled(enabled);
            auto* group = new QActionGroup(sub);
            for (const auto& [value, text] : texts) {
                QAction* action = sub->addAction(text);
                action->setCheckable(true);
                action->setChecked(value == current);
                group->addAction(action);
                QObject::connect(action, &QAction::triggered, sub, [apply, value = value, hooks] {
                    apply(value);
                    hooks.refresh();
                });
            }
        };

        std::map<Settings::RendererBackend, QString> renderers;
        for (const auto& [backend, text] : ConfigurationShared::renderer_backend_texts_map) {
            if (backend != Settings::RendererBackend::Null) {
                renderers.emplace(backend, text);
            }
        }
        add_choices(
            QObject::tr("&Renderer"), renderers, Settings::values.renderer_backend.GetValue(),
            [](Settings::RendererBackend backend) {
                Settings::values.renderer_backend.SetValue(backend);
            },
            !hooks.renderer_locked());
        add_choices(
            QObject::tr("GPU &Accuracy"), ConfigurationShared::gpu_accuracy_texts_map,
            Settings::values.gpu_accuracy.GetValue(),
            [](Settings::GpuAccuracy accuracy) {
                Settings::values.gpu_accuracy.SetValue(accuracy);
                QtCommon::system->ApplySettings();
            },
            true);
        add_choices(
            QObject::tr("Upscaling &Filter"), ConfigurationShared::scaling_filter_texts_map,
            Settings::values.scaling_filter.GetValue(),
            [](Settings::ScalingFilter filter) { Settings::values.scaling_filter.SetValue(filter); },
            true);
        add_choices(
            QObject::tr("Anti-&Aliasing"), ConfigurationShared::anti_aliasing_texts_map,
            Settings::values.anti_aliasing.GetValue(),
            [](Settings::AntiAliasing mode) { Settings::values.anti_aliasing.SetValue(mode); },
            true);
#ifdef HAS_RESHADE
        QMenu* effects = menu.addMenu(QObject::tr("&Effects"));
        auto* effects_group = new QActionGroup(effects);
        const bool fx_on = Settings::values.post_shader_enabled.GetValue();
        const std::string active = VideoCore::GetActiveFxPreset();
        QAction* off = effects->addAction(QObject::tr("Off"));
        off->setCheckable(true);
        off->setChecked(!fx_on);
        effects_group->addAction(off);
        QObject::connect(off, &QAction::triggered, effects, [hooks] {
            Settings::values.post_shader_enabled.SetValue(false);
            hooks.refresh();
        });
        for (const auto& preset : VideoCore::GetFxPresetCatalog()) {
            QAction* action = effects->addAction(QString::fromStdString(preset.name));
            action->setCheckable(true);
            action->setChecked(fx_on && preset.name == active);
            effects_group->addAction(action);
            QObject::connect(action, &QAction::triggered, effects, [name = preset.name, hooks] {
                VideoCore::ApplyFxPreset(name);
                Settings::values.post_shader_enabled.SetValue(true);
                hooks.refresh();
            });
        }
        if (hooks.open_effects) {
            effects->addSeparator();
            effects->addAction(QObject::tr("Configure Effects..."), effects, hooks.open_effects);
        }
#endif
        menu.addSeparator();
        menu.addAction(QObject::tr("More Settings..."), &menu, hooks.open_settings);

        const QSize size = menu.sizeHint();
        menu.exec(button->mapToGlobal(QPoint(0, -size.height())));
    });
    return button;
}

QWidget* SettingsPage(QWidget* page) {
    if (!Enabled() || !page) {
        return page;
    }
    // One scroll area per page, kept on the page (it lives as long as the page's dialog).
    if (auto* area = qobject_cast<QScrollArea*>(page->property("FritchiiScroll").value<QObject*>())) {
        return area;
    }
    auto* area = new QScrollArea;
    area->setFrameShape(QFrame::NoFrame);
    area->setWidgetResizable(true);
    area->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    area->setWidget(page);
    area->setAccessibleName(page->accessibleName());
    page->setProperty("FritchiiScroll", QVariant::fromValue<QObject*>(area));
    return area;
}

const char* LoadingBarStyle() {
    return R"(
QProgressBar {
  background-color: rgba(255, 255, 255, 30);
  border: none;
  border-radius: 3px;
  max-height: 6px;
}
QProgressBar::chunk {
  background-color: #a78bfa;
  border-radius: 3px;
})";
}

namespace {
constexpr int KeptSettingsBackups = 10;
constexpr const char* BackupFolderFormat = "yyyy-MM-dd HHmmss";

QString ConfigDirPath() {
    return QString::fromStdString(Common::FS::PathToUTF8String(
        Common::FS::GetEdenPath(Common::FS::EdenPath::ConfigDir)));
}

bool CopyDirFiles(const QString& from, const QString& to) {
    const QDir source(from);
    if (!source.exists()) {
        return true;
    }
    if (!QDir().mkpath(to)) {
        return false;
    }
    bool ok = true;
    for (const QString& name : source.entryList(QDir::Files)) {
        QFile::remove(QDir(to).filePath(name));
        ok &= QFile::copy(source.filePath(name), QDir(to).filePath(name));
    }
    return ok;
}
} // namespace

QString BackupSettings() {
    const QDir config(ConfigDirPath());
    const QString folder =
        config.filePath(QStringLiteral("backups/") +
                        QDateTime::currentDateTime().toString(QString::fromLatin1(BackupFolderFormat)));
    if (!QDir().mkpath(folder)) {
        LOG_WARNING(Frontend, "Settings backup: cannot create {}", folder.toStdString());
        return {};
    }
    bool ok = true;
    for (const QString name : {QStringLiteral("qt-config.ini"), QStringLiteral("frizti.ini")}) {
        if (config.exists(name)) {
            ok &= QFile::copy(config.filePath(name), QDir(folder).filePath(name));
        }
    }
    ok &= CopyDirFiles(config.filePath(QStringLiteral("custom")),
                       QDir(folder).filePath(QStringLiteral("custom")));
    LOG_INFO(Frontend, "Settings backup {}: {}", folder.toStdString(), ok ? "saved" : "incomplete");
    // Keep the newest few.
    const auto backups = SettingsBackups();
    for (size_t i = KeptSettingsBackups; i < backups.size(); ++i) {
        QDir(backups[i].first).removeRecursively();
    }
    return ok ? folder : QString{};
}

std::vector<std::pair<QString, QString>> SettingsBackups() {
    const QDir root(QDir(ConfigDirPath()).filePath(QStringLiteral("backups")));
    std::vector<std::pair<QString, QString>> backups;
    // Folder names sort by time.
    const QStringList names = root.entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name | QDir::Reversed);
    for (const QString& name : names) {
        const QDateTime when =
            QDateTime::fromString(name, QString::fromLatin1(BackupFolderFormat));
        if (!when.isValid() || !QFile::exists(QDir(root.filePath(name)).filePath(QStringLiteral("qt-config.ini")))) {
            continue;
        }
        backups.emplace_back(root.filePath(name),
                             QLocale().toString(when, QStringLiteral("d MMM yyyy, HH:mm")));
    }
    return backups;
}

bool RestoreSettingsFiles(const QString& backup) {
    const QDir config(ConfigDirPath());
    const QDir from(backup);
    if (!from.exists(QStringLiteral("qt-config.ini"))) {
        return false;
    }
    bool ok = true;
    for (const QString name : {QStringLiteral("qt-config.ini"), QStringLiteral("frizti.ini")}) {
        if (from.exists(name)) {
            QFile::remove(config.filePath(name));
            ok &= QFile::copy(from.filePath(name), config.filePath(name));
        }
    }
    QDir(config.filePath(QStringLiteral("custom"))).removeRecursively();
    ok &= CopyDirFiles(from.filePath(QStringLiteral("custom")),
                       config.filePath(QStringLiteral("custom")));
    LOG_INFO(Frontend, "Settings restored from {}: {}", backup.toStdString(), ok ? "done" : "incomplete");
    return ok;
}

QString& PendingSettingsRestore() {
    static QString pending;
    return pending;
}

void OpenDiscord() {
    QDesktopServices::openUrl(QUrl(QString::fromLatin1(DiscordInvite)));
}

void OpenDonate() {
    QDesktopServices::openUrl(QUrl(QString::fromLatin1(DonateLink)));
}

namespace {
QString OcarinaGameConfigPath() {
    return QDir(ConfigDirPath()).filePath(QStringLiteral("custom/0400486024DEC000.ini"));
}
} // namespace

QStringList OcarinaGameOverrides() {
    QFile file(OcarinaGameConfigPath());
    QStringList names;
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return names;
    }
    const QString suffix = QStringLiteral("\\use_global=false");
    while (!file.atEnd()) {
        const QString line = QString::fromUtf8(file.readLine()).trimmed();
        if (line.endsWith(suffix)) {
            names << line.chopped(suffix.size());
        }
    }
    return names;
}

bool ClearOcarinaGameOverrides() {
    QFile file(OcarinaGameConfigPath());
    if (!file.open(QIODevice::ReadOnly)) {
        return false;
    }
    QByteArray text = file.readAll();
    file.close();
    text.replace("\\use_global=false", "\\use_global=true");
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        return false;
    }
    const bool ok = file.write(text) == text.size();
    LOG_INFO(Frontend, "Ocarina of Time's game-specific settings back on the global ones: {}",
             ok ? "done" : "failed");
    return ok;
}

} // namespace FritchiiUi
