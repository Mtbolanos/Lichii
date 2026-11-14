// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2016 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <memory>
#include <tuple>
#include <QPushButton>
#include <QScreen>
#include <QScrollArea>
#include <QScrollBar>
#include <QStackedWidget>
#include <QStyle>
#include <QTabBar>
#include <QTimer>
#include "common/logging.h"
#include "common/settings.h"
#include "common/settings_enums.h"
#include "core/core.h"
#include "qt_common/config/uisettings.h"
#include "qt_common/util/vk.h"
#include "ui_configure.h"
#include "lichii/configuration/configuration_shared.h"
#include "lichii/configuration/configure_applets.h"
#include "lichii/configuration/configure_audio.h"
#include "lichii/configuration/configure_cpu.h"
#include "lichii/configuration/configure_debug_tab.h"
#include "lichii/configuration/configure_dialog.h"
#include "lichii/configuration/configure_filesystem.h"
#include "lichii/configuration/configure_general.h"
#include "lichii/configuration/configure_graphics.h"
#include "lichii/configuration/configure_graphics_advanced.h"
#include "lichii/configuration/configure_graphics_extensions.h"
#include "lichii/configuration/configure_hotkeys.h"
#include "lichii/configuration/configure_input.h"
#include "lichii/configuration/configure_input_player.h"
#include "lichii/configuration/configure_network.h"
#include "lichii/configuration/configure_profile_manager.h"
#include "lichii/configuration/configure_system.h"
#include "lichii/configuration/configure_ui.h"
#include "lichii/configuration/configure_web.h"
#include "lichii/hotkeys.h"

ConfigureDialog::ConfigureDialog(QWidget* parent, HotkeyRegistry& registry_,
                                 InputCommon::InputSubsystem* input_subsystem,
                                 std::vector<VkDeviceInfo::Record>& vk_device_records,
                                 Core::System& system_, bool enable_web_config)
    : QDialog(parent), ui{std::make_unique<Ui::ConfigureDialog>()}, registry(registry_),
      system{system_},
      builder{std::make_unique<ConfigurationShared::Builder>(this, !system_.IsPoweredOn())},
      applets_tab{std::make_unique<ConfigureApplets>(system_, nullptr, *builder, this)},
      audio_tab{std::make_unique<ConfigureAudio>(system_, nullptr, *builder, this)},
      cpu_tab{std::make_unique<ConfigureCpu>(system_, nullptr, *builder, this)},
      debug_tab_tab{std::make_unique<ConfigureDebugTab>(system_, this)},
      filesystem_tab{std::make_unique<ConfigureFilesystem>(this)},
      general_tab{std::make_unique<ConfigureGeneral>(system_, nullptr, *builder, this)},
      graphics_advanced_tab{
          std::make_unique<ConfigureGraphicsAdvanced>(system_, nullptr, *builder, this)},
      graphics_extensions_tab{
          std::make_unique<ConfigureGraphicsExtensions>(system_, nullptr, *builder, this)},
      ui_tab{std::make_unique<ConfigureUi>(system_, this)},
      graphics_tab{std::make_unique<ConfigureGraphics>(
          system_, vk_device_records, [&]() { graphics_advanced_tab->ExposeComputeOption(); },
          [this](Settings::AspectRatio ratio, Settings::ResolutionSetup setup) {
              ui_tab->UpdateScreenshotInfo(ratio, setup);
          },
          nullptr, *builder, this)},
      hotkeys_tab{std::make_unique<ConfigureHotkeys>(system_.HIDCore(), this)},
      input_tab{std::make_unique<ConfigureInput>(system_, this)},
      network_tab{std::make_unique<ConfigureNetwork>(system_, this)},
      profile_tab{std::make_unique<ConfigureProfileManager>(system_, this)},
      system_tab{std::make_unique<ConfigureSystem>(system_, nullptr, *builder, this)},
      web_tab{std::make_unique<ConfigureWeb>(this)} {
    Settings::SetConfiguringGlobal(true);

    ui->setupUi(this);

    ui->tabWidget->addTab(applets_tab.get(), tr("Applets"));
    ui->tabWidget->addTab(audio_tab.get(), tr("Audio"));
    ui->tabWidget->addTab(cpu_tab.get(), tr("CPU"));
    ui->tabWidget->addTab(debug_tab_tab.get(), tr("Debug"));
    ui->tabWidget->addTab(filesystem_tab.get(), tr("Filesystem"));
    ui->tabWidget->addTab(general_tab.get(), tr("General"));
    ui->tabWidget->addTab(graphics_tab.get(), tr("Graphics"));
    ui->tabWidget->addTab(graphics_advanced_tab.get(), tr("GraphicsAdvanced"));
    ui->tabWidget->addTab(graphics_extensions_tab.get(), tr("GraphicsExtra"));
    ui->tabWidget->addTab(hotkeys_tab.get(), tr("Hotkeys"));
    ui->tabWidget->addTab(input_tab.get(), tr("Controls"));
    ui->tabWidget->addTab(profile_tab.get(), tr("Profiles"));
    ui->tabWidget->addTab(network_tab.get(), tr("Network"));
    ui->tabWidget->addTab(system_tab.get(), tr("System"));
    ui->tabWidget->addTab(ui_tab.get(), tr("Game List"));
    ui->tabWidget->addTab(web_tab.get(), tr("Web"));

    web_tab->SetWebServiceConfigEnabled(enable_web_config);
    hotkeys_tab->Populate(registry);

    input_tab->Initialize(input_subsystem);

    general_tab->SetResetCallback([&] { this->close(); });

    SetConfiguration();

    const std::array<std::pair<QWidget*, const char*>, 15> tab_icons{{
        {general_tab.get(), "sliders"},
        {hotkeys_tab.get(), "keyboard"},
        {ui_tab.get(), "list"},
        {web_tab.get(), "cloud"},
        {debug_tab_tab.get(), "bug"},
        {system_tab.get(), "system"},
        {profile_tab.get(), "user"},
        {network_tab.get(), "globe"},
        {filesystem_tab.get(), "drive"},
        {applets_tab.get(), "apps"},
        {cpu_tab.get(), "cpu"},
        {graphics_tab.get(), "monitor"},
        {graphics_advanced_tab.get(), "layers"},
        {graphics_extensions_tab.get(), "wand"},
        {audio_tab.get(), "audio"},
    }};
    for (const auto& [tab, icon] : tab_icons) {
        tab->setProperty("tabIcon", QString::fromLatin1(icon));
        ConfigurationShared::UseToggles(tab);
    }
    for (auto* const tab : input_tab->GetSubTabs()) {
        tab->setProperty("tabIcon", QStringLiteral("gamepad"));
    }

    PopulateSelectionList();

    connect(ui->tabWidget, &QTabWidget::currentChanged, this, [this](int index) {
        if (index != -1) {
            debug_tab_tab->SetCurrentIndex(0);
        }
    });
    connect(ui->tabWidget, &QTabWidget::currentChanged, this,
            [this] { QTimer::singleShot(0, this, &ConfigureDialog::FitPageWidth); });
    connect(ui_tab.get(), &ConfigureUi::LanguageChanged, this, &ConfigureDialog::OnLanguageChanged);
    connect(general_tab.get(), &ConfigureGeneral::ExternalContentDirsChanged, this,
            &ConfigureDialog::ExternalContentDirsChanged);
    connect(ui->selectorList, &QListWidget::itemSelectionChanged, this,
            &ConfigureDialog::UpdateVisibleTabs);

    if (system.IsPoweredOn()) {
        QPushButton* apply_button = ui->buttonBox->addButton(QDialogButtonBox::Apply);
        connect(apply_button, &QAbstractButton::clicked, this,
                &ConfigureDialog::HandleApplyButtonClicked);
    }

    ui->selectorList->setFixedWidth(std::max(184, ui->selectorList->sizeHintForColumn(0) +
                                                      2 * ui->selectorList->frameWidth() + 32));
    ConfigurationShared::FitToScreen(this, QSize(1040, 760));
    ui->selectorList->setCurrentRow(0);

    // Selects the leftmost button on the bottom bar (Cancel as of writing)
    ui->buttonBox->setFocus();
}

ConfigureDialog::~ConfigureDialog() = default;

void ConfigureDialog::SetConfiguration() {}

void ConfigureDialog::ApplyConfiguration() {
    general_tab->ApplyConfiguration();
    ui_tab->ApplyConfiguration();
    system_tab->ApplyConfiguration();
    profile_tab->ApplyConfiguration();
    filesystem_tab->ApplyConfiguration();
    input_tab->ApplyConfiguration();
    hotkeys_tab->ApplyConfiguration(registry);
    cpu_tab->ApplyConfiguration();
    graphics_tab->ApplyConfiguration();
    graphics_advanced_tab->ApplyConfiguration();
    graphics_extensions_tab->ApplyConfiguration();
    audio_tab->ApplyConfiguration();
    debug_tab_tab->ApplyConfiguration();
    web_tab->ApplyConfiguration();
    network_tab->ApplyConfiguration();
    applets_tab->ApplyConfiguration();
    system.ApplySettings();
    Settings::LogSettings();
}

void ConfigureDialog::changeEvent(QEvent* event) {
    if (event->type() == QEvent::LanguageChange) {
        RetranslateUI();
    }

    QDialog::changeEvent(event);
}

void ConfigureDialog::showEvent(QShowEvent* event) {
    QDialog::showEvent(event);
    QTimer::singleShot(0, this, &ConfigureDialog::FitPageWidth);
}

void ConfigureDialog::FitPageWidth() {
    if (!isVisible()) {
        return;
    }
    auto* const area = qobject_cast<QScrollArea*>(ui->tabWidget->currentWidget());
    if (area == nullptr || area->widget() == nullptr || area->parentWidget() == nullptr) {
        return;
    }
    QWidget* const page = area->widget();
    // The stacked holder has the same size for every tab, also for one not laid out yet.
    const QWidget* const holder = area->parentWidget();

    const auto missing_width = [&] {
        const QSize page_min = page->minimumSizeHint().expandedTo(page->minimumSize());
        int view_width = holder->width() - 2 * area->frameWidth();
        if (page_min.height() > holder->height()) {
            view_width -= area->verticalScrollBar()->sizeHint().width();
        }
        const int page_missing = page_min.width() - view_width;
        const int tabs_missing = ui->tabWidget->tabBar()->sizeHint().width() - holder->width();
        return std::max(page_missing, tabs_missing);
    };

    const int missing = missing_width();
    if (missing <= 0) {
        return;
    }

    const QScreen* const current_screen = screen();
    if (current_screen == nullptr) {
        return;
    }
    const QRect available = current_screen->availableGeometry();
    const int frame_extra = frameGeometry().width() - width();
    const int max_width = available.width() * 95 / 100 - frame_extra;
    const int wanted_width = width() + missing;
    const int new_width = std::min(wanted_width, max_width);

    if (new_width > width()) {
        // Grow around the current center, then keep the whole frame on the screen.
        const int grow = new_width - width();
        resize(new_width, height());
        QRect frame = frameGeometry();
        frame.moveLeft(frame.left() - grow / 2);
        if (frame.right() > available.right()) {
            frame.moveRight(available.right());
        }
        if (frame.left() < available.left()) {
            frame.moveLeft(available.left());
        }
        move(frame.topLeft());
    }

    if (wanted_width > new_width && !page->property("compactPage").toBool()) {
        // Even the screen is too narrow: use tighter buttons so less of the page scrolls.
        page->setProperty("compactPage", true);
        for (auto* const button : page->findChildren<QPushButton*>()) {
            button->style()->unpolish(button);
            button->style()->polish(button);
        }
        page->updateGeometry();
    }
}

void ConfigureDialog::RetranslateUI() {
    const int old_row = ui->selectorList->currentRow();
    const int old_index = ui->tabWidget->currentIndex();

    ui->retranslateUi(this);

    PopulateSelectionList();
    ui->selectorList->setCurrentRow(old_row);

    UpdateVisibleTabs();
    ui->tabWidget->setCurrentIndex(old_index);
}

void ConfigureDialog::HandleApplyButtonClicked() {
    UISettings::values.configuration_applied = true;
    ApplyConfiguration();
}

Q_DECLARE_METATYPE(QList<QWidget*>);

void ConfigureDialog::PopulateSelectionList() {
    const std::array<std::tuple<QString, const char*, QList<QWidget*>>, 6> items{
        {{tr("General"),
          "sliders",
          {general_tab.get(), hotkeys_tab.get(), ui_tab.get(), web_tab.get(), debug_tab_tab.get()}},
         {tr("System"),
          "system",
          {system_tab.get(), profile_tab.get(), network_tab.get(), filesystem_tab.get(),
           applets_tab.get()}},
         {tr("CPU"), "cpu", {cpu_tab.get()}},
         {tr("Graphics"),
          "monitor",
          {graphics_tab.get(), graphics_advanced_tab.get(), graphics_extensions_tab.get()}},
         {tr("Audio"), "audio", {audio_tab.get()}},
         {tr("Controls"), "gamepad", input_tab->GetSubTabs()}},
    };

    [[maybe_unused]] const QSignalBlocker blocker(ui->selectorList);

    ui->selectorList->clear();
    for (const auto& [name, icon, tabs] : items) {
        auto* const item =
            new QListWidgetItem(ConfigurationShared::PageIcon(QString::fromLatin1(icon)), name);
        item->setData(Qt::UserRole, QVariant::fromValue(tabs));

        ui->selectorList->addItem(item);
    }
}

void ConfigureDialog::OnLanguageChanged(const QString& locale) {
    emit LanguageChanged(locale);
    //  Reloading the game list is needed to force retranslation.
    UISettings::values.is_game_list_reload_pending = true;
    // first apply the configuration, and then restore the display
    ApplyConfiguration();
    RetranslateUI();
    SetConfiguration();
}

void ConfigureDialog::UpdateVisibleTabs() {
    const auto items = ui->selectorList->selectedItems();
    if (items.isEmpty()) {
        return;
    }

    [[maybe_unused]] const QSignalBlocker blocker(ui->tabWidget);

    ui->tabWidget->clear();

    const auto tabs = qvariant_cast<QList<QWidget*>>(items[0]->data(Qt::UserRole));

    for (auto* const tab : tabs) {
        LOG_DEBUG(Frontend, "{}", tab->accessibleName().toStdString());
        auto& area = scroll_areas[tab];
        if (area == nullptr) {
            area = ConfigurationShared::ScrollTab(tab);
        }
        ui->tabWidget->addTab(area,
                              ConfigurationShared::PageIcon(tab->property("tabIcon").toString()),
                              tab->accessibleName());
    }

    QTimer::singleShot(0, this, &ConfigureDialog::FitPageWidth);
}
