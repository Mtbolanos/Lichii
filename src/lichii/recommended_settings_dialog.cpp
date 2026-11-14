// SPDX-License-Identifier: GPL-3.0-or-later

#include <string>
#include <QMessageBox>
#include <QPushButton>
#include <QStringList>
#include "qt_common/config/shared_translation.h"
#include "lichii/recommended_settings_dialog.h"

namespace {

// One "name: previous value → new value" line per change, worded as the settings dialog words
// them.
QStringList Describe(QWidget* parent,
                     const std::vector<Settings::RecommendedSettingChange>& changes,
                     const QString& on, const QString& off) {
    const auto titles = ConfigurationShared::InitializeTranslations(parent);
    const auto enums = ConfigurationShared::ComboboxEnumeration(parent);

    const auto title_of = [&](const Settings::BasicSetting& setting) {
        const auto it = titles->find(setting.Id());
        return it == titles->end() ? QString{} : it->second.first;
    };
    const auto text_of = [&](const Settings::BasicSetting& setting, const std::string& value) {
        if (setting.TypeId() == "bool") {
            return value == "true" ? on : off;
        }
        if (const auto it = enums->find(setting.EnumIndex()); it != enums->end()) {
            for (const auto& [id, text] : it->second) {
                if (std::to_string(id) == value) {
                    return text;
                }
            }
        }
        QString text = QString::fromStdString(value);
        if (setting.Specialization() & Settings::Specialization::Percentage) {
            text.append(QLatin1Char{'%'});
        }
        return text;
    };

    QStringList lines;
    for (const auto& change : changes) {
        QString title = title_of(*change.setting);
        if (title.isEmpty()) {
            // A checkbox paired with a value is named by the value's setting.
            for (const auto& [key, other] : Settings::values.linkage.by_key) {
                if (other->PairedSetting() == change.setting) {
                    title = title_of(*other);
                }
            }
        }
        if (title.isEmpty()) {
            title = QString::fromStdString(change.setting->GetLabel());
        }
        if (title.endsWith(QLatin1Char{':'})) {
            title.chop(1);
        }
        if (!change.game.empty()) {
            title = RecommendedSettingsDialog::tr("%1 (game %2)")
                        .arg(title, QString::fromStdString(change.game));
        }
        lines.append(QStringLiteral("%1: %2 \u2192 %3")
                         .arg(title, text_of(*change.setting, change.previous),
                              text_of(*change.setting, change.current)));
    }
    return lines;
}

void ShowChanges(QWidget* parent, const QString& title, const QString& text,
                 const QStringList& lines, const QString& footer) {
    QMessageBox box(QMessageBox::Information, title, text, QMessageBox::Ok, parent);
    // A long list goes behind the details button so the box stays on the screen.
    if (lines.size() <= 12) {
        // Rich text, so that a line is not wrapped in the middle of a value.
        QString list;
        for (const QString& line : lines) {
            list += QStringLiteral("<nobr>%1</nobr><br>").arg(line.toHtmlEscaped());
        }
        box.setInformativeText(list + QStringLiteral("<br>") + footer.toHtmlEscaped());
    } else {
        box.setInformativeText(footer);
        box.setDetailedText(lines.join(QLatin1Char{'\n'}));
    }
    box.exec();
}

} // namespace

bool RecommendedSettingsDialog::Ask(QWidget* parent) {
    QMessageBox box(parent);
    box.setIcon(QMessageBox::Question);
    box.setWindowTitle(tr("Recommended settings"));
    box.setText(tr("Do you want to use the recommended settings from the Lichii developer?"));
    box.setInformativeText(
        tr("They improve performance for most games. Games with their own value for one of these "
           "settings in their per-game settings use the recommended value too; other per-game "
           "settings do not change. Lichii keeps your own settings, global and per-game, so you "
           "can go back at any time with Help > Restore My Previous Settings."));
    auto* const use = box.addButton(tr("Use recommended settings"), QMessageBox::AcceptRole);
    box.addButton(tr("Not now"), QMessageBox::RejectRole);
    box.setDefaultButton(use);
    box.exec();
    return box.clickedButton() == use;
}

void RecommendedSettingsDialog::ShowApplied(
    QWidget* parent, const std::vector<Settings::RecommendedSettingChange>& changes) {
    if (changes.empty()) {
        QMessageBox::information(
            parent, tr("Recommended settings"),
            tr("Your settings already match the recommended settings. Nothing changed."));
        return;
    }
    ShowChanges(parent, tr("Recommended settings"),
                tr("Lichii now uses the recommended settings, in every game."),
                Describe(parent, changes, tr("On"), tr("Off")),
                tr("To go back, use Help > Restore My Previous Settings."));
}

void RecommendedSettingsDialog::ShowRestored(
    QWidget* parent, const std::vector<Settings::RecommendedSettingChange>& changes) {
    if (changes.empty()) {
        QMessageBox::information(
            parent, tr("Previous settings"),
            tr("Your settings already match your previous settings. Nothing changed."));
        return;
    }
    ShowChanges(parent, tr("Previous settings"), tr("Lichii uses your previous settings again."),
                Describe(parent, changes, tr("On"), tr("Off")),
                tr("To use the recommended settings again, use Help > Apply Recommended "
                   "Settings."));
}
