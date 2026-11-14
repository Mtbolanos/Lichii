// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <QCoreApplication>
#include <QDesktopServices>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QPointer>
#include <QUrl>
#include <QtConcurrent/QtConcurrent>
#include "common/nextendo.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/internal_network/network_interface.h"
#include "ui_configure_network.h"
#include "lichii/configuration/configure_network.h"

namespace {

std::string NextendoClientId(const QString& text) {
    const std::string id = text.trimmed().toStdString();
    return id.empty() ? Settings::values.nextendo_client_id.GetDefault() : id;
}

QString OnlineRefusal(const std::string& reason) {
    if (reason == "unverified") {
        return ConfigureNetwork::tr("Online play needs a verified e-mail address. Verify it on "
                                    "nextendo.network.");
    }
    if (reason == "discord_unlinked") {
        return ConfigureNetwork::tr("Online play needs a Discord account. Link one on "
                                    "nextendo.network.");
    }
    if (reason == "disabled") {
        return ConfigureNetwork::tr("nextendo.network has suspended this account.");
    }
    if (reason == "elsewhere") {
        return ConfigureNetwork::tr("This account is already playing online on another device.");
    }
    if (reason == "unknown") {
        return ConfigureNetwork::tr("nextendo.network doesn't recognise this account.");
    }
    return ConfigureNetwork::tr("nextendo.network refuses online play for this account (%1).")
        .arg(QString::fromStdString(reason));
}

} // namespace

ConfigureNetwork::ConfigureNetwork(const Core::System& system_, QWidget* parent, bool per_game_)
    : QWidget(parent), ui(std::make_unique<Ui::ConfigureNetwork>()), system{system_},
      per_game{per_game_} {
    ui->setupUi(this);
    for (const auto& iface : Network::GetAvailableNetworkInterfaces())
        ui->network_interface->addItem(QString::fromStdString(iface.name));
    ui->nextendo_status->setTextFormat(Qt::PlainText);
    ui->nextendo_status->setTextInteractionFlags(Qt::TextSelectableByMouse);
    RetranslateUI();
    this->SetConfiguration();

    ui->nextendo_group->setVisible(!per_game);
    // Global settings: per-game configuration doesn't offer them.
    ui->redirect_group->setVisible(!per_game);
    if (!per_game) {
        RenewNextendoToken();
    }

    connect(ui->nextendo_enabled, &QCheckBox::toggled, this, [this](bool enabled) {
        if (enabled) {
            ui->airplane_mode->setChecked(false);
        }
    });
    connect(ui->redirect_ca_browse, &QPushButton::clicked, this, [this] {
        const QString path = QFileDialog::getOpenFileName(
            this, tr("Select the CA certificate"), QFileInfo(ui->redirect_ca->text()).dir().path(),
            tr("DER certificates (*.der *.cer *.crt);;All files (*)"));
        if (!path.isEmpty()) {
            ui->redirect_ca->setText(QDir::toNativeSeparators(path));
        }
    });
    connect(ui->nextendo_sign_in, &QPushButton::clicked, this, &ConfigureNetwork::SignInToNextendo);
    connect(ui->nextendo_sign_out, &QPushButton::clicked, this, [this] {
        Common::Nextendo::ClearAccount();
        nextendo_online.reset();
        UpdateNextendoStatus();
    });
}

ConfigureNetwork::~ConfigureNetwork() {
    nextendo_stop.request_stop();
}

void ConfigureNetwork::ApplyConfiguration() {
    Settings::values.network_interface = ui->network_interface->currentText().toStdString();
    Settings::values.airplane_mode = ui->airplane_mode->isChecked();
    Settings::values.lan_play = ui->lan_play->isChecked();
    if (!per_game) {
        Settings::values.nextendo_enabled = ui->nextendo_enabled->isChecked();
        Settings::values.nextendo_client_id = NextendoClientId(ui->nextendo_client_id->text());
        Settings::values.online_service_redirect = ui->online_service_redirect->isChecked();
        Settings::values.online_redirect_hosts = ui->redirect_hosts->text().trimmed().toStdString();
        Settings::values.online_redirect_server =
            ui->redirect_server->text().trimmed().toStdString();
        Settings::values.online_redirect_ca = ui->redirect_ca->text().trimmed().toStdString();
        Settings::values.online_redirect_loopback_only = ui->redirect_loopback_only->isChecked();
    }
}

void ConfigureNetwork::changeEvent(QEvent* event) {
    if (event->type() == QEvent::LanguageChange) {
        RetranslateUI();
    }

    QWidget::changeEvent(event);
}

void ConfigureNetwork::RetranslateUI() {
    ui->retranslateUi(this);
    if constexpr (!Common::Nextendo::OnlinePlayAvailable) {
        ui->nextendo_enabled->setVisible(false);
        ui->nextendo_description->setText(
            tr("Sign in with your Nextendo Network account. Online play through Nextendo isn't "
               "available in this build yet."));
    }
    if (nextendo_signing_in) {
        ui->nextendo_sign_in->setText(tr("Cancel sign-in"));
    }
}

void ConfigureNetwork::SetConfiguration() {
    const bool runtime_lock = !system.IsPoweredOn();
    auto const network_interface = Settings::values.network_interface.GetValue();
    auto const airplane_mode = Settings::values.airplane_mode.GetValue();
    ui->network_interface->setCurrentText(QString::fromStdString(network_interface));
    ui->network_interface->setEnabled(runtime_lock);
    ui->airplane_mode->setChecked(airplane_mode);
    ui->lan_play->setChecked(Settings::values.lan_play.GetValue());
    ui->online_service_redirect->setChecked(Settings::values.online_service_redirect.GetValue());
    ui->redirect_hosts->setText(
        QString::fromStdString(Settings::values.online_redirect_hosts.GetValue()));
    ui->redirect_server->setText(
        QString::fromStdString(Settings::values.online_redirect_server.GetValue()));
    ui->redirect_ca->setText(QString::fromStdString(Settings::values.online_redirect_ca.GetValue()));
    ui->redirect_loopback_only->setChecked(
        Settings::values.online_redirect_loopback_only.GetValue());
    ui->redirect_group->setEnabled(runtime_lock);
    ui->nextendo_enabled->setChecked(Settings::values.nextendo_enabled.GetValue());
    ui->nextendo_client_id->setText(
        QString::fromStdString(Settings::values.nextendo_client_id.GetValue()));
    UpdateNextendoStatus();
}

void ConfigureNetwork::SignInToNextendo() {
    if (nextendo_signing_in) {
        nextendo_stop.request_stop();
        return;
    }
    const std::string client_id = NextendoClientId(ui->nextendo_client_id->text());
    nextendo_stop = std::stop_source{};
    nextendo_signing_in = true;
    ui->nextendo_sign_in->setText(tr("Cancel sign-in"));
    ui->nextendo_status->setText(tr("Finish signing in to Nextendo in your browser."));

    auto* watcher = new QFutureWatcher<Common::Nextendo::SignInResult>(this);
    connect(watcher, &QFutureWatcher<Common::Nextendo::SignInResult>::finished, this,
            [this, watcher] {
                const auto result = watcher->result();
                watcher->deleteLater();
                nextendo_signing_in = false;
                ui->nextendo_sign_in->setText(tr("Sign in to Nextendo"));
                if (result.ok) {
                    nextendo_online = result.online_status;
                }
                UpdateNextendoStatus(
                    QString::fromStdString(result.ok ? result.game_token_error : result.error),
                    result.ok);
            });
    const QPointer<ConfigureNetwork> self{this};
    watcher->setFuture(QtConcurrent::run([client_id, self, stop = nextendo_stop.get_token()] {
        return Common::Nextendo::SignIn(
            client_id,
            [self](const std::string& url) {
                QMetaObject::invokeMethod(
                    QCoreApplication::instance(),
                    [self, address = QString::fromStdString(url)] {
                        if (!QDesktopServices::openUrl(QUrl{address}) && self) {
                            self->ui->nextendo_status->setText(
                                tr("Open this address in your browser to sign in: %1")
                                    .arg(address));
                        }
                    },
                    Qt::QueuedConnection);
            },
            stop);
    }));
}

void ConfigureNetwork::RenewNextendoToken() {
    if (!Common::Nextendo::LoadAccount()) {
        return;
    }
    const std::string client_id = NextendoClientId(ui->nextendo_client_id->text());
    using Status = std::optional<Common::Nextendo::OnlineStatus>;
    auto* watcher = new QFutureWatcher<Status>(this);
    connect(watcher, &QFutureWatcher<Status>::finished, this, [this, watcher] {
        const Status status = watcher->result();
        watcher->deleteLater();
        if (!nextendo_signing_in) {
            nextendo_online = status;
            UpdateNextendoStatus();
        }
    });
    watcher->setFuture(
        QtConcurrent::run([client_id] { return Common::Nextendo::CheckOnlineStatus(client_id); }));
}

void ConfigureNetwork::UpdateNextendoStatus(const QString& error, bool signed_in) {
    const auto account = Common::Nextendo::LoadAccount();
    ui->nextendo_sign_out->setEnabled(account.has_value());
    QString status;
    if (!error.isEmpty() && !signed_in) {
        status = tr("Sign-in failed: %1").arg(error);
    } else if (!account) {
        switch (Common::Nextendo::LastSignOutReason()) {
        case Common::Nextendo::SignOutReason::SessionExpired:
            status = tr("Your Nextendo session expired. Sign in again.");
            break;
        case Common::Nextendo::SignOutReason::SessionRevoked:
            status = tr("Your session was ended on nextendo.network. Sign in again.");
            break;
        case Common::Nextendo::SignOutReason::None:
            status = tr("Not signed in.");
            break;
        }
    } else {
        status = account->username.empty()
                     ? tr("Signed in.")
                     : tr("Signed in as %1.").arg(QString::fromStdString(account->username));
        if (account->nex_token.empty()) {
            status += QLatin1Char(' ') +
                      tr("Nextendo did not issue an online play token, so games can't log in "
                         "to Nextendo yet.");
            if (!error.isEmpty()) {
                status += QLatin1Char(' ') + error;
            }
        } else if (nextendo_online && !nextendo_online->allow) {
            status += QLatin1Char(' ') + OnlineRefusal(nextendo_online->reason);
        }
    }
    ui->nextendo_status->setText(status);
}
