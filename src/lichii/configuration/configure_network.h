// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <optional>
#include <stop_token>
#include <QWidget>
#include "common/nextendo.h"

namespace Ui {
class ConfigureNetwork;
}

class ConfigureNetwork : public QWidget {
    Q_OBJECT

public:
    explicit ConfigureNetwork(const Core::System& system_, QWidget* parent = nullptr,
                              bool per_game = false);
    ~ConfigureNetwork() override;

    void ApplyConfiguration();

private:
    void changeEvent(QEvent*) override;
    void RetranslateUI();
    void SetConfiguration();
    void SignInToNextendo();
    void RenewNextendoToken();
    void UpdateNextendoStatus(const QString& error = {}, bool signed_in = false);

    std::unique_ptr<Ui::ConfigureNetwork> ui;
    std::stop_source nextendo_stop;
    bool nextendo_signing_in{};
    std::optional<Common::Nextendo::OnlineStatus> nextendo_online;

    const Core::System& system;
    // The account is global, so a game's properties don't offer it.
    const bool per_game;
};
