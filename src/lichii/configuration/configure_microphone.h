// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <QWidget>

class QComboBox;
class QLabel;
class QProgressBar;
class QPushButton;
class QSlider;
class QTimer;
namespace Core {
class System;
}
namespace AudioCore::Sink {
class Sink;
class SinkStream;
} // namespace AudioCore::Sink

class ConfigureMicrophone final : public QWidget {
    Q_OBJECT
public:
    explicit ConfigureMicrophone(Core::System& system, QWidget* parent = nullptr);
    ~ConfigureMicrophone() override;
    void ApplyConfiguration();

protected:
    void hideEvent(QHideEvent* event) override;

private:
    void RefreshDevices();
    void StartTest();
    void StopTest();

    Core::System& system;
    QComboBox* devices;
    QPushButton* refresh;
    QPushButton* test;
    QSlider* volume;
    QLabel* volume_label;
    QLabel* status;
    QProgressBar* level;
    QTimer* timer;
    std::unique_ptr<AudioCore::Sink::Sink> preview_sink;
    AudioCore::Sink::SinkStream* preview_stream{};
};
