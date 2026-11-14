// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <QComboBox>
#include <QFormLayout>
#include <QHideEvent>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QSlider>
#include <QTimer>
#include <QVBoxLayout>

#include "audio_core/sink/sink.h"
#include "audio_core/sink/sink_details.h"
#include "common/settings.h"
#include "core/core.h"
#include "lichii/configuration/configure_microphone.h"

/// Build device, gain, and level controls; keep device selection fixed during emulation.
ConfigureMicrophone::ConfigureMicrophone(Core::System& system_, QWidget* parent)
    : QWidget{parent}, system{system_} {
    setAccessibleName(tr("Microphone"));
    auto* layout = new QVBoxLayout(this);
    auto* description = new QLabel(tr("Choose the microphone used by games. Start a test and speak "
                                      "to check the input level."),
                                   this);
    description->setWordWrap(true);
    layout->addWidget(description);
    auto* form = new QFormLayout;
    devices = new QComboBox(this);
    form->addRow(tr("Input device:"), devices);
    refresh = new QPushButton(tr("Refresh devices"), this);
    form->addRow(QString{}, refresh);
    volume = new QSlider(Qt::Horizontal, this);
    volume->setRange(0, 200);
    volume->setValue(Settings::values.microphone_volume.GetValue());
    volume_label = new QLabel(tr("%1%").arg(volume->value()), this);
    auto* gain_layout = new QHBoxLayout;
    gain_layout->addWidget(volume);
    gain_layout->addWidget(volume_label);
    form->addRow(tr("Microphone volume:"), gain_layout);
    level = new QProgressBar(this);
    level->setRange(0, 100);
    level->setValue(0);
    form->addRow(tr("Input level:"), level);
    layout->addLayout(form);
    test = new QPushButton(tr("Start microphone test"), this);
    layout->addWidget(test);
    status = new QLabel(this);
    status->setWordWrap(true);
    layout->addWidget(status);
    layout->addStretch();
    timer = new QTimer(this);
    timer->setInterval(50);
    connect(timer, &QTimer::timeout, this, [this] {
        if (!preview_stream->IsReady()) {
            StopTest();
            status->setText(tr("Microphone disconnected or capture failed."));
            return;
        }
        level->setValue(static_cast<int>(preview_stream->TakeInputPeak() * 100 / 32768));
    });
    connect(test, &QPushButton::clicked, this, [this] {
        if (preview_stream) {
            StopTest();
        } else {
            StartTest();
        }
    });
    connect(refresh, &QPushButton::clicked, this, [this] { RefreshDevices(); });
    connect(devices, &QComboBox::currentIndexChanged, this, [this] { StopTest(); });
    connect(volume, &QSlider::valueChanged, this, [this](int value) {
        volume_label->setText(tr("%1%").arg(value));
        if (preview_stream) {
            preview_stream->SetInputPreviewGain(value / 100.0f);
        }
    });
    RefreshDevices();
    // Devices are bound when the audio core is created.
    devices->setEnabled(!system.IsPoweredOn());
    refresh->setEnabled(!system.IsPoweredOn());
    volume->setEnabled(!system.IsPoweredOn());
    if (system.IsPoweredOn()) {
        status->setText(tr("Stop emulation to change the microphone input device or volume."));
    }
}

ConfigureMicrophone::~ConfigureMicrophone() {
    StopTest();
}

/// Refresh inputs while retaining the selected device, even if temporarily unavailable.
void ConfigureMicrophone::RefreshDevices() {
    StopTest();
    auto selected = devices->currentData().toString();
    if (selected.isEmpty()) {
        selected = QString::fromStdString(Settings::values.audio_input_device_id.GetValue());
    }
    devices->clear();
    devices->addItem(tr("System default"), QStringLiteral("auto"));
    for (const auto& name :
         AudioCore::Sink::GetDeviceListForSink(Settings::values.sink_id.GetValue(), true)) {
        if (name != "null") {
            devices->addItem(QString::fromStdString(name), QString::fromStdString(name));
        }
    }
    auto index = devices->findData(selected);
    if (index < 0 && selected != QStringLiteral("auto")) {
        devices->addItem(tr("%1 (unavailable)").arg(selected), selected);
        index = devices->count() - 1;
    }
    devices->setCurrentIndex(index < 0 ? 0 : index);
}

/// Start independent capture for the level meter, reporting device or permission failures.
void ConfigureMicrophone::StartTest() {
    using namespace AudioCore::Sink;
    const auto available = GetDeviceListForSink(Settings::values.sink_id.GetValue(), true);
    const auto selected = devices->currentData().toString().toStdString();
    if (available == std::vector<std::string>{"null"} ||
        (selected != auto_device_name &&
         std::find(available.begin(), available.end(), selected) == available.end())) {
        status->setText(tr("Microphone input is unavailable. Choose an available input device."));
        return;
    }
    preview_sink = CreateSinkFromID(Settings::values.sink_id.GetValue(), selected);
    preview_stream = preview_sink->AcquireSinkStream(system, 2, "Microphone test", StreamType::In);
    if (preview_stream && preview_stream->IsReady()) {
        preview_stream->EnableInputPreview();
        preview_stream->SetInputPreviewGain(volume->value() / 100.0f);
        preview_stream->Start();
    }
    if (!preview_stream || !preview_stream->IsReady() || preview_stream->IsPaused()) {
        StopTest();
        status->setText(tr("Unable to open the microphone. Check the device and microphone "
                           "permissions in your operating system."));
        return;
    }
    test->setText(tr("Stop microphone test"));
    status->setText(tr("Listening. Speak into your microphone."));
    timer->start();
}

/// Stop metering and release preview capture before its controls or device selection change.
void ConfigureMicrophone::StopTest() {
    timer->stop();
    preview_sink.reset();
    preview_stream = nullptr;
    level->setValue(0);
    test->setText(tr("Start microphone test"));
    status->clear();
}

/// Save gain and the input device when emulation is stopped; release any active preview.
void ConfigureMicrophone::ApplyConfiguration() {
    StopTest();
    if (!system.IsPoweredOn()) {
        Settings::values.audio_input_device_id.SetValue(
            devices->currentData().toString().toStdString());
        Settings::values.microphone_volume.SetValue(static_cast<u16>(volume->value()));
    }
}

void ConfigureMicrophone::hideEvent(QHideEvent* event) {
    StopTest();
    QWidget::hideEvent(event);
}
