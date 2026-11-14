// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "performance_overlay.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <QFontMetricsF>
#include <QGuiApplication>
#include <QMouseEvent>
#include <QPainter>
#include <QPolygonF>
#include <QStringList>
#include <QWindow>

#include "common/settings.h"
#include "core/core.h"
#include "main_window.h"
#include "qt_common/qt_common.h"
#include "video_core/gpu.h"
#include "video_core/renderer_base.h"

namespace {

using Common::DkbCpuTrace::Role;

// Layout of the "6 · 3 + big number" design at 100% scale; Qt scales it with the DPI.
constexpr int OverlayWidth = 300;
constexpr int OverlayHeight = 158;
constexpr qreal PaddingX = 10;
constexpr qreal PaddingTop = 6;
constexpr qreal ContentWidth = 280;
constexpr qreal HeaderHeight = 44;
constexpr qreal GraphTop = PaddingTop + HeaderHeight + 4;
constexpr qreal GraphHeight = 72;
constexpr qreal PlotWidth = 236;
constexpr qreal AxisX = 240;
constexpr qreal LabelX = 247;
constexpr qreal LabelHeight = 14;
constexpr qreal FooterTop = GraphTop + GraphHeight + 10;

constexpr auto HistoryWindow = std::chrono::seconds{5};
constexpr auto StatsWindow = std::chrono::milliseconds{500};
constexpr std::size_t FpsSamples = 10;
/// A frame this much slower than the median is a spike.
constexpr float SpikeRatio = 1.2f;

constexpr QRgb TextColor = qRgb(0xF2, 0xF3, 0xF5);
constexpr QRgb MutedColor = qRgb(0xA0, 0xA5, 0xAD);
constexpr QRgb DimColor = qRgb(0x6E, 0x73, 0x7B);
constexpr QRgb AccentColor = qRgb(0x5E, 0xB4, 0xFF);
constexpr QRgb SpikeColor = qRgb(0xFF, 0x5D, 0x52);
constexpr QRgb BackgroundColor = qRgba(0, 0, 0, 140);
constexpr QRgb LineColor = qRgba(0xF2, 0xF3, 0xF5, 217);
constexpr QRgb AxisColor = qRgba(255, 255, 255, 46);
constexpr QRgb MedianLineColor = qRgba(255, 255, 255, 56);

/// The busiest of these threads is the CPU side's limit: the guest cores and GPU emulation.
constexpr std::array CpuRoles{
    Role::GuestCore0, Role::GuestCore1,      Role::GuestCore2,  Role::GuestCore3, Role::GuestCore4,
    Role::GuestCore5, Role::GuestSingleCore, Role::GpuCommand, Role::GpuBack,
};

QFont MonoFont(int pixel_size, QFont::Weight weight = QFont::Normal) {
    QFont font;
    font.setFamilies({QStringLiteral("Geist Mono"), QStringLiteral("Cascadia Mono"),
                      QStringLiteral("JetBrains Mono"), QStringLiteral("Consolas"),
                      QStringLiteral("SF Mono"), QStringLiteral("Menlo"),
                      QStringLiteral("DejaVu Sans Mono"), QStringLiteral("Liberation Mono")});
    font.setStyleHint(QFont::Monospace);
    font.setPixelSize(pixel_size);
    font.setWeight(weight);
    font.setLetterSpacing(QFont::AbsoluteSpacing, (pixel_size >= 40 ? -0.03 : -0.01) * pixel_size);
    return font;
}

/// The baseline of text centred in a line box, as CSS places it.
qreal Baseline(const QFont& font, qreal top, qreal line_height) {
    const QFontMetricsF metrics{font};
    return top + (line_height - metrics.ascent() - metrics.descent()) / 2 + metrics.ascent();
}

struct Run {
    QString text;
    QFont font;
    QColor color;
};

qreal Width(const std::vector<Run>& runs) {
    qreal width = 0;
    for (const Run& run : runs) {
        width += QFontMetricsF{run.font}.horizontalAdvance(run.text);
    }
    return width;
}

void DrawRuns(QPainter& painter, qreal x, qreal baseline, const std::vector<Run>& runs) {
    for (const Run& run : runs) {
        painter.setFont(run.font);
        painter.setPen(run.color);
        painter.drawText(QPointF{x, baseline}, run.text);
        x += QFontMetricsF{run.font}.horizontalAdvance(run.text);
    }
}

QString Percent(const std::optional<double>& percent) {
    return percent ? QStringLiteral("%1%").arg(std::round(*percent), 0, 'f', 0)
                   : QStringLiteral("—");
}

} // namespace

PerformanceOverlay::PerformanceOverlay(MainWindow* parent) : QWidget(parent), m_mainWindow{parent} {
    setAttribute(Qt::WA_TranslucentBackground);
    setWindowFlags(Qt::FramelessWindowHint | Qt::Tool);
    setFixedSize(OverlayWidth, OverlayHeight);
    raise();

    // pos/stats
    resetPosition(m_mainWindow->pos());
    connect(parent, &MainWindow::positionChanged, this, &PerformanceOverlay::resetPosition);
    connect(m_mainWindow, &MainWindow::statsUpdated, this, &PerformanceOverlay::updateStats);
}

void PerformanceOverlay::resetPosition(const QPoint& _) {
    auto pos = m_mainWindow->pos();
    move(pos.x() + m_offset.x(), pos.y() + m_offset.y());
}

void PerformanceOverlay::updateStats(const Core::PerfStatsResults& results,
                                     const VideoCore::ShaderNotify& shaders) {
    auto& system = *QtCommon::system;
    auto& perf_stats = system.GetPerfStats();
    perf_stats.SetFrameTimeHistoryEnabled(isVisible());
    if (!isVisible()) {
        return;
    }

    m_fps.reset();
    if (!std::isnan(results.average_game_fps)) {
        m_fps = results.average_game_fps;
        m_fpsSamples.push_back(static_cast<float>(std::round(*m_fps)));
        if (m_fpsSamples.size() > FpsSamples) {
            m_fpsSamples.pop_front();
        }
    }

    m_frames = perf_stats.GetRecentFrameTimes(HistoryWindow);
    m_frameMs.reset();
    if (!m_frames.empty() &&
        Core::FrameTimeHistory::Clock::now() - m_frames.back().end < std::chrono::seconds{1}) {
        double sum = 0;
        int count = 0;
        for (auto it = m_frames.rbegin();
             it != m_frames.rend() && m_frames.back().end - it->end < StatsWindow; ++it) {
            sum += it->ms;
            ++count;
        }
        m_frameMs = sum / count;
    }

    // Busy time over the window against the window is busy ms per frame against frame time.
    auto cpu_snapshot = Common::DkbCpuTrace::ReadSnapshot();
    m_cpuPercent.reset();
    if (m_cpuSnapshot) {
        const double wall_ns = std::chrono::duration<double, std::nano>(
                                   cpu_snapshot.wall_time - m_cpuSnapshot->wall_time)
                                   .count();
        std::optional<u64> busiest_ns;
        for (const Role role : CpuRoles) {
            const auto index = static_cast<size_t>(role);
            if (const auto delta = Common::DkbCpuTrace::CpuTimeDelta(m_cpuSnapshot->threads[index],
                                                                     cpu_snapshot.threads[index])) {
                busiest_ns = std::max(busiest_ns.value_or(0), *delta);
            }
        }
        if (busiest_ns && wall_ns > 0) {
            m_cpuPercent = std::min(100.0, static_cast<double>(*busiest_ns) / wall_ns * 100.0);
        }
    }
    m_cpuSnapshot = cpu_snapshot;

    auto& renderer = system.GPU().Renderer();
    renderer.RequestGpuBusySample();
    const auto gpu_ms = renderer.GetGpuBusyMs();
    m_gpuPercent.reset();
    if (gpu_ms && m_frameMs && *m_frameMs > 0) {
        m_gpuPercent = std::min(100.0, *gpu_ms / *m_frameMs * 100.0);
    }

    QStringList tags;
    if (const QString& dlss = m_mainWindow->DlssStatusText(); !dlss.isEmpty()) {
        tags << dlss;
    }
    switch (Settings::values.scaling_filter.GetValue()) {
    case Settings::ScalingFilter::Fsr:
        tags << QStringLiteral("FSR");
        break;
    case Settings::ScalingFilter::Fsr3:
        tags << QStringLiteral("FSR 3");
        break;
    case Settings::ScalingFilter::Sgsr:
    case Settings::ScalingFilter::SgsrEdge:
        tags << QStringLiteral("SGSR");
        break;
    default:
        break;
    }
    if (const auto& scaling = Settings::values.resolution_info; scaling.active) {
        tags << QStringLiteral("%1×").arg(scaling.up_factor, 0, 'g', 3);
    }
    if (const double generated = m_mainWindow->FrameGenFps(); generated >= 0.5) {
        tags << tr("FG +%1").arg(std::round(generated), 0, 'f', 0);
    }
    m_tags = tags.join(QStringLiteral("  "));

    update();
}

void PerformanceOverlay::paintEvent(QPaintEvent* event) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setRenderHint(QPainter::TextAntialiasing);

    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor::fromRgba(BackgroundColor));
    painter.drawRoundedRect(rect(), 5.0, 5.0);

    const QColor text = QColor::fromRgb(TextColor);
    const QColor muted = QColor::fromRgb(MutedColor);
    const QColor accent = QColor::fromRgb(AccentColor);
    const QFont fps_font = MonoFont(40, QFont::DemiBold);
    const QFont small_font = MonoFont(13);
    const QFont small_bold_font = MonoFont(13, QFont::DemiBold);
    const QFont footer_font = MonoFont(11);

    // Header: the big fps number on the left, frame time and CPU/GPU % on the right.
    const qreal fps_baseline = Baseline(fps_font, PaddingTop, HeaderHeight);
    const std::vector<Run> fps_runs{
        {m_fps ? QString::number(std::round(*m_fps), 'f', 0) : QStringLiteral("—"), fps_font, text},
    };
    DrawRuns(painter, PaddingX, fps_baseline, fps_runs);
    DrawRuns(painter, PaddingX + Width(fps_runs) + 6, fps_baseline,
             {{QStringLiteral("fps"), small_font, muted}});

    const qreal right = PaddingX + ContentWidth;
    const qreal right_top = PaddingTop + (HeaderHeight - 34) / 2;
    const std::vector<Run> ms_runs{
        {m_frameMs ? QString::number(*m_frameMs, 'f', 1) : QStringLiteral("—"), small_bold_font,
         text},
        {QStringLiteral(" ms"), small_font, muted},
    };
    DrawRuns(painter, right - Width(ms_runs), Baseline(small_font, right_top, 17), ms_runs);

    // The higher percentage is the limit and gets the accent.
    const bool both = m_cpuPercent && m_gpuPercent;
    const bool cpu_limit = both && std::round(*m_cpuPercent) >= std::round(*m_gpuPercent);
    const bool gpu_limit = both && !cpu_limit;
    const std::vector<Run> load_runs{
        {QStringLiteral("CPU ") + Percent(m_cpuPercent), cpu_limit ? small_bold_font : small_font,
         cpu_limit ? accent : muted},
        {QStringLiteral("  "), small_font, muted},
        {QStringLiteral("GPU ") + Percent(m_gpuPercent), gpu_limit ? small_bold_font : small_font,
         gpu_limit ? accent : muted},
    };
    DrawRuns(painter, right - Width(load_runs), Baseline(small_font, right_top + 17, 17),
             load_runs);

    paintGraph(painter, QRectF{PaddingX, GraphTop, ContentWidth, GraphHeight});

    // Footer: fps min/median/max over the last 5 s, and the features that are on.
    QString fps_range = QStringLiteral("— — —");
    const std::vector<float> fps_samples(m_fpsSamples.begin(), m_fpsSamples.end());
    if (const auto summary = Core::Summarize(fps_samples)) {
        fps_range = QStringLiteral("%1 %2 %3")
                        .arg(summary->min_ms, 0, 'f', 0)
                        .arg(std::round(summary->median_ms), 0, 'f', 0)
                        .arg(summary->max_ms, 0, 'f', 0);
    }
    const qreal footer_baseline = Baseline(footer_font, FooterTop, LabelHeight);
    const std::vector<Run> range_runs{
        {fps_range, footer_font, muted},
        {QStringLiteral(" fps"), footer_font, QColor::fromRgb(DimColor)},
    };
    DrawRuns(painter, PaddingX, footer_baseline, range_runs);

    const qreal tags_width = ContentWidth - Width(range_runs) - 12;
    const QString tags =
        QFontMetricsF{footer_font}.elidedText(m_tags, Qt::ElideRight, tags_width);
    DrawRuns(painter, right - QFontMetricsF{footer_font}.horizontalAdvance(tags), footer_baseline,
             {{tags, footer_font, muted}});
}

void PerformanceOverlay::paintGraph(QPainter& painter, const QRectF& area) {
    painter.fillRect(QRectF{area.left() + AxisX, area.top(), 1, area.height()},
                     QColor::fromRgba(AxisColor));
    if (m_frames.size() < 2) {
        return;
    }

    std::vector<float> frame_ms;
    frame_ms.reserve(m_frames.size());
    for (const auto& frame : m_frames) {
        frame_ms.push_back(frame.ms);
    }
    const auto summary = *Core::Summarize(frame_ms);

    // Min and max span the plot; a steady frame rate keeps a small span so jitter stays flat.
    float low = summary.min_ms;
    float high = summary.max_ms;
    const float min_span = std::max(2.0f, summary.median_ms * 0.1f);
    if (high - low < min_span) {
        const float middle = (low + high) / 2;
        low = middle - min_span / 2;
        high = middle + min_span / 2;
    }
    const qreal plot_top = area.top() + 6;
    const qreal plot_bottom = area.bottom() - 7;
    const auto y_of = [&](float ms) {
        return plot_top + (high - ms) / (high - low) * (plot_bottom - plot_top);
    };
    const auto newest = m_frames.back().end;
    const double window_s = std::chrono::duration<double>(HistoryWindow).count();

    QPolygonF line;
    line.reserve(static_cast<qsizetype>(m_frames.size()));
    for (const auto& frame : m_frames) {
        const double age_s = std::chrono::duration<double>(newest - frame.end).count();
        line << QPointF{area.left() + PlotWidth * (1.0 - age_s / window_s), y_of(frame.ms)};
    }

    painter.setBrush(Qt::NoBrush);
    QPen median_pen{QColor::fromRgba(MedianLineColor), 0.75};
    median_pen.setDashPattern({2.0 / 0.75, 3.0 / 0.75});
    painter.setPen(median_pen);
    const qreal median_y = y_of(summary.median_ms);
    painter.drawLine(QPointF{area.left(), median_y}, QPointF{area.left() + PlotWidth, median_y});

    QPen line_pen{QColor::fromRgba(LineColor), 1.1};
    line_pen.setJoinStyle(Qt::RoundJoin);
    painter.setPen(line_pen);
    painter.drawPolyline(line);

    const float spike_ms = summary.median_ms * SpikeRatio;
    QPen spike_pen{QColor::fromRgb(SpikeColor), 1.1};
    spike_pen.setCapStyle(Qt::RoundCap);
    spike_pen.setJoinStyle(Qt::RoundJoin);
    painter.setPen(spike_pen);
    for (qsizetype i = 1; i < line.size(); ++i) {
        if (frame_ms[i - 1] > spike_ms || frame_ms[i] > spike_ms) {
            painter.drawLine(line[i - 1], line[i]);
        }
    }

    // Min, median and max as ticks on the axis at their real heights, labelled beside them.
    struct Tick {
        float ms;
        QColor color;
        QFont font;
    };
    const std::array<Tick, 3> ticks{{
        {summary.max_ms,
         QColor::fromRgb(summary.max_ms > spike_ms ? SpikeColor : TextColor),
         MonoFont(11, QFont::Medium)},
        {summary.median_ms, QColor::fromRgb(TextColor), MonoFont(11, QFont::Medium)},
        {summary.min_ms, QColor::fromRgb(MutedColor), MonoFont(11)},
    }};
    std::array<qreal, 3> label_tops{};
    for (std::size_t i = 0; i < ticks.size(); ++i) {
        const qreal y = y_of(ticks[i].ms);
        painter.fillRect(QRectF{area.left() + AxisX - 2, y - 0.5, 5, 1}, ticks[i].color);
        label_tops[i] = y - LabelHeight / 2;
    }
    // Push overlapping labels apart, keeping them within the graph and the footer's padding.
    for (std::size_t i = 1; i < label_tops.size(); ++i) {
        label_tops[i] = std::max(label_tops[i], label_tops[i - 1] + LabelHeight);
    }
    label_tops[2] = std::min(label_tops[2], area.bottom() + 7 - LabelHeight);
    for (std::size_t i = label_tops.size() - 1; i-- > 0;) {
        label_tops[i] = std::min(label_tops[i], label_tops[i + 1] - LabelHeight);
    }
    for (std::size_t i = 0; i < ticks.size(); ++i) {
        DrawRuns(painter, area.left() + LabelX,
                 Baseline(ticks[i].font, label_tops[i], LabelHeight),
                 {{QString::number(ticks[i].ms, 'f', 1), ticks[i].font, ticks[i].color}});
    }
}

void PerformanceOverlay::hideEvent(QHideEvent* event) {
    QWidget::hideEvent(event);
    if (QtCommon::system && QtCommon::system->IsPoweredOn()) {
        QtCommon::system->GetPerfStats().SetFrameTimeHistoryEnabled(false);
    }
    m_fps.reset();
    m_fpsSamples.clear();
    m_frameMs.reset();
    m_cpuPercent.reset();
    m_gpuPercent.reset();
    m_cpuSnapshot.reset();
    m_frames.clear();
}

void PerformanceOverlay::mousePressEvent(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton) {
        return;
    }
    // Wayland ignores move() on top-level windows, so the overlay sat where the compositor put it:
    // let the compositor drag it instead.
    if (QGuiApplication::platformName().startsWith(QStringLiteral("wayland")) && windowHandle() &&
        windowHandle()->startSystemMove()) {
        return;
    }
    m_drag_start_pos = event->pos();
}

void PerformanceOverlay::mouseMoveEvent(QMouseEvent* event) {
    // drag
    if (event->buttons() & Qt::LeftButton) {
        QPoint new_global_pos = event->globalPosition().toPoint() - m_drag_start_pos;
        m_offset = new_global_pos - m_mainWindow->pos();
        move(new_global_pos);
    }
}

void PerformanceOverlay::closeEvent(QCloseEvent* event) {
    emit closed();
}
