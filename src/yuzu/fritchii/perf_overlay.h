// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <deque>
#include <functional>

#include <QWidget>

class MainWindow;

namespace Core {
struct PerfStatsResults;
}
namespace VideoCore {
class ShaderNotify;
}

namespace FritchiiUi {

/// Fritchii's performance overlay (View > Performance Overlay): a Graphite panel with the frame
/// rate in big type, the frame time and the lowest recent frame rate, and a violet line of the
/// last minute and a half against the target (60, or 30 with OoT at 30 FPS), plus Fritchii's RAM
/// and VRAM with what is left of each (Windows). Freezes and loading screens break the line instead of pulling
/// it to zero. Drag it anywhere; right-click hides it.
class PerfOverlay : public QWidget {
    Q_OBJECT
public:
    PerfOverlay(MainWindow* main_window, std::function<void()> on_hidden);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;

private:
    void OnStats(const Core::PerfStatsResults& results);
    void FollowWindow();

    MainWindow* main_window;
    std::function<void()> on_hidden;
    std::deque<double> fps_samples; ///< NaN marks a gap (freeze, loading)
    double fps = 0;
    double frametime_ms = 0;
    double ram_used_gb = -1;  ///< Fritchii's working set (-1: unknown on this OS)
    double ram_free_gb = -1;  ///< RAM the PC has free
    double vram_used_gb = -1; ///< Fritchii's video memory on its GPU (-1: unknown)
    double vram_left_gb = -1; ///< What Windows still lets Fritchii use there (budget - used)
    QPoint offset{24, 80};
    QPoint drag_start;
};

} // namespace FritchiiUi
