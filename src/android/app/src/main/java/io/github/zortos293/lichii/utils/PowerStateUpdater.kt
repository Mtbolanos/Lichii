// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

package io.github.zortos293.lichii.utils

import android.os.Handler
import android.os.Looper
import io.github.zortos293.lichii.NativeLibrary
import io.github.zortos293.lichii.LichiiApplication

object PowerStateUpdater {

    private lateinit var handler: Handler
    private lateinit var runnable: Runnable
    private const val UPDATE_INTERVAL_MS = 1000L
    private var isStarted = false

    fun start() {
        if (isStarted) {
            return
        }

        val context = LichiiApplication.appContext

        handler = Handler(Looper.getMainLooper())
        runnable = Runnable {
            val info = PowerStateUtils.getBatteryInfo(context)
            NativeLibrary.updatePowerState(info[0], info[1] == 1, info[2] == 1)
            handler.postDelayed(runnable, UPDATE_INTERVAL_MS)
        }
        handler.post(runnable)
        isStarted = true
    }

    fun stop() {
        if (!isStarted) {
            return
        }
        if (::handler.isInitialized) {
            handler.removeCallbacks(runnable)
        }
        isStarted = false
    }
}
