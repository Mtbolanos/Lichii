// SPDX-License-Identifier: GPL-2.0-or-later

package io.github.zortos293.lichii.features.input

import android.view.InputDevice
import androidx.annotation.Keep
import io.github.zortos293.lichii.LichiiApplication
import io.github.zortos293.lichii.R
import io.github.zortos293.lichii.utils.InputHandler.getGUID

@Keep
interface LichiiInputDevice {
    fun getName(): String

    fun getGUID(): String

    fun getPort(): Int

    fun getSupportsVibration(): Boolean

    fun vibrate(intensity: Float)

    fun getAxes(): Array<Int> = arrayOf()
    fun hasKeys(keys: IntArray): BooleanArray = BooleanArray(0)
}

class LichiiPhysicalDevice(
    private val device: InputDevice,
    private val port: Int,
    useSystemVibrator: Boolean
) : LichiiInputDevice {
    private val vibrator = if (useSystemVibrator) {
        LichiiVibrator.getSystemVibrator()
    } else {
        LichiiVibrator.getControllerVibrator(device)
    }

    override fun getName(): String {
        return device.name
    }

    override fun getGUID(): String {
        return device.getGUID()
    }

    override fun getPort(): Int {
        return port
    }

    override fun getSupportsVibration(): Boolean {
        return vibrator.supportsVibration()
    }

    override fun vibrate(intensity: Float) {
        vibrator.vibrate(intensity)
    }

    override fun getAxes(): Array<Int> = device.motionRanges.map { it.axis }.toTypedArray()
    override fun hasKeys(keys: IntArray): BooleanArray = device.hasKeys(*keys)
}

class LichiiInputOverlayDevice(
    private val vibration: Boolean,
    private val port: Int
) : LichiiInputDevice {
    private val vibrator = LichiiVibrator.getSystemVibrator()

    override fun getName(): String {
        return LichiiApplication.appContext.getString(R.string.input_overlay)
    }

    override fun getGUID(): String {
        return "00000000000000000000000000000000"
    }

    override fun getPort(): Int {
        return port
    }

    override fun getSupportsVibration(): Boolean {
        if (vibration) {
            return vibrator.supportsVibration()
        }
        return false
    }

    override fun vibrate(intensity: Float) {
        if (vibration) {
            vibrator.vibrate(intensity)
        }
    }
}
