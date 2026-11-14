// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

package io.github.zortos293.lichii.features.settings.model

import io.github.zortos293.lichii.utils.NativeConfig

enum class ByteSetting(override val key: String) : AbstractByteSetting {
    AUDIO_VOLUME("volume"),
    GPU_LOG_LEVEL("gpu_log_level");

    override fun getByte(needsGlobal: Boolean): Byte = NativeConfig.getByte(key, needsGlobal)

    override fun setByte(value: Byte) {
        if (NativeConfig.isPerGameConfigLoaded()) {
            global = false
        }
        NativeConfig.setByte(key, value)
    }

    override val defaultValue: Byte by lazy { NativeConfig.getDefaultToString(key).toByte() }

    override fun getValueAsString(needsGlobal: Boolean): String = getByte(needsGlobal).toString()

    override fun reset() = NativeConfig.setByte(key, defaultValue)
}
