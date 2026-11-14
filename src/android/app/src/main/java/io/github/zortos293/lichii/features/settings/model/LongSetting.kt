// SPDX-License-Identifier: GPL-2.0-or-later

package io.github.zortos293.lichii.features.settings.model

import io.github.zortos293.lichii.utils.NativeConfig

enum class LongSetting(override val key: String) : AbstractLongSetting {
    CUSTOM_RTC("custom_rtc");

    override fun getLong(needsGlobal: Boolean): Long = NativeConfig.getLong(key, needsGlobal)

    override fun setLong(value: Long) {
        if (NativeConfig.isPerGameConfigLoaded()) {
            global = false
        }
        NativeConfig.setLong(key, value)
    }

    override val defaultValue: Long by lazy { NativeConfig.getDefaultToString(key).toLong() }

    override fun getValueAsString(needsGlobal: Boolean): String = getLong(needsGlobal).toString()

    override fun reset() = NativeConfig.setLong(key, defaultValue)
}
