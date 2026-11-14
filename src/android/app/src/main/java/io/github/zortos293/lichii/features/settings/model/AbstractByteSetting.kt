// SPDX-License-Identifier: GPL-2.0-or-later

package io.github.zortos293.lichii.features.settings.model

interface AbstractByteSetting : AbstractSetting {
    fun getByte(needsGlobal: Boolean = false): Byte
    fun setByte(value: Byte)
}
