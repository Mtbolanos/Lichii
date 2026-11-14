// SPDX-License-Identifier: GPL-2.0-or-later

package io.github.zortos293.lichii.features.settings.model

interface AbstractFloatSetting : AbstractSetting {
    fun getFloat(needsGlobal: Boolean = false): Float
    fun setFloat(value: Float)
}
