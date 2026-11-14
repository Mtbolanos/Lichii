// SPDX-License-Identifier: GPL-2.0-or-later

package io.github.zortos293.lichii.features.settings.model

interface AbstractStringSetting : AbstractSetting {
    fun getString(needsGlobal: Boolean = false): String
    fun setString(value: String)
}
