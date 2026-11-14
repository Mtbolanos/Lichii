// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

package io.github.zortos293.lichii.features.settings.model.view

class FxPresetSetting(
    titleString: String,
    descriptionString: String = "",
    val deletable: Boolean = false,
    val onApply: () -> Unit,
    val onDelete: () -> Unit = {}
) : SettingsItem(emptySetting, 0, titleString, 0, descriptionString) {
    override val type = TYPE_FX_PRESET
}
