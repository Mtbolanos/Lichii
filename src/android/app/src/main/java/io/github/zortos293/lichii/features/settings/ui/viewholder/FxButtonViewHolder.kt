// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

package io.github.zortos293.lichii.features.settings.ui.viewholder

import android.view.View
import io.github.zortos293.lichii.databinding.ListItemSettingFxButtonBinding
import io.github.zortos293.lichii.features.settings.model.view.FxButtonSetting
import io.github.zortos293.lichii.features.settings.model.view.SettingsItem
import io.github.zortos293.lichii.features.settings.ui.SettingsAdapter

class FxButtonViewHolder(
    val binding: ListItemSettingFxButtonBinding,
    adapter: SettingsAdapter
) : SettingViewHolder(binding.root, adapter) {
    private lateinit var setting: FxButtonSetting

    override fun bind(item: SettingsItem) {
        setting = item as FxButtonSetting

        binding.fxButton.text = item.title
        binding.fxButton.setOnClickListener { setting.onClick.invoke() }
    }

    override fun onClick(clicked: View) {}

    override fun onLongClick(clicked: View): Boolean = true
}
