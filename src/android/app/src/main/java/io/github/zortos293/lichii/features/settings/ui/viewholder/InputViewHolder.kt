// SPDX-License-Identifier: GPL-2.0-or-later

package io.github.zortos293.lichii.features.settings.ui.viewholder

import android.view.View
import io.github.zortos293.lichii.databinding.ListItemSettingInputBinding
import io.github.zortos293.lichii.features.input.NativeInput
import io.github.zortos293.lichii.features.settings.model.view.AnalogInputSetting
import io.github.zortos293.lichii.features.settings.model.view.ButtonInputSetting
import io.github.zortos293.lichii.features.settings.model.view.InputSetting
import io.github.zortos293.lichii.features.settings.model.view.ModifierInputSetting
import io.github.zortos293.lichii.features.settings.model.view.SettingsItem
import io.github.zortos293.lichii.features.settings.ui.SettingsAdapter
import io.github.zortos293.lichii.utils.ViewUtils.setVisible

class InputViewHolder(val binding: ListItemSettingInputBinding, adapter: SettingsAdapter) :
    SettingViewHolder(binding.root, adapter) {
    private lateinit var setting: InputSetting

    override fun bind(item: SettingsItem) {
        setting = item as InputSetting
        binding.textSettingName.text = setting.title
        binding.textSettingValue.text = setting.getSelectedValue()

        when (item) {
            is AnalogInputSetting -> {
                val param = NativeInput.getStickParam(item.playerIndex, item.nativeAnalog)
                binding.buttonOptions.setVisible(
                    param.get("engine", "") == "analog_from_button" ||
                        param.has("axis_x") || param.has("axis_y")
                )
            }

            is ButtonInputSetting -> {
                val param = NativeInput.getButtonParam(item.playerIndex, item.nativeButton)
                binding.buttonOptions.setVisible(
                    param.has("code") || param.has("button") || param.has("hat") ||
                        param.has("axis")
                )
            }

            is ModifierInputSetting -> {
                val params = NativeInput.getStickParam(item.playerIndex, item.nativeAnalog)
                binding.buttonOptions.setVisible(params.has("modifier"))
            }
        }

        binding.buttonOptions.setOnClickListener(null)
        binding.buttonOptions.setOnClickListener {
            adapter.onInputOptionsClick(binding.buttonOptions, setting, bindingAdapterPosition)
        }
    }

    override fun onClick(clicked: View) =
        adapter.onInputClick(setting, bindingAdapterPosition)

    override fun onLongClick(clicked: View): Boolean =
        adapter.onLongClick(setting, bindingAdapterPosition)
}
