// SPDX-License-Identifier: GPL-2.0-or-later

package io.github.zortos293.lichii.features.settings.ui.viewholder

import android.view.View
import android.widget.CompoundButton
import io.github.zortos293.lichii.databinding.ListItemSettingSwitchBinding
import io.github.zortos293.lichii.features.settings.model.view.SettingsItem
import io.github.zortos293.lichii.features.settings.model.view.SwitchSetting
import io.github.zortos293.lichii.features.settings.ui.SettingsAdapter
import io.github.zortos293.lichii.utils.ViewUtils.setVisible

class SwitchSettingViewHolder(val binding: ListItemSettingSwitchBinding, adapter: SettingsAdapter) :
    SettingViewHolder(binding.root, adapter) {

    private lateinit var setting: SwitchSetting

    override fun bind(item: SettingsItem) {
        setting = item as SwitchSetting
        binding.textSettingName.text = setting.title
        binding.textSettingDescription.setVisible(setting.description.isNotEmpty())
        binding.textSettingDescription.text = setting.description

        // TODO(alekpop): A race condition occurs here if the button is clicked too fast
        binding.switchWidget.setOnCheckedChangeListener(null)
        binding.switchWidget.isChecked = setting.getIsChecked(setting.needsRuntimeGlobal)
        binding.switchWidget.setOnCheckedChangeListener { _: CompoundButton, _: Boolean ->
            adapter.onBooleanClick(setting, binding.switchWidget.isChecked, bindingAdapterPosition)
        }

        binding.buttonClear.setVisible(setting.clearable)
        binding.buttonClear.setOnClickListener {
            adapter.onClearClick(setting, bindingAdapterPosition)
        }

        setStyle(setting.isEditable, binding)
    }

    override fun onClick(clicked: View) {
        if (setting.isEditable) {
            binding.switchWidget.toggle()
        }
    }

    override fun onLongClick(clicked: View): Boolean {
        if (setting.isEditable) {
            return adapter.onLongClick(setting, bindingAdapterPosition)
        }
        return false
    }
}
