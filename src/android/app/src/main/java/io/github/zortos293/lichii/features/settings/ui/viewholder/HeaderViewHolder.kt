// SPDX-License-Identifier: GPL-2.0-or-later

package io.github.zortos293.lichii.features.settings.ui.viewholder

import android.view.View
import io.github.zortos293.lichii.databinding.ListItemSettingsHeaderBinding
import io.github.zortos293.lichii.features.settings.model.view.SettingsItem
import io.github.zortos293.lichii.features.settings.ui.SettingsAdapter

class HeaderViewHolder(val binding: ListItemSettingsHeaderBinding, adapter: SettingsAdapter) :
    SettingViewHolder(binding.root, adapter) {

    init {
        itemView.setOnClickListener(null)
    }

    override fun bind(item: SettingsItem) {
        binding.textHeaderName.text = item.title
    }

    override fun onClick(clicked: View) {
        // no-op
    }

    override fun onLongClick(clicked: View): Boolean {
        // no-op
        return true
    }
}
