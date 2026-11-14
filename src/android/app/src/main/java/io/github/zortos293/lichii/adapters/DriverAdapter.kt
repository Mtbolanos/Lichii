// SPDX-License-Identifier: GPL-2.0-or-later

package io.github.zortos293.lichii.adapters

import android.view.LayoutInflater
import android.view.ViewGroup
import io.github.zortos293.lichii.R
import io.github.zortos293.lichii.databinding.CardDriverOptionBinding
import io.github.zortos293.lichii.features.settings.model.StringSetting
import io.github.zortos293.lichii.model.Driver
import io.github.zortos293.lichii.model.DriverViewModel
import io.github.zortos293.lichii.utils.ViewUtils.marquee
import io.github.zortos293.lichii.utils.ViewUtils.setVisible
import io.github.zortos293.lichii.viewholder.AbstractViewHolder

class DriverAdapter(private val driverViewModel: DriverViewModel) :
    AbstractSingleSelectionList<Driver, DriverAdapter.DriverViewHolder>(
        driverViewModel.driverList.value
    ) {
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): DriverViewHolder {
        CardDriverOptionBinding.inflate(LayoutInflater.from(parent.context), parent, false)
            .also { return DriverViewHolder(it) }
    }

    inner class DriverViewHolder(val binding: CardDriverOptionBinding) :
        AbstractViewHolder<Driver>(binding) {
        override fun bind(model: Driver) {
            binding.apply {
                radioButton.isChecked = model.selected
                root.setOnClickListener {
                    selectItem(bindingAdapterPosition) {
                        driverViewModel.onDriverSelected(it)
                        driverViewModel.showClearButton(!StringSetting.DRIVER_PATH.global)
                    }
                }
                buttonDelete.setOnClickListener {
                    removeSelectableItem(
                        bindingAdapterPosition
                    ) { removedPosition: Int, selectedPosition: Int ->
                        driverViewModel.onDriverRemoved(removedPosition, selectedPosition)
                        driverViewModel.showClearButton(!StringSetting.DRIVER_PATH.global)
                    }
                }

                // Delay marquee by 3s
                title.marquee()
                version.marquee()
                description.marquee()
                title.text = model.title
                version.text = model.version
                description.text = model.description
                buttonDelete.setVisible(
                    model.title != binding.root.context.getString(R.string.system_gpu_driver)
                )
            }
        }
    }
}
