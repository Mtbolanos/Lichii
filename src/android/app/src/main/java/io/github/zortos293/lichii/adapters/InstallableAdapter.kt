// SPDX-License-Identifier: GPL-2.0-or-later

package io.github.zortos293.lichii.adapters

import android.view.LayoutInflater
import android.view.ViewGroup
import io.github.zortos293.lichii.databinding.CardInstallableBinding
import io.github.zortos293.lichii.model.Installable
import io.github.zortos293.lichii.utils.ViewUtils.setVisible
import io.github.zortos293.lichii.viewholder.AbstractViewHolder

class InstallableAdapter(installables: List<Installable>) :
    AbstractListAdapter<Installable, InstallableAdapter.InstallableViewHolder>(installables) {
    override fun onCreateViewHolder(
        parent: ViewGroup,
        viewType: Int
    ): InstallableAdapter.InstallableViewHolder {
        CardInstallableBinding.inflate(LayoutInflater.from(parent.context), parent, false)
            .also { return InstallableViewHolder(it) }
    }

    inner class InstallableViewHolder(val binding: CardInstallableBinding) :
        AbstractViewHolder<Installable>(binding) {
        override fun bind(model: Installable) {
            binding.title.setText(model.titleId)
            binding.description.setText(model.descriptionId)

            binding.buttonInstall.setVisible(model.install != null)
            binding.buttonInstall.setOnClickListener { model.install?.invoke() }
            binding.buttonExport.setVisible(model.export != null)
            binding.buttonExport.setOnClickListener { model.export?.invoke() }
        }
    }
}
