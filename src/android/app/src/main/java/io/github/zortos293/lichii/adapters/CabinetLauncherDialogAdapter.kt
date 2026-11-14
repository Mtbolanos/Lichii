// SPDX-License-Identifier: GPL-2.0-or-later

package io.github.zortos293.lichii.adapters

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.core.content.res.ResourcesCompat
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import io.github.zortos293.lichii.HomeNavigationDirections
import io.github.zortos293.lichii.NativeLibrary
import io.github.zortos293.lichii.R
import io.github.zortos293.lichii.LichiiApplication
import io.github.zortos293.lichii.databinding.DialogListItemBinding
import io.github.zortos293.lichii.model.CabinetMode
import io.github.zortos293.lichii.adapters.CabinetLauncherDialogAdapter.CabinetModeViewHolder
import io.github.zortos293.lichii.model.AppletInfo
import io.github.zortos293.lichii.model.Game
import io.github.zortos293.lichii.viewholder.AbstractViewHolder

class CabinetLauncherDialogAdapter(val fragment: Fragment) :
    AbstractListAdapter<CabinetMode, CabinetModeViewHolder>(
        CabinetMode.values().copyOfRange(1, CabinetMode.entries.size).toList()
    ) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): CabinetModeViewHolder {
        DialogListItemBinding.inflate(LayoutInflater.from(parent.context), parent, false)
            .also { return CabinetModeViewHolder(it) }
    }

    inner class CabinetModeViewHolder(val binding: DialogListItemBinding) :
        AbstractViewHolder<CabinetMode>(binding) {
        override fun bind(model: CabinetMode) {
            binding.icon.setImageDrawable(
                ResourcesCompat.getDrawable(
                    binding.icon.context.resources,
                    model.iconId,
                    binding.icon.context.theme
                )
            )
            binding.title.setText(model.titleId)

            binding.root.setOnClickListener { onClick(model) }
        }

        private fun onClick(mode: CabinetMode) {
            val appletPath = NativeLibrary.getAppletLaunchPath(AppletInfo.Cabinet.entryId)
            NativeLibrary.setCurrentAppletId(AppletInfo.Cabinet.appletId)
            NativeLibrary.setCabinetMode(mode.id)
            val appletGame = Game(
                title = LichiiApplication.appContext.getString(R.string.cabinet_applet),
                path = appletPath
            )
            val action = HomeNavigationDirections.actionGlobalEmulationActivity(appletGame)
            fragment.findNavController().navigate(action)
        }
    }
}
