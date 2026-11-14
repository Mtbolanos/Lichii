// SPDX-License-Identifier: GPL-2.0-or-later

package io.github.zortos293.lichii.viewholder

import androidx.recyclerview.widget.RecyclerView
import androidx.viewbinding.ViewBinding
import io.github.zortos293.lichii.adapters.AbstractDiffAdapter
import io.github.zortos293.lichii.adapters.AbstractListAdapter

/**
 * [RecyclerView.ViewHolder] meant to work together with a [AbstractDiffAdapter] or a
 * [AbstractListAdapter] so we can run [bind] on each list item without needing a manual hookup.
 */
abstract class AbstractViewHolder<Model>(binding: ViewBinding) :
    RecyclerView.ViewHolder(binding.root) {
    abstract fun bind(model: Model)
}
