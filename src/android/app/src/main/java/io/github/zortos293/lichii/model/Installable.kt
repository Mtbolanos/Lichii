// SPDX-License-Identifier: GPL-2.0-or-later

package io.github.zortos293.lichii.model

import androidx.annotation.StringRes

data class Installable(
    @StringRes val titleId: Int,
    @StringRes val descriptionId: Int,
    val install: (() -> Unit)? = null,
    val export: (() -> Unit)? = null
)
