// SPDX-License-Identifier: GPL-2.0-or-later

package io.github.zortos293.lichii.model

import android.os.Parcelable
import kotlinx.parcelize.Parcelize

@Parcelize
data class License(
    val titleId: Int,
    val descriptionId: Int,
    val linkId: Int,
    val copyrightId: Int,
    val licenseId: Int
) : Parcelable
