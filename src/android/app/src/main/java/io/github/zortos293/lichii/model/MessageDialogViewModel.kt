// SPDX-License-Identifier: GPL-2.0-or-later

package io.github.zortos293.lichii.model

import androidx.lifecycle.ViewModel

class MessageDialogViewModel : ViewModel() {
    var positiveAction: (() -> Unit)? = null
    var negativeAction: (() -> Unit)? = null

    fun clear() {
        positiveAction = null
        negativeAction = null
    }
}
