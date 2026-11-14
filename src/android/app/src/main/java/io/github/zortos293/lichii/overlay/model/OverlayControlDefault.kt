// SPDX-License-Identifier: GPL-2.0-or-later

package io.github.zortos293.lichii.overlay.model

import androidx.annotation.IntegerRes

data class OverlayControlDefault(
    val buttonId: String,
    @IntegerRes val landscapePositionResource: Pair<Int, Int>,
    @IntegerRes val portraitPositionResource: Pair<Int, Int>,
    @IntegerRes val foldablePositionResource: Pair<Int, Int>
)
