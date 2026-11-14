// SPDX-License-Identifier: GPL-2.0-or-later

package io.github.zortos293.lichii.model

enum class GameVerificationResult(val int: Int) {
    Success(0),
    Failed(1),
    NotImplemented(2);

    companion object {
        fun from(int: Int): GameVerificationResult =
            entries.firstOrNull { it.int == int } ?: Success
    }
}
