// SPDX-License-Identifier: GPL-2.0-or-later

package io.github.zortos293.lichii.model

enum class InstallResult(val int: Int) {
    Success(0),
    Overwrite(1),
    Failure(2),
    BaseInstallAttempted(3);

    companion object {
        fun from(int: Int): InstallResult = entries.firstOrNull { it.int == int } ?: Success
    }
}
