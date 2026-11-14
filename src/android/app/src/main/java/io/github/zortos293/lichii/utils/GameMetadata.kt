// SPDX-License-Identifier: GPL-2.0-or-later

package io.github.zortos293.lichii.utils

object GameMetadata {
    external fun getIsValid(path: String): Boolean

    external fun getTitle(path: String): String

    external fun getProgramId(path: String): String

    external fun getDeveloper(path: String): String

    external fun getVersion(path: String, reload: Boolean): String

    external fun getIcon(path: String): ByteArray

    external fun getIsHomebrew(path: String): Boolean

    external fun resetMetadata()
}
