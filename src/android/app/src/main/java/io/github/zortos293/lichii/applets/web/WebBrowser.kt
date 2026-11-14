// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

package io.github.zortos293.lichii.applets.web

import android.content.Intent
import android.net.Uri
import androidx.annotation.Keep
import io.github.zortos293.lichii.NativeLibrary
import io.github.zortos293.lichii.utils.Log

/**
    Should run WebBrowser as a new intent.
*/

@Keep
object WebBrowser {
    @JvmStatic
    fun openExternal(url: String) {
        val activity = NativeLibrary.sEmulationActivity.get() ?: run {
            return
        }

        activity.runOnUiThread {
            try {
                val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url)).apply {
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                }
                activity.startActivity(intent)
            } catch (e: Exception) {
                Log.error("WebBrowser failed to launch $url: ${e.message}")
            }
        }
    }
}
