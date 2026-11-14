// SPDX-License-Identifier: GPL-2.0-or-later

package io.github.zortos293.lichii.ui.main

interface ThemeProvider {
    /**
     * Provides theme ID by overriding an activity's 'setTheme' method and returning that result
     */
    var themeId: Int
}
