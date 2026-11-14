// SPDX-License-Identifier: GPL-2.0-or-later

package io.github.zortos293.lichii.features.settings.model.view

import androidx.annotation.StringRes
import io.github.zortos293.lichii.features.input.NativeInput
import io.github.zortos293.lichii.features.input.model.AnalogDirection
import io.github.zortos293.lichii.features.input.model.InputType
import io.github.zortos293.lichii.features.input.model.NativeAnalog
import io.github.zortos293.lichii.utils.ParamPackage

class AnalogInputSetting(
    override val playerIndex: Int,
    val nativeAnalog: NativeAnalog,
    val analogDirection: AnalogDirection,
    @StringRes titleId: Int = 0,
    titleString: String = ""
) : InputSetting(titleId, titleString) {
    override val type = TYPE_INPUT
    override val inputType = InputType.Stick

    override fun getSelectedValue(): String {
        val params = NativeInput.getStickParam(playerIndex, nativeAnalog)
        val analog = analogToText(params, analogDirection.param)
        return getDisplayString(params, analog)
    }

    override fun setSelectedValue(param: ParamPackage) =
        NativeInput.setStickParam(playerIndex, nativeAnalog, param)
}
