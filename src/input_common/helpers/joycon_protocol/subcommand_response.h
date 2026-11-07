// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/input.h"
#include "input_common/helpers/joycon_protocol/joycon_types.h"

namespace InputCommon::Joycon {

template <typename Reader>
Common::Input::DriverResult ReadSubCommandResponse(SubCommand command, SubCommandResponse& output,
                                                  Reader&& read) {
    // Standard replies are 49 bytes; the response storage also accommodates padded USB reports.
    constexpr int MinimumReplySize = 0x31;
    constexpr int MaxAttempts = 12;
    output = {};
    for (int attempt = 0; attempt < MaxAttempts; ++attempt) {
        SubCommandResponse candidate{};
        const int size = read(candidate);
        if (size < MinimumReplySize ||
            candidate.input_report.report_mode != ReportMode::SUBCMD_REPLY ||
            candidate.sub_command != command) {
            continue;
        }
        output = candidate;
        return Common::Input::DriverResult::Success;
    }
    return Common::Input::DriverResult::Timeout;
}

} // namespace InputCommon::Joycon
