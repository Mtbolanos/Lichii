// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "input_common/helpers/joycon_protocol/subcommand_response.h"

namespace {
using Common::Input::DriverResult;
using namespace InputCommon::Joycon;

struct Packet {
    int size;
    ReportMode report;
    SubCommand command;
    u8 payload;
};

class ScriptedReader {
public:
    explicit ScriptedReader(std::vector<Packet> packets_) : packets{std::move(packets_)} {}

    int operator()(SubCommandResponse& output) {
        REQUIRE(reads < 12);
        const size_t index = reads++;
        if (index >= packets.size()) {
            return 0;
        }
        const auto& packet = packets[index];
        if (packet.size > 0) {
            SubCommandResponse reply{};
            reply.input_report.report_mode = packet.report;
            reply.sub_command = packet.command;
            reply.command_data.fill(packet.payload);
            std::memcpy(&output, &reply,
                        std::min(static_cast<size_t>(packet.size), sizeof(reply)));
        }
        return packet.size;
    }

    size_t reads{};

private:
    std::vector<Packet> packets;
};
} // namespace

TEST_CASE("Joy-Con replies must match both report and subcommand", "[joycon_reply]") {
    Packet unrelated{49, ReportMode::SUBCMD_REPLY, SubCommand::REQ_DEV_INFO, 0x11};
    SECTION("Unrelated subcommand reply") {}
    SECTION("Input report with a coincidental command byte") {
        unrelated.report = ReportMode::STANDARD_FULL_60HZ;
        unrelated.command = SubCommand::SET_REPORT_MODE;
    }
    ScriptedReader reader{{unrelated,
                           {49, ReportMode::SUBCMD_REPLY, SubCommand::SET_REPORT_MODE, 0x22}}};
    SubCommandResponse output{};
    REQUIRE(ReadSubCommandResponse(SubCommand::SET_REPORT_MODE, output, reader) ==
            DriverResult::Success);
    CHECK(reader.reads == 2);
    CHECK(output.input_report.report_mode == ReportMode::SUBCMD_REPLY);
    CHECK(output.sub_command == SubCommand::SET_REPORT_MODE);
    CHECK(output.command_data[0] == 0x22);
}

TEST_CASE("Joy-Con reads cannot reuse stale or truncated reply fields", "[joycon_reply]") {
    for (const int size : {-1, 0, 1, 14, 15, 48}) {
        CAPTURE(size);
        ScriptedReader reader{std::vector<Packet>(
            12, {size, ReportMode::SUBCMD_REPLY, SubCommand::SET_REPORT_MODE, 0x44})};
        SubCommandResponse output{};
        output.input_report.report_mode = ReportMode::SUBCMD_REPLY;
        output.sub_command = SubCommand::SET_REPORT_MODE;
        output.command_data.fill(0x77);
        CHECK(ReadSubCommandResponse(SubCommand::SET_REPORT_MODE, output, reader) ==
              DriverResult::Timeout);
        CHECK(reader.reads == 12);
        const SubCommandResponse empty{};
        CHECK(std::memcmp(&output, &empty, sizeof(output)) == 0);
    }
}

TEST_CASE("Joy-Con response reads accept standard and padded packets", "[joycon_reply]") {
    for (const int size : {49, 64}) {
        CAPTURE(size);
        ScriptedReader reader{{{size, ReportMode::SUBCMD_REPLY, SubCommand::SET_REPORT_MODE, 0x66}}};
        SubCommandResponse output{};
        REQUIRE(ReadSubCommandResponse(SubCommand::SET_REPORT_MODE, output, reader) ==
                DriverResult::Success);
        CHECK(reader.reads == 1);
        CHECK(output.command_data[0] == 0x66);
        CHECK(output.command_data[33] == 0x66);
        CHECK(output.command_data[34] == (size == 49 ? 0 : 0x66));
    }
}

TEST_CASE("Joy-Con unmatched replies exhaust a bounded read budget", "[joycon_reply]") {
    std::vector<Packet> packets(12, {49, ReportMode::SUBCMD_REPLY, SubCommand::REQ_DEV_INFO, 0x11});
    SECTION("All replies mismatch") {
        ScriptedReader reader{packets};
        SubCommandResponse output{};
        CHECK(ReadSubCommandResponse(SubCommand::SET_REPORT_MODE, output, reader) ==
              DriverResult::Timeout);
        CHECK(reader.reads == 12);
    }
    SECTION("Final read matches") {
        packets.back().command = SubCommand::SET_REPORT_MODE;
        packets.back().payload = 0x55;
        ScriptedReader reader{packets};
        SubCommandResponse output{};
        CHECK(ReadSubCommandResponse(SubCommand::SET_REPORT_MODE, output, reader) ==
              DriverResult::Success);
        CHECK(reader.reads == 12);
        CHECK(output.command_data[0] == 0x55);
    }
}
