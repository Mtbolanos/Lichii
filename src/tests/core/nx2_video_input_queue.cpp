// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/hle/service/nx2_video_input_queue.h"

TEST_CASE("NX2 video retains packets submitted beyond the readiness watermark", "[nx2_video_input]") {
    Service::Nx2VideoInputQueue queue;
    constexpr u64 packets = 24;
    for (u64 index = 0; index < packets; ++index) {
        CAPTURE(index);
        queue.Push({{static_cast<u8>(index), 0x82, 0x49, 0x83}, index * 16667,
                    index == packets - 1});
        REQUIRE(queue.Size() == index + 1);
        REQUIRE(queue.BelowReadyWatermark() == (index + 1 < queue.ReadyWatermark));
    }
    for (u64 index = 0; index < packets; ++index) {
        const auto input = queue.Pop();
        REQUIRE(input.data == std::vector<u8>{static_cast<u8>(index), 0x82, 0x49, 0x83});
        REQUIRE(input.pts == index * 16667);
        REQUIRE(input.end == (index == packets - 1));
        REQUIRE(queue.BelowReadyWatermark() == (packets - index - 1 < queue.ReadyWatermark));
    }
    REQUIRE(queue.Empty());
}

TEST_CASE("NX2 video refuses packets only well past the readiness watermark", "[nx2_video_input]") {
    Service::Nx2VideoInputQueue queue;
    STATIC_REQUIRE(queue.RefuseAt > queue.ReadyWatermark);
    for (u64 index = 0; index < queue.RefuseAt; ++index) {
        CAPTURE(index);
        REQUIRE_FALSE(queue.Full());
        queue.Push({{static_cast<u8>(index)}, index * 16667, false});
    }
    REQUIRE(queue.Full());
    queue.Pop();
    REQUIRE_FALSE(queue.Full());
    REQUIRE_FALSE(queue.BelowReadyWatermark());
}

TEST_CASE("NX2 video queues preserve independent interleaved sessions", "[nx2_video_input]") {
    std::array<Service::Nx2VideoInputQueue, 3> queues;
    for (u64 frame = 0; frame < 32; ++frame) {
        for (u8 session = 0; session < queues.size(); ++session) {
            queues[session].Push({{session, static_cast<u8>(frame)}, frame * 17000 + session, false});
        }
    }
    for (u64 frame = 0; frame < 32; ++frame) {
        for (u8 session = 0; session < queues.size(); ++session) {
            REQUIRE_FALSE(queues[session].Empty());
            auto input = queues[session].Pop();
            REQUIRE(input.data == std::vector<u8>{session, static_cast<u8>(frame)});
            REQUIRE(input.pts == frame * 17000 + session);
        }
    }
    for (auto& queue : queues) {
        REQUIRE(queue.Empty());
        queue.Push({{0x11}, 42, true});
        queue.Clear();
        REQUIRE(queue.Empty());
        REQUIRE(queue.BelowReadyWatermark());
    }
}
