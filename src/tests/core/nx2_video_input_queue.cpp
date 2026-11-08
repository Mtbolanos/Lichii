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

namespace {
Service::Nx2VideoInputQueue FullQueue() {
    Service::Nx2VideoInputQueue queue;
    for (u64 index = 0; index < queue.RefuseAt; ++index) {
        REQUIRE(queue.Submit({{static_cast<u8>(index)}, index * 16667, false}));
    }
    return queue;
}

std::vector<u64> DrainPts(Service::Nx2VideoInputQueue& queue) {
    std::vector<u64> pts;
    while (!queue.Empty()) {
        pts.push_back(queue.Pop().pts);
    }
    return pts;
}

std::vector<u64> PtsRange(u64 count) {
    std::vector<u64> pts;
    for (u64 index = 0; index < count; ++index) {
        pts.push_back(index * 16667);
    }
    return pts;
}
} // namespace

TEST_CASE("NX2 video keeps refused packets a client never resends", "[nx2_video_input]") {
    // Mario Tennis Fever ignores busy and sends the next packet.
    auto queue = FullQueue();
    const u64 first = queue.RefuseAt;
    for (u64 index = first; index < first + 3; ++index) {
        REQUIRE_FALSE(queue.Submit({{static_cast<u8>(index)}, index * 16667, false}));
    }
    while (queue.Size() > 2) {
        queue.Pop();
    }
    REQUIRE(queue.Submit({{static_cast<u8>(first + 3)}, (first + 3) * 16667, false}));
    const auto expected = PtsRange(first + 4);
    REQUIRE(DrainPts(queue) == std::vector<u64>(expected.end() - 4, expected.end()));
}

TEST_CASE("NX2 video takes a resent refused packet once", "[nx2_video_input]") {
    // Ocarina of Time waits for the ready event and sends the refused packet again.
    auto queue = FullQueue();
    const u64 refused = queue.RefuseAt;
    const Service::Nx2VideoInputQueue::Input packet{{0xAB, 0xCD}, refused * 16667, false};
    REQUIRE_FALSE(queue.Submit(packet));
    REQUIRE_FALSE(queue.Submit(packet));
    REQUIRE(queue.Size() == queue.RefuseAt);
    queue.Pop();
    queue.Pop();
    REQUIRE(queue.Submit(packet));
    REQUIRE(queue.Submit({{0xEF}, (refused + 1) * 16667, false}));
    auto expected = PtsRange(refused + 2);
    expected.erase(expected.begin(), expected.begin() + 2);
    REQUIRE(DrainPts(queue) == expected);
}

TEST_CASE("NX2 video queues a held packet before end of stream", "[nx2_video_input]") {
    auto queue = FullQueue();
    const u64 refused = queue.RefuseAt;
    REQUIRE_FALSE(queue.Submit({{0x01}, refused * 16667, false}));
    REQUIRE(queue.Submit({{}, (refused + 1) * 16667, true}));
    REQUIRE(DrainPts(queue) == PtsRange(refused + 2));
}

TEST_CASE("NX2 video clear forgets a held packet", "[nx2_video_input]") {
    auto queue = FullQueue();
    REQUIRE_FALSE(queue.Submit({{0x01}, queue.RefuseAt * 16667, false}));
    queue.Clear();
    REQUIRE(queue.Submit({{0x02}, 0, false}));
    REQUIRE(DrainPts(queue) == std::vector<u64>{0});
}
