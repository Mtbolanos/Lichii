// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <deque>
#include <optional>
#include <utility>
#include <vector>

#include "common/common_types.h"

namespace Service {

class Nx2VideoInputQueue {
public:
    struct Input {
        std::vector<u8> data;
        u64 pts;
        bool end;
    };

    static constexpr size_t ReadyWatermark = 8;
    /// The decoder refuses packets (nn::codec 3603, busy) from this many on. Well above the
    /// watermark: a client may hold a ready signal when the event stops.
    static constexpr size_t RefuseAt = ReadyWatermark * 3;

    void Push(Input input) {
        inputs.push_back(std::move(input));
    }

    /// Queues a packet from the client; false means answer busy. A refused packet is held rather
    /// than lost: Ocarina of Time sends it again after the ready event, and the copy is dropped,
    /// while Mario Tennis Fever ignores the result and moves on (its menu clip sends ~23 packets
    /// in 4 ms when it loops), so the held packet goes in ahead of the next one. A lost packet
    /// leaves VP9 decoding against missing references until the next keyframe.
    bool Submit(Input input) {
        if (held) {
            if (held->pts != input.pts || held->data != input.data) {
                inputs.push_back(std::move(*held));
            }
            held.reset();
        }
        if (!input.end && Full()) {
            held = std::move(input);
            return false;
        }
        inputs.push_back(std::move(input));
        return true;
    }

    Input Pop() {
        auto input = std::move(inputs.front());
        inputs.pop_front();
        return input;
    }

    void Clear() {
        inputs.clear();
        held.reset();
    }

    bool Empty() const {
        return inputs.empty();
    }

    size_t Size() const {
        return inputs.size();
    }

    bool BelowReadyWatermark() const {
        return inputs.size() < ReadyWatermark;
    }

    /// Whether SubmitInput refuses packets other than end of stream: RefuseAt or more queued.
    bool Full() const {
        return inputs.size() >= RefuseAt;
    }

private:
    std::deque<Input> inputs;
    std::optional<Input> held; ///< The last refused packet, until the client's next one
};

} // namespace Service
