// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <deque>
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
    /// watermark: a client may hold a ready signal when the event stops, and some send one packet
    /// per signal and ignore the result, so those packets must still fit (a refused one is lost).
    static constexpr size_t RefuseAt = ReadyWatermark * 3;

    void Push(Input input) {
        inputs.push_back(std::move(input));
    }

    Input Pop() {
        auto input = std::move(inputs.front());
        inputs.pop_front();
        return input;
    }

    void Clear() {
        inputs.clear();
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
};

} // namespace Service
