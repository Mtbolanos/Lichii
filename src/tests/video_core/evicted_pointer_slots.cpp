// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <catch2/catch_test_macros.hpp>

#include "video_core/evicted_pointer_slots.h"
#include "video_core/shader_environment.h"

namespace {

using VideoCommon::EvictedPointerSlots;

constexpr u64 SlotKey(u32 index, u32 offset) {
    return (u64{index} << 32) | offset;
}

/// A vertex program whose constant buffer slots all hold `value`. It has no GPU memory, so a
/// classification that reads a non-null slot's target would crash.
class SlotEnvironment final : public VideoCommon::GenericEnvironment {
public:
    explicit SlotEnvironment(u32 value_) : value{value_} {
        stage = Shader::Stage::VertexB;
        code = {0x123456789abcdef0, 0xfedcba9876543210};
        cached_lowest = 0;
        cached_highest = sizeof(u64);
    }

    u32 ReadCbufValue(u32, u32) override {
        return value;
    }
    Shader::TextureType ReadTextureType(u32) override {
        return Shader::TextureType::Color2D;
    }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return {};
    }
    bool IsTexturePixelFormatInteger(u32) override {
        return false;
    }
    u32 ReadViewportTransformState() override {
        return 1;
    }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32, u32) override {
        return std::nullopt;
    }

private:
    u32 value;
};

} // Anonymous namespace

TEST_CASE("A re-translation keeps the pointer slots of the pipeline it replaces",
          "[video_core][evicted_pointer_slots]") {
    constexpr u64 Slot = SlotKey(0, 0x690);

    SlotEnvironment fresh{0};
    REQUIRE_FALSE(fresh.IsCbufStorageBufferPointer(0, 0x690));
    REQUIRE(fresh.MappedStoragePointerKeys().empty());

    EvictedPointerSlots evicted;
    EvictedPointerSlots::Slots slots;
    slots[1] = {Slot};
    evicted.Remember(0xabc, slots);
    const auto kept = evicted.Take(0xabc);
    REQUIRE(kept.has_value());

    SlotEnvironment null_slot{0};
    null_slot.KeepStoragePointers((*kept)[1]);
    REQUIRE(null_slot.IsCbufStorageBufferPointer(0, 0x690));
    REQUIRE(null_slot.MappedStoragePointerKeys() == std::vector<u64>{Slot});

    SlotEnvironment unmapped_slot{0xdead0000};
    unmapped_slot.KeepStoragePointers((*kept)[1]);
    REQUIRE(unmapped_slot.IsCbufStorageBufferPointer(0, 0x690));
    REQUIRE(unmapped_slot.MappedStoragePointerKeys() == std::vector<u64>{Slot});
}

TEST_CASE("Evicted pointer slots are taken once and bounded",
          "[video_core][evicted_pointer_slots]") {
    EvictedPointerSlots evicted;
    evicted.Remember(1, {});
    REQUIRE(evicted.Size() == 0);
    REQUIRE_FALSE(evicted.Take(1).has_value());

    EvictedPointerSlots::Slots slots;
    slots[5] = {SlotKey(3, 0x10), SlotKey(4, 0x20)};
    evicted.Remember(2, slots);
    REQUIRE(evicted.Size() == 1);
    const auto taken = evicted.Take(2);
    REQUIRE(taken.has_value());
    REQUIRE((*taken)[5] == slots[5]);
    REQUIRE_FALSE(evicted.Take(2).has_value());
    REQUIRE(evicted.Size() == 0);

    for (u64 hash = 0; hash < EvictedPointerSlots::MaxEntries + 100; ++hash) {
        evicted.Remember(hash, slots);
    }
    REQUIRE(evicted.Size() == EvictedPointerSlots::MaxEntries);
    evicted.Remember(EvictedPointerSlots::MaxEntries + 50, slots);
    REQUIRE(evicted.Size() == EvictedPointerSlots::MaxEntries);
}
