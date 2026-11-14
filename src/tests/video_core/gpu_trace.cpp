// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <filesystem>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/gpu_trace/sample.h"
#include "video_core/gpu_trace/trace_file.h"

namespace {
using namespace VideoCommon::GpuTrace;

std::filesystem::path TempTrace(const char* name) {
    return std::filesystem::temp_directory_path() / name;
}
} // Anonymous namespace

TEST_CASE("GPU trace records read back in order, blobs stored once", "[video_core][gpu_trace]") {
    const auto path = TempTrace("lichii-gpu-trace-roundtrip.lxtrace");
    const std::vector<u8> page(PageSize, 0x5a);
    {
        Writer writer{path};
        REQUIRE(writer.IsOpen());
        writer.WritePod(RecordType::SmmuMap, SmmuMapRecord{0x1000, 0x2000, 0x3000});
        writer.WriteMemory(0x2000, page);
        writer.WriteMemory(0x3000, page);
        writer.Write(RecordType::SnapshotEnd, std::span<const u8>{});
        REQUIRE(writer.Finish());
    }
    {
        Reader reader{path};
        REQUIRE(reader.Error().empty());
        std::vector<RecordType> types;
        while (const auto record = reader.Next()) {
            types.push_back(record->type);
            if (record->type == RecordType::Memory) {
                CHECK(record->As<MemoryRecord>().hash == HashBytes(page));
            }
            if (record->type == RecordType::Blob) {
                CHECK(record->payload.size() == sizeof(BlobRecord) + PageSize);
            }
        }
        CHECK(reader.Error().empty());
        CHECK(types == std::vector{RecordType::SmmuMap, RecordType::Blob, RecordType::Memory,
                                   RecordType::Memory, RecordType::SnapshotEnd, RecordType::End});
    }
    std::filesystem::remove(path);
}

TEST_CASE("GPU trace reader rejects a truncated file", "[video_core][gpu_trace]") {
    const auto path = TempTrace("lichii-gpu-trace-truncated.lxtrace");
    REQUIRE(WriteSampleTrace(path));
    const auto size = std::filesystem::file_size(path);
    std::filesystem::resize_file(path, size / 2);
    {
        Reader reader{path};
        REQUIRE(reader.Error().empty());
        bool ended = false;
        while (const auto record = reader.Next()) {
            ended = record->type == RecordType::End;
        }
        CHECK(!ended);
        CHECK(!reader.Error().empty());
    }
    std::filesystem::remove(path);
}

TEST_CASE("The sample GPU trace has a snapshot, submits and presents", "[video_core][gpu_trace]") {
    const auto path = TempTrace("lichii-gpu-trace-sample.lxtrace");
    REQUIRE(WriteSampleTrace(path));
    {
        Reader reader{path};
        REQUIRE(reader.Error().empty());
        size_t submits = 0;
        size_t presents = 0;
        bool snapshot_end = false;
        bool end = false;
        while (const auto record = reader.Next()) {
            switch (record->type) {
            case RecordType::SnapshotEnd:
                snapshot_end = true;
                break;
            case RecordType::Submit:
                CHECK(snapshot_end);
                ++submits;
                break;
            case RecordType::Present:
                CHECK(snapshot_end);
                ++presents;
                break;
            case RecordType::End:
                end = true;
                break;
            default:
                break;
            }
        }
        CHECK(reader.Error().empty());
        CHECK(end);
        CHECK(submits == 2);
        CHECK(presents == 4);
    }
    std::filesystem::remove(path);
}
