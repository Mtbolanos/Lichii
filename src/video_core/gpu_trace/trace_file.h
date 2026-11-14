// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "common/common_types.h"

namespace VideoCommon::GpuTrace {

/// A GPU trace (.lxtrace): what the emulated GPU thread consumed over a window, in the order it
/// consumed it, after a snapshot of the GPU state at the window's start. See docs/nx2/gpu-trace.md.
///
/// File: a 64-byte header, then one zstd stream of records. Each record is a RecordHeader followed
/// by `size` payload bytes.
inline constexpr std::array<char, 8> Magic{'L', 'X', 'G', 'T', 'R', 'A', 'C', 'E'};
inline constexpr u32 FormatVersion = 1;
inline constexpr u64 PageBits = 12;
inline constexpr u64 PageSize = u64{1} << PageBits;

struct FileHeader {
    std::array<char, 8> magic = Magic;
    u32 version = FormatVersion;
    u32 header_size = sizeof(FileHeader);
    std::array<u8, 48> reserved{};
};
static_assert(sizeof(FileHeader) == 64);

enum class RecordType : u32 {
    /// JSON: build, program id, settings, switches, recording window.
    Config = 1,
    /// AddressSpaceRecord: a GPU address space (GMMU) with its page geometry.
    AddressSpace = 2,
    /// GmmuMapRecord.
    GmmuMap = 3,
    /// GmmuUnmapRecord.
    GmmuUnmap = 4,
    /// SmmuMapRecord: device pages [device, +size) show physical memory [physical, +size).
    SmmuMap = 5,
    /// SmmuUnmapRecord.
    SmmuUnmap = 6,
    /// ChannelRecord.
    Channel = 7,
    /// StateRecord header followed by the state bytes (state.h).
    State = 8,
    /// BlobRecord header followed by the bytes: content stored once and named by its hash.
    Blob = 9,
    /// MemoryRecord: physical memory [physical, +size) holds a blob. Before SnapshotEnd it is the
    /// start content, after it a guest CPU write the GPU sees from this point on.
    Memory = 10,
    /// No payload: the snapshot is complete, the command stream follows.
    SnapshotEnd = 11,
    /// SubmitRecord: a command list pushed to a channel.
    Submit = 12,
    /// PresentRecord: a composite request (the layers and fences of GPU::RequestComposite).
    Present = 13,
    /// ResumeRecord: a channel suspended on a semaphore acquire moved again at this point.
    Resume = 14,
    /// Empty payload: the last record.
    End = 15,
};

struct RecordHeader {
    RecordType type;
    u32 reserved;
    u64 size;
};
static_assert(sizeof(RecordHeader) == 16);

using Hash = std::array<u64, 2>;

[[nodiscard]] Hash HashBytes(std::span<const u8> bytes);

struct AddressSpaceRecord {
    u32 id;
    u32 address_space_bits;
    u64 split_address;
    u32 big_page_bits;
    u32 page_bits;
};

struct GmmuMapRecord {
    u32 address_space;
    u32 kind;
    u64 gpu_address;
    u64 device_address;
    u64 size;
    u8 big_pages;
    u8 sparse;
    std::array<u8, 6> padding{};
};

struct GmmuUnmapRecord {
    u32 address_space;
    u32 padding{};
    u64 gpu_address;
    u64 size;
};

struct SmmuMapRecord {
    u64 device_address;
    u64 physical_address;
    u64 size;
};

struct SmmuUnmapRecord {
    u64 device_address;
    u64 size;
};

struct ChannelRecord {
    s32 bind_id;
    u32 address_space;
    u64 program_id;
};

struct StateRecord {
    u32 kind;
    s32 channel;
};

struct BlobRecord {
    Hash hash;
};

struct MemoryRecord {
    u64 physical_address;
    u64 size;
    Hash hash;
};

/// Followed by u64 GPFIFO entries[entry_count], u32 prefetch words[prefetch_count], u32
/// submitted_offsets[offset_count], u32 submitted words[word_count] and u32
/// deferred_entries[deferred_count] (entries whose words weren't copied at submit).
struct SubmitRecord {
    s32 channel;
    u32 entry_count;
    u32 prefetch_count;
    u32 offset_count;
    u64 word_count;
    u32 deferred_count;
    u32 padding{};
};

struct PresentLayer {
    u64 address;
    u32 offset;
    u32 width;
    u32 height;
    u32 stride;
    u32 pixel_format;
    u32 transform_flags;
    s32 crop_left;
    s32 crop_top;
    s32 crop_right;
    s32 crop_bottom;
    u32 blending;
    u32 native_tiling;
    u32 layer_stack_mask;
    u32 color_space;
    u32 linear_unit_luminance;
    u32 padding{};
};

struct PresentFence {
    s32 id;
    u32 value;
};

/// Followed by PresentLayer[layer_count] and PresentFence[fence_count].
struct PresentRecord {
    u64 time_ns;
    u32 layer_count;
    u32 fence_count;
};

struct ResumeRecord {
    s32 channel;
    u32 padding{};
};

/// Writes a trace. Not thread-safe: the recorder calls it from one thread at a time.
class Writer {
public:
    explicit Writer(const std::filesystem::path& path, int compression_level = 3);
    ~Writer();

    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;

    [[nodiscard]] bool IsOpen() const;

    void Write(RecordType type, std::span<const u8> payload);
    void Write(RecordType type, std::span<const std::span<const u8>> parts);

    template <typename T>
    void WritePod(RecordType type, const T& record) {
        Write(type, std::span<const u8>{reinterpret_cast<const u8*>(&record), sizeof(T)});
    }

    /// Stores `bytes` as a blob unless one with its hash was written, and returns the hash.
    Hash WriteBlob(std::span<const u8> bytes);

    /// Writes a Memory record for [physical, +bytes.size()) holding `bytes`.
    void WriteMemory(u64 physical, std::span<const u8> bytes);

    /// Finishes the stream with an End record and closes the file. Returns false on a write error.
    bool Finish();

    [[nodiscard]] u64 BytesIn() const {
        return bytes_in;
    }

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    u64 bytes_in = 0;
};

struct Record {
    RecordType type{};
    std::vector<u8> payload;

    template <typename T>
    [[nodiscard]] const T& As() const {
        return *reinterpret_cast<const T*>(payload.data());
    }
};

/// Reads a trace record by record, in file order.
class Reader {
public:
    explicit Reader(const std::filesystem::path& path);
    ~Reader();

    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;

    /// Empty when the file can't be opened or isn't a trace.
    [[nodiscard]] const std::string& Error() const {
        return error;
    }

    /// The next record, or nullopt at the end of the stream or on a read error (see Error).
    [[nodiscard]] std::optional<Record> Next();

private:
    bool Fill(void* out, size_t size);

    struct Impl;
    std::unique_ptr<Impl> impl;
    std::string error;
};

/// Whether a record's payload is large enough for its fixed part.
[[nodiscard]] bool HasPayload(const Record& record, size_t size);

} // namespace VideoCommon::GpuTrace
