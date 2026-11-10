// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/renderer_vulkan/vk_precompiled_kept.h"

#include <algorithm>
#include <cstring>
#include <istream>
#include <streambuf>
#include <string_view>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include "common/cityhash.h"
#include "common/zstd_compression.h"
#include "shader_recompiler/info_serialize.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/shader_environment.h"

namespace Vulkan::PrecompiledKept {
namespace {

// A new layout, or a Shader::Info field the serializer learns (info_serialize.cpp), needs a new
// magic: kept files written before are then prepared again. 2: the precompiled record offsets
// (the boot check reads each sampled record's key from them). 4: Shader::Info's
// storage_read_extents.
constexpr std::string_view Magic{"NX2KEPT4"};
constexpr std::string_view EndMagic{"NX2KEPTE"};
constexpr u8 EntryInfo = 1;
constexpr u8 EntrySpirv = 2;
constexpr u8 EntryRecord = 3;
constexpr u8 EntryEnd = 4;
/// A SPIR-V stage larger than this is taken for a damaged blob.
constexpr size_t MaxSpirvBytes = size_t{64} << 20;

/// An input stream over bytes in memory, for FileEnvironment::Deserialize and RecordFits.
class MemoryStreamBuf final : public std::streambuf {
public:
    MemoryStreamBuf(char* data, size_t size) {
        setg(data, data, data + size);
    }

protected:
    pos_type seekoff(off_type offset, std::ios_base::seekdir dir,
                     std::ios_base::openmode) override {
        char* const base = eback();
        const off_type from = dir == std::ios_base::beg   ? 0
                              : dir == std::ios_base::cur ? gptr() - base
                                                          : egptr() - base;
        const off_type target = from + offset;
        if (target < 0 || target > egptr() - base) {
            return pos_type(off_type(-1));
        }
        setg(base, base + target, egptr());
        return pos_type(target);
    }
    pos_type seekpos(pos_type position, std::ios_base::openmode which) override {
        return seekoff(off_type(position), std::ios_base::beg, which);
    }
};

template <typename T>
void Put(std::ofstream& file, const T& value) {
    file.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

template <typename T>
bool Get(const PositionalFile& file, u64 offset, T& value) {
    return file.ReadAt(offset, std::span(reinterpret_cast<u8*>(&value), sizeof(value)));
}

bool ValidRecord(const Record& record, size_t blob_count, size_t env_count) {
    if (record.env_count == 0 || record.env_count > record.envs.size() || record.kind > 1) {
        return false;
    }
    for (size_t i = 0; i < record.env_count; ++i) {
        if (record.envs[i] >= env_count) {
            return false;
        }
    }
    for (size_t stage = 0; stage < record.infos.size(); ++stage) {
        if ((record.infos[stage] != NoBlob && record.infos[stage] >= blob_count) ||
            (record.spirv[stage] != NoBlob && record.spirv[stage] >= blob_count)) {
            return false;
        }
    }
    return true;
}

std::span<const Record> EqualRange(const std::vector<Record>& records, u64 key_hash) {
    const auto [first, last] = std::equal_range(
        records.begin(), records.end(), key_hash, [](const auto& lhs, const auto& rhs) {
            if constexpr (std::is_same_v<std::decay_t<decltype(lhs)>, Record>) {
                return lhs.key_hash < rhs;
            } else {
                return lhs < rhs.key_hash;
            }
        });
    return std::span<const Record>(records.data() + (first - records.begin()),
                                   static_cast<size_t>(last - first));
}

} // Anonymous namespace

std::atomic<u64>& BytesRead() {
    static std::atomic<u64> bytes{};
    return bytes;
}

u64 SampleSeed() {
    std::filesystem::path exe;
#ifdef _WIN32
    std::wstring buffer(32768, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, buffer.data(),
                                            static_cast<DWORD>(buffer.size()));
    buffer.resize(length);
    exe = buffer;
#else
    std::error_code link_error;
    exe = std::filesystem::read_symlink("/proc/self/exe", link_error);
#endif
    std::error_code error;
    const u64 size = exe.empty() ? 0 : std::filesystem::file_size(exe, error);
    const s64 modified =
        exe.empty() ? 0
                    : static_cast<s64>(
                          std::filesystem::last_write_time(exe, error).time_since_epoch().count());
    const std::array<u64, 2> identity{size, static_cast<u64>(modified)};
    return Common::CityHash64(reinterpret_cast<const char*>(identity.data()), sizeof(identity));
}

std::vector<size_t> SamplePositions(size_t count, size_t wanted, u64 seed) {
    std::vector<size_t> positions;
    if (count <= wanted) {
        positions.resize(count);
        for (size_t i = 0; i < count; ++i) {
            positions[i] = i;
        }
        return positions;
    }
    positions.reserve(wanted);
    for (size_t i = 0; i < wanted; ++i) {
        // SplitMix64 of the seed and the stretch.
        u64 z = seed + 0x9E3779B97F4A7C15ULL * (i + 1);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= z >> 31;
        const size_t first = i * count / wanted;
        const size_t last = (i + 1) * count / wanted;
        positions.push_back(first + static_cast<size_t>(z % (last - first)));
    }
    return positions;
}

PositionalFile::PositionalFile(const std::filesystem::path& path) {
#ifdef _WIN32
    handle = CreateFileW(path.c_str(), GENERIC_READ,
                         FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
#else
    fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
#endif
}

PositionalFile::~PositionalFile() {
#ifdef _WIN32
    if (handle != INVALID_HANDLE_VALUE) {
        CloseHandle(handle);
    }
#else
    if (fd >= 0) {
        close(fd);
    }
#endif
}

bool PositionalFile::IsOpen() const {
#ifdef _WIN32
    return handle != INVALID_HANDLE_VALUE;
#else
    return fd >= 0;
#endif
}

bool PositionalFile::ReadAt(u64 offset, std::span<u8> out) const {
    if (!IsOpen()) {
        return false;
    }
    while (!out.empty()) {
#ifdef _WIN32
        OVERLAPPED position{};
        position.Offset = static_cast<DWORD>(offset);
        position.OffsetHigh = static_cast<DWORD>(offset >> 32);
        const DWORD wanted = static_cast<DWORD>((std::min<size_t>)(out.size(), 1u << 30));
        DWORD read = 0;
        if (!ReadFile(handle, out.data(), wanted, &read, &position) || read == 0) {
            return false;
        }
#else
        const ssize_t read = pread(fd, out.data(), out.size(), static_cast<off_t>(offset));
        if (read <= 0) {
            return false;
        }
#endif
        offset += static_cast<u64>(read);
        out = out.subspan(static_cast<size_t>(read));
    }
    return true;
}

Index::Index(const std::filesystem::path& kept, const std::filesystem::path& precompiled)
    : kept_file{kept}, precompiled_file{precompiled} {}

std::shared_ptr<const Index> Index::Load(const std::filesystem::path& kept,
                                         const std::filesystem::path& precompiled,
                                         const std::string& stamp, u32 cache_version,
                                         std::string& why) {
    std::error_code error;
    const u64 kept_size = std::filesystem::file_size(kept, error);
    if (error) {
        why = "no kept file";
        return nullptr;
    }
    const u64 precompiled_size = std::filesystem::file_size(precompiled, error);
    if (error) {
        why = "no precompiled file";
        return nullptr;
    }
    auto index = std::make_shared<Index>(kept, precompiled);
    if (!index->kept_file.IsOpen() || !index->precompiled_file.IsOpen()) {
        why = "cannot open the files";
        return nullptr;
    }
    const auto& file = index->kept_file;
    // Header and stamp.
    std::array<char, 8> magic{};
    u32 stamp_size{};
    if (kept_size < 32 || !Get(file, 0, magic) || !Get(file, 8, stamp_size) ||
        std::string_view{magic.data(), magic.size()} != Magic) {
        why = "another format";
        return nullptr;
    }
    if (stamp_size != stamp.size()) {
        why = "made for another file, driver, release or settings";
        return nullptr;
    }
    std::string file_stamp(stamp_size, '\0');
    if (!file.ReadAt(12, std::span(reinterpret_cast<u8*>(file_stamp.data()), stamp_size)) ||
        file_stamp != stamp) {
        why = "made for another file, driver, release or settings";
        return nullptr;
    }
    // Trailer and index.
    u64 index_offset{};
    std::array<char, 8> end_magic{};
    if (!Get(file, kept_size - 16, index_offset) || !Get(file, kept_size - 8, end_magic) ||
        std::string_view{end_magic.data(), end_magic.size()} != EndMagic ||
        index_offset > kept_size - 16 - 12) {
        why = "not finished";
        return nullptr;
    }
    std::array<u32, 3> counts{};
    if (!Get(file, index_offset, counts)) {
        why = "damaged";
        return nullptr;
    }
    const u64 records_bytes = (u64{counts[0]} + counts[1]) * sizeof(Record);
    const u64 blobs_bytes = u64{counts[2]} * 16;
    const u64 offsets_bytes = (u64{counts[0]} + counts[1]) * sizeof(u64);
    if (index_offset + 12 + records_bytes + blobs_bytes + offsets_bytes + 16 != kept_size) {
        why = "damaged";
        return nullptr;
    }
    index->graphics.resize(counts[0]);
    index->compute.resize(counts[1]);
    std::vector<std::array<u32, 4>> blob_entries(counts[2]);
    u64 at = index_offset + 12;
    if (!file.ReadAt(at, std::span(reinterpret_cast<u8*>(index->graphics.data()),
                                   index->graphics.size() * sizeof(Record))) ||
        !file.ReadAt(at + counts[0] * sizeof(Record),
                     std::span(reinterpret_cast<u8*>(index->compute.data()),
                               index->compute.size() * sizeof(Record))) ||
        !file.ReadAt(at + records_bytes, std::span(reinterpret_cast<u8*>(blob_entries.data()),
                                                   blob_entries.size() * 16))) {
        why = "damaged";
        return nullptr;
    }
    index->record_offsets_at = at + records_bytes + blobs_bytes;
    index->blobs.reserve(blob_entries.size());
    for (const auto& entry : blob_entries) {
        const u64 offset = u64{entry[0]} | (u64{entry[1]} << 32);
        const u32 size = entry[2];
        if (offset > index_offset || size > index_offset - offset) {
            why = "damaged";
            return nullptr;
        }
        index->blobs.emplace_back(offset, size);
    }
    // The precompiled file's environment index (tools/nx2-precompile/README.md): "NX2PREP1",
    // u32 cache version, u32 environment count, u32 record count, u32 reserved, then per
    // environment u64 offset and u64 size.
    const auto& prep = index->precompiled_file;
    std::array<u32, 4> header{};
    if (!Get(prep, 0, magic) || std::string_view{magic.data(), magic.size()} != "NX2PREP1" ||
        !Get(prep, 8, header) || header[0] != cache_version ||
        u64{header[1]} * 16 > precompiled_size) {
        why = "the precompiled file has another format or version";
        return nullptr;
    }
    index->environments.resize(header[1]);
    if (!prep.ReadAt(24, std::span(reinterpret_cast<u8*>(index->environments.data()),
                                   index->environments.size() * 16))) {
        why = "the precompiled file is damaged";
        return nullptr;
    }
    for (const auto& [offset, size] : index->environments) {
        if (offset > precompiled_size || size > precompiled_size - offset) {
            why = "the precompiled file is damaged";
            return nullptr;
        }
    }
    const auto valid = [&](const std::vector<Record>& records, u8 kind) {
        return std::ranges::all_of(records, [&](const Record& record) {
                   return record.kind == kind &&
                          ValidRecord(record, index->blobs.size(), index->environments.size());
               }) &&
               std::ranges::is_sorted(records, {}, &Record::key_hash);
    };
    if (!valid(index->graphics, 0) || !valid(index->compute, 1)) {
        why = "damaged";
        return nullptr;
    }
    return index;
}

std::span<const Record> Index::Graphics(u64 key_hash) const {
    return EqualRange(graphics, key_hash);
}

std::span<const Record> Index::Compute(u64 key_hash) const {
    return EqualRange(compute, key_hash);
}

size_t Index::MemoryBytes() const noexcept {
    return (graphics.capacity() + compute.capacity()) * sizeof(Record) +
           blobs.capacity() * sizeof(blobs[0]) + environments.capacity() * sizeof(environments[0]);
}

bool Index::ReadPrecompiled(u64 offset, std::span<u8> out) const {
    if (!precompiled_file.ReadAt(offset, out)) {
        return false;
    }
    BytesRead().fetch_add(out.size(), std::memory_order_relaxed);
    return true;
}

bool Index::RecordOffset(bool compute_record, size_t position, u64& offset) const {
    if (position >= (compute_record ? compute.size() : graphics.size())) {
        return false;
    }
    const size_t slot = compute_record ? graphics.size() + position : position;
    if (!record_offsets.empty()) {
        if (slot >= record_offsets.size()) {
            return false;
        }
        offset = record_offsets[slot];
        return true;
    }
    return Get(kept_file, record_offsets_at + slot * sizeof(u64), offset);
}

bool Index::ReadBlob(u32 blob, std::vector<u8>& bytes) const {
    if (blob >= blobs.size()) {
        return false;
    }
    const auto [offset, size] = blobs[blob];
    bytes.resize(size);
    if (!kept_file.ReadAt(offset, bytes)) {
        return false;
    }
    BytesRead().fetch_add(size, std::memory_order_relaxed);
    return true;
}

bool Index::ReadInfoBytes(u32 blob, std::vector<u8>& bytes) const {
    return ReadBlob(blob, bytes);
}

bool Index::ReadInfo(u32 blob, Shader::Info& info) const {
    std::vector<u8> bytes;
    if (!ReadBlob(blob, bytes)) {
        return false;
    }
    std::span<const u8> in{bytes};
    return Shader::DeserializeInfo(in, info) && in.empty();
}

bool Index::ReadSpirv(u32 blob, std::vector<u32>& words) const {
    std::vector<u8> compressed;
    if (!ReadBlob(blob, compressed)) {
        return false;
    }
    const auto bytes = Common::Compression::TryDecompressDataZSTD(compressed, MaxSpirvBytes);
    if (!bytes || bytes->empty() || bytes->size() % sizeof(u32) != 0) {
        return false;
    }
    words.resize(bytes->size() / sizeof(u32));
    std::memcpy(words.data(), bytes->data(), bytes->size());
    return true;
}

bool Index::ReadEnvironment(u32 env, VideoCommon::FileEnvironment& out) const {
    if (env >= environments.size()) {
        return false;
    }
    const auto [offset, size] = environments[env];
    std::vector<char> buffer(static_cast<size_t>(size));
    if (!precompiled_file.ReadAt(offset, std::span(reinterpret_cast<u8*>(buffer.data()),
                                                   buffer.size()))) {
        return false;
    }
    BytesRead().fetch_add(size, std::memory_order_relaxed);
    MemoryStreamBuf memory{buffer.data(), buffer.size()};
    std::istream stream{&memory};
    try {
        stream.exceptions(std::ios_base::failbit | std::ios_base::badbit);
        if (!VideoCommon::FileEnvironment::RecordFits(stream, size)) {
            return false;
        }
        out.Deserialize(stream);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool Writer::Scan(const std::filesystem::path& part, const std::string& stamp, u64& valid_end) {
    std::error_code error;
    const u64 size = std::filesystem::file_size(part, error);
    if (error) {
        return false;
    }
    PositionalFile in{part};
    std::array<char, 8> magic{};
    u32 stamp_size{};
    if (!Get(in, 0, magic) || !Get(in, 8, stamp_size) ||
        std::string_view{magic.data(), magic.size()} != Magic || stamp_size != stamp.size()) {
        return false;
    }
    std::string file_stamp(stamp_size, '\0');
    if (!in.ReadAt(12, std::span(reinterpret_cast<u8*>(file_stamp.data()), stamp_size)) ||
        file_stamp != stamp) {
        return false;
    }
    u64 at = 12 + stamp_size;
    valid_end = at;
    for (;;) {
        u8 type{};
        if (!Get(in, at, type)) {
            break;
        }
        if (type == EntryInfo || type == EntrySpirv) {
            u64 hash{};
            u32 bytes{};
            if (!Get(in, at + 1, hash) || !Get(in, at + 9, bytes) || at + 13 + bytes > size) {
                break;
            }
            auto& by_hash = type == EntryInfo ? infos_by_hash : spirv_by_hash;
            by_hash.try_emplace(hash, static_cast<u32>(blobs.size()));
            blobs.emplace_back(at + 13, bytes);
            if (type == EntrySpirv) {
                spirv_bytes += bytes;
            }
            at += 13 + bytes;
        } else if (type == EntryRecord) {
            u32 index{};
            Record record{};
            if (!Get(in, at + 1, index) || !Get(in, at + 5, record)) {
                break;
            }
            if (index < records.size() && ValidRecord(record, blobs.size(), ~size_t{0})) {
                records[index] = record;
                if (!present[index]) {
                    present[index] = 1;
                    ++kept;
                }
            }
            at += 5 + sizeof(Record);
        } else {
            // The end of a finished file (its index follows), or damage: continued from here.
            break;
        }
        valid_end = at;
    }
    resumed = kept;
    return true;
}

bool Writer::Open(const std::filesystem::path& part, const std::string& stamp, bool resume,
                  size_t total) {
    std::scoped_lock lock{mutex};
    if (file.is_open()) {
        file.close();
    }
    file.clear();
    records.assign(total, Record{});
    present.assign(total, 0);
    infos_by_hash.clear();
    spirv_by_hash.clear();
    blobs.clear();
    kept = resumed = 0;
    spirv_bytes = 0;
    failed = false;
    u64 valid_end{};
    if (resume && Scan(part, stamp, valid_end)) {
        std::error_code error;
        std::filesystem::resize_file(part, valid_end, error);
        if (!error) {
            file.open(part, std::ios::binary | std::ios::app);
            offset = valid_end;
            if (file) {
                return true;
            }
        }
    }
    records.assign(total, Record{});
    present.assign(total, 0);
    infos_by_hash.clear();
    spirv_by_hash.clear();
    blobs.clear();
    kept = resumed = 0;
    spirv_bytes = 0;
    if (file.is_open()) {
        file.close();
    }
    file.clear();
    file.open(part, std::ios::binary | std::ios::trunc);
    if (!file) {
        failed = true;
        return false;
    }
    file.write(Magic.data(), Magic.size());
    Put(file, static_cast<u32>(stamp.size()));
    file.write(stamp.data(), static_cast<std::streamsize>(stamp.size()));
    offset = 12 + stamp.size();
    return static_cast<bool>(file);
}

u32 Writer::KeepBlob(u8 type, u64 hash, std::span<const u8> bytes) {
    auto& by_hash = type == EntryInfo ? infos_by_hash : spirv_by_hash;
    const auto [it, added] = by_hash.try_emplace(hash, static_cast<u32>(blobs.size()));
    if (added) {
        Put(file, type);
        Put(file, hash);
        Put(file, static_cast<u32>(bytes.size()));
        file.write(reinterpret_cast<const char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
        blobs.emplace_back(offset + 13, static_cast<u32>(bytes.size()));
        offset += 13 + bytes.size();
        if (type == EntrySpirv) {
            spirv_bytes += bytes.size();
        }
    }
    return it->second;
}

void Writer::Keep(size_t index, u8 kind, u8 flags, u64 key_hash, std::span<const u32> envs,
                  const std::array<std::vector<u8>, 5>& infos,
                  const std::array<std::vector<u32>, 5>& spirv) {
    // Hashed and compressed before the lock: the workers keep in parallel.
    std::array<u64, 5> info_hash{};
    std::array<u64, 5> spirv_hash{};
    std::array<std::vector<u8>, 5> compressed;
    for (size_t stage = 0; stage < infos.size(); ++stage) {
        if (infos[stage].empty() || spirv[stage].empty()) {
            continue;
        }
        // The size goes into the hash, so two blobs share an id only with the same bytes.
        info_hash[stage] = Common::CityHash64(reinterpret_cast<const char*>(infos[stage].data()),
                                              infos[stage].size()) ^
                           infos[stage].size();
        const auto* const words = reinterpret_cast<const u8*>(spirv[stage].data());
        const size_t size = spirv[stage].size() * sizeof(u32);
        spirv_hash[stage] =
            Common::CityHash64(reinterpret_cast<const char*>(words), size) ^ size;
        compressed[stage] = Common::Compression::CompressDataZSTD(words, size, 3);
    }
    std::scoped_lock lock{mutex};
    if (failed || !file.is_open() || index >= records.size() || envs.empty() ||
        envs.size() > 5) {
        return;
    }
    Record record{};
    record.key_hash = key_hash;
    record.envs.fill(NoBlob);
    record.infos.fill(NoBlob);
    record.spirv.fill(NoBlob);
    std::ranges::copy(envs, record.envs.begin());
    record.env_count = static_cast<u8>(envs.size());
    record.kind = kind;
    record.flags = flags;
    for (size_t stage = 0; stage < infos.size(); ++stage) {
        if (compressed[stage].empty()) {
            continue;
        }
        record.infos[stage] = KeepBlob(EntryInfo, info_hash[stage], infos[stage]);
        record.spirv[stage] = KeepBlob(EntrySpirv, spirv_hash[stage], compressed[stage]);
    }
    Put(file, EntryRecord);
    Put(file, static_cast<u32>(index));
    Put(file, record);
    offset += 5 + sizeof(Record);
    records[index] = record;
    if (!present[index]) {
        present[index] = 1;
        ++kept;
    }
    if (!file) {
        failed = true;
    }
}

void Writer::Flush() {
    std::scoped_lock lock{mutex};
    if (file.is_open()) {
        file.flush();
    }
}

void Writer::SortedRecords(std::span<const u64> record_offsets, std::vector<Record>& graphics,
                           std::vector<Record>& compute, std::vector<u64>& offsets) const {
    std::vector<size_t> graphics_order;
    std::vector<size_t> compute_order;
    for (size_t index = 0; index < records.size(); ++index) {
        if (present[index]) {
            (records[index].kind == 1 ? compute_order : graphics_order).push_back(index);
        }
    }
    const auto by_key = [this](size_t index) { return records[index].key_hash; };
    std::ranges::stable_sort(graphics_order, {}, by_key);
    std::ranges::stable_sort(compute_order, {}, by_key);
    graphics.clear();
    compute.clear();
    offsets.clear();
    offsets.reserve(graphics_order.size() + compute_order.size());
    for (const size_t index : graphics_order) {
        graphics.push_back(records[index]);
        offsets.push_back(index < record_offsets.size() ? record_offsets[index] : 0);
    }
    for (const size_t index : compute_order) {
        compute.push_back(records[index]);
        offsets.push_back(index < record_offsets.size() ? record_offsets[index] : 0);
    }
}

std::shared_ptr<const Index> Writer::PartIndex(
    const std::filesystem::path& part, const std::filesystem::path& precompiled,
    std::span<const std::pair<u64, u64>> environments, std::span<const u64> record_offsets) {
    std::scoped_lock lock{mutex};
    if (record_offsets.size() != records.size()) {
        return nullptr;
    }
    auto index = std::make_shared<Index>(part, precompiled);
    if (!index->kept_file.IsOpen() || !index->precompiled_file.IsOpen()) {
        return nullptr;
    }
    std::vector<Record> graphics;
    std::vector<Record> compute;
    std::vector<u64> offsets;
    SortedRecords(record_offsets, graphics, compute, offsets);
    index->blobs = blobs;
    index->environments.assign(environments.begin(), environments.end());
    // Records a damaged log could hold are left out of the check (the finished file's index is
    // checked whole when it is loaded).
    std::vector<u64> compute_offsets;
    for (size_t i = 0; i < graphics.size() + compute.size(); ++i) {
        const bool is_compute = i >= graphics.size();
        const Record& record = is_compute ? compute[i - graphics.size()] : graphics[i];
        if (!ValidRecord(record, index->blobs.size(), index->environments.size())) {
            continue;
        }
        (is_compute ? index->compute : index->graphics).push_back(record);
        (is_compute ? compute_offsets : index->record_offsets).push_back(offsets[i]);
    }
    index->record_offsets.insert(index->record_offsets.end(), compute_offsets.begin(),
                                 compute_offsets.end());
    return index;
}

bool Writer::Finish(std::span<const u64> record_offsets) {
    std::scoped_lock lock{mutex};
    if (!file.is_open() || failed || record_offsets.size() != records.size()) {
        file.close();
        return false;
    }
    Put(file, EntryEnd);
    offset += 1;
    const u64 index_offset = offset;
    std::vector<Record> graphics;
    std::vector<Record> compute;
    std::vector<u64> offsets;
    SortedRecords(record_offsets, graphics, compute, offsets);
    Put(file, static_cast<u32>(graphics.size()));
    Put(file, static_cast<u32>(compute.size()));
    Put(file, static_cast<u32>(blobs.size()));
    for (const Record& record : graphics) {
        Put(file, record);
    }
    for (const Record& record : compute) {
        Put(file, record);
    }
    for (const auto& [blob_offset, size] : blobs) {
        Put(file, blob_offset);
        Put(file, size);
        Put(file, u32{0});
    }
    for (const u64 record_offset : offsets) {
        Put(file, record_offset);
    }
    Put(file, index_offset);
    file.write(EndMagic.data(), EndMagic.size());
    offset += 12 + (graphics.size() + compute.size()) * (sizeof(Record) + sizeof(u64)) +
              blobs.size() * 16 + 16;
    file.flush();
    const bool ok = static_cast<bool>(file);
    file.close();
    return ok;
}

void Writer::Close() {
    std::scoped_lock lock{mutex};
    if (file.is_open()) {
        file.flush();
        file.close();
    }
}

Writer::Totals Writer::GetTotals() {
    std::scoped_lock lock{mutex};
    return Totals{
        .records = kept,
        .infos = infos_by_hash.size(),
        .spirv = spirv_by_hash.size(),
        .spirv_bytes = spirv_bytes,
        .file_bytes = offset,
    };
}

} // namespace Vulkan::PrecompiledKept
