// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <exception>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "video_core/pipeline_cache_files.h"

namespace VideoCommon {
namespace {

constexpr std::streamoff PipelineCacheHeaderSize = 8 + sizeof(u32);

struct RecordRange {
    std::streamoff begin;
    std::streamoff end;
};

/// Calls `on_record` with the key and extent of every record. False, with `detail` set, when
/// the header or any record does not parse.
bool ScanPipelineRecords(const std::filesystem::path& path, const std::array<char, 8>& magic,
                         u32 version, const PipelineRecordKeyReader& read_key,
                         const std::function<void(std::string&&, RecordRange)>& on_record,
                         std::string& detail) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        detail = "it cannot be opened";
        return false;
    }
    const auto end = static_cast<std::streamoff>(file.tellg());
    file.seekg(0);
    std::array<char, 8> file_magic{};
    u32 file_version{};
    file.read(file_magic.data(), file_magic.size());
    file.read(reinterpret_cast<char*>(&file_version), sizeof(file_version));
    if (!file || file_magic != magic || file_version != version) {
        detail = "its header does not match";
        return false;
    }
    std::streamoff offset = PipelineCacheHeaderSize;
    try {
        file.exceptions(std::ios::failbit | std::ios::badbit);
        while (offset < end) {
            auto key = read_key(file, end);
            const auto next = static_cast<std::streamoff>(file.tellg());
            if (!key || next <= offset || next > end) {
                detail = fmt::format("the record at offset {} does not parse", offset);
                return false;
            }
            on_record(std::move(*key), RecordRange{offset, next});
            offset = next;
        }
    } catch (const std::exception& e) {
        detail = fmt::format("the record at offset {} does not parse ({})", offset, e.what());
        return false;
    } catch (...) {
        detail = fmt::format("the record at offset {} does not parse", offset);
        return false;
    }
    return true;
}

/// Entries of a used-pipelines list. `strict`: every entry its count announces must be there.
std::optional<std::vector<std::pair<u64, u8>>> ReadUsedPipelineList(
    const std::filesystem::path& path, u32 magic, bool strict) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return std::nullopt;
    }
    const auto size = static_cast<std::streamoff>(file.tellg());
    std::vector<std::pair<u64, u8>> entries;
    if (size == 0 && !strict) {
        return entries;
    }
    file.seekg(0);
    u32 file_magic{};
    u32 count{};
    if (!file.read(reinterpret_cast<char*>(&file_magic), sizeof(file_magic)) ||
        file_magic != magic || !file.read(reinterpret_cast<char*>(&count), sizeof(count))) {
        return std::nullopt;
    }
    constexpr std::streamoff EntrySize = sizeof(u64) + sizeof(u8);
    entries.reserve(std::min<size_t>(count, static_cast<size_t>(size / EntrySize)));
    for (u32 i = 0; i < count; ++i) {
        u64 hash{};
        u8 age{};
        if (!file.read(reinterpret_cast<char*>(&hash), sizeof(hash)) ||
            !file.read(reinterpret_cast<char*>(&age), sizeof(age))) {
            if (strict) {
                return std::nullopt;
            }
            break;
        }
        entries.emplace_back(hash, age);
    }
    return entries;
}

/// Renames `temporary` over `destination`, removing it when that fails.
bool ReplaceWith(const std::filesystem::path& temporary, const std::filesystem::path& destination) {
    std::error_code error;
    std::filesystem::rename(temporary, destination, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        return false;
    }
    return true;
}

} // Anonymous namespace

PipelineCacheMergeResult MergePipelineCache(const std::filesystem::path& source,
                                            const std::filesystem::path& destination,
                                            const std::array<char, 8>& magic, u32 source_version,
                                            u32 destination_version,
                                            const PipelineRecordKeyReader& read_key,
                                            std::optional<PipelineKeySet>& known_keys) {
    using Status = PipelineCacheMergeResult::Status;
    PipelineCacheMergeResult result;
    std::error_code error;
    if (!std::filesystem::is_regular_file(source, error)) {
        result.status = Status::Missing;
        return result;
    }
    if (!PipelineCacheHeaderMatches(source, magic, source_version)) {
        result.status = Status::WrongHeader;
        return result;
    }
    const bool has_destination = std::filesystem::exists(destination, error);
    u64 destination_size = 0;
    if (!error && has_destination) {
        destination_size = std::filesystem::file_size(destination, error);
    }
    if (error) {
        result.status = Status::BadDestination;
        result.detail = error.message();
        return result;
    }
    // An empty file (a session that ended before its first pipeline) is written anew.
    const bool append = destination_size > 0;
    if (!known_keys) {
        PipelineKeySet keys;
        if (append && !ScanPipelineRecords(
                          destination, magic, destination_version, read_key,
                          [&](std::string&& key, RecordRange) { keys.insert(std::move(key)); },
                          result.detail)) {
            result.status = Status::BadDestination;
            return result;
        }
        known_keys = std::move(keys);
    }

    std::vector<RecordRange> ranges;
    PipelineKeySet added;
    PipelineKeySet known;
    u64 added_bytes = 0;
    if (!ScanPipelineRecords(
            source, magic, source_version, read_key,
            [&](std::string&& key, RecordRange range) {
                if (known_keys->contains(key)) {
                    known.insert(std::move(key));
                    return;
                }
                added.insert(std::move(key));
                ++result.added_records;
                added_bytes += static_cast<u64>(range.end - range.begin);
                if (!ranges.empty() && ranges.back().end == range.begin) {
                    ranges.back().end = range.end;
                } else {
                    ranges.push_back(range);
                }
            },
            result.detail)) {
        result.status = Status::BadSource;
        result.added_records = 0;
        return result;
    }
    result.added_pipelines = added.size();
    result.known_pipelines = known.size();
    result.status = Status::Merged;
    if (ranges.empty()) {
        return result;
    }

    const u64 merged_size =
        (append ? destination_size : static_cast<u64>(PipelineCacheHeaderSize)) + added_bytes;
    constexpr u64 SpareBytes = u64{64} << 20;
    const auto space = std::filesystem::space(destination.parent_path(), error);
    if (!error && space.available < merged_size + SpareBytes) {
        result.status = Status::NoSpace;
        result.detail = fmt::format("{} MiB needed, {} MiB free", (merged_size + SpareBytes) >> 20,
                                    space.available >> 20);
        return result;
    }

    auto temporary = destination;
    temporary += ".merging";
    const auto fail = [&](std::string detail) {
        std::filesystem::remove(temporary, error);
        result.status = Status::WriteFailed;
        result.detail = std::move(detail);
        return result;
    };
    if (append && !std::filesystem::copy_file(destination, temporary,
                                              std::filesystem::copy_options::overwrite_existing,
                                              error)) {
        return fail(fmt::format("copying it failed: {}", error.message()));
    }
    {
        std::ofstream out(temporary, std::ios::binary | (append ? std::ios::app : std::ios::trunc));
        std::ifstream in(source, std::ios::binary);
        if (!append) {
            out.write(magic.data(), magic.size());
            out.write(reinterpret_cast<const char*>(&destination_version),
                      sizeof(destination_version));
        }
        std::vector<char> buffer(size_t{1} << 20);
        for (const RecordRange& range : ranges) {
            in.seekg(range.begin);
            for (std::streamoff left = range.end - range.begin; left > 0 && in && out;) {
                const auto chunk =
                    std::min(left, static_cast<std::streamoff>(buffer.size()));
                in.read(buffer.data(), chunk);
                out.write(buffer.data(), chunk);
                left -= chunk;
            }
        }
        out.close();
        if (!in || out.fail()) {
            return fail("writing the merged copy failed");
        }
    }
    if (std::filesystem::file_size(temporary, error) != merged_size || error) {
        return fail("the merged copy has the wrong size");
    }
    if (!ReplaceWith(temporary, destination)) {
        result.status = Status::WriteFailed;
        result.detail = "replacing the cache with the merged copy failed";
        return result;
    }
    known_keys->merge(added);
    return result;
}

std::optional<size_t> MergeUsedPipelineLists(const std::filesystem::path& source,
                                             const std::filesystem::path& destination, u32 magic) {
    const auto incoming = ReadUsedPipelineList(source, magic, true);
    if (!incoming) {
        return std::nullopt;
    }
    std::error_code error;
    std::vector<std::pair<u64, u8>> entries;
    if (std::filesystem::exists(destination, error)) {
        auto current = ReadUsedPipelineList(destination, magic, false);
        if (!current) {
            return std::nullopt;
        }
        entries = std::move(*current);
    } else if (error) {
        return std::nullopt;
    }
    std::unordered_map<u64, size_t> index;
    index.reserve(entries.size() + incoming->size());
    for (size_t i = 0; i < entries.size(); ++i) {
        index.try_emplace(entries[i].first, i);
    }
    size_t added = 0;
    bool changed = false;
    for (const auto& [hash, age] : *incoming) {
        const auto [it, inserted] = index.try_emplace(hash, entries.size());
        if (inserted) {
            entries.emplace_back(hash, age);
            ++added;
            changed = true;
        } else if (age < entries[it->second].second) {
            entries[it->second].second = age;
            changed = true;
        }
    }
    if (!changed) {
        return added;
    }
    auto temporary = destination;
    temporary += ".merging";
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        const u32 count = static_cast<u32>(entries.size());
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        for (const auto& [hash, age] : entries) {
            out.write(reinterpret_cast<const char*>(&hash), sizeof(hash));
            out.write(reinterpret_cast<const char*>(&age), sizeof(age));
        }
        out.close();
        if (out.fail()) {
            std::filesystem::remove(temporary, error);
            return std::nullopt;
        }
    }
    if (!ReplaceWith(temporary, destination)) {
        return std::nullopt;
    }
    return added;
}

std::optional<PipelineCacheFileStamp> StampPipelineCacheFile(const std::filesystem::path& file) {
    std::error_code error;
    const u64 size = std::filesystem::file_size(file, error);
    if (error) {
        return std::nullopt;
    }
    const auto modified = std::filesystem::last_write_time(file, error);
    if (error) {
        return std::nullopt;
    }
    return PipelineCacheFileStamp{
        .size = size,
        .modified = static_cast<s64>(modified.time_since_epoch().count()),
    };
}

PipelineMergeLog ReadPipelineMergeLog(const std::filesystem::path& file) {
    PipelineMergeLog log;
    std::ifstream stream(file);
    std::string line;
    while (std::getline(stream, line)) {
        std::istringstream fields(line);
        PipelineCacheFileStamp stamp;
        std::string name;
        if (fields >> stamp.size >> stamp.modified && std::getline(fields >> std::ws, name) &&
            !name.empty()) {
            log.insert_or_assign(std::move(name), stamp);
        }
    }
    return log;
}

bool WritePipelineMergeLog(const std::filesystem::path& file, const PipelineMergeLog& log) {
    auto temporary = file;
    temporary += ".tmp";
    {
        std::ofstream stream(temporary, std::ios::trunc);
        for (const auto& [name, stamp] : log) {
            stream << stamp.size << ' ' << stamp.modified << ' ' << name << '\n';
        }
        stream.close();
        if (stream.fail()) {
            std::error_code error;
            std::filesystem::remove(temporary, error);
            return false;
        }
    }
    return ReplaceWith(temporary, file);
}

} // namespace VideoCommon
