// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/file_sys/nup_container.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <span>
#include <string_view>
#include <unordered_map>

#include <fmt/format.h>
#include <openssl/evp.h>
#define ZSTD_STATIC_LINKING_ONLY
#include <zstd.h>

#include "common/fs/file.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/swap.h"
#include "core/file_sys/vfs/vfs_offset.h"

namespace FileSys {
namespace {

constexpr u32 NoDir = 0xFFFFFFFF;
constexpr std::size_t SuperblockSize = 0x80;
constexpr std::size_t PackageRecordSize = 0x40;
constexpr std::size_t DirRecordSize = 0x18;
constexpr std::size_t FileRecordSize = 0x50;
constexpr std::size_t TocHeaderSize = 0x30;
constexpr std::size_t TlvHeaderSize = 0x8;
constexpr u32 TocMagic = 0x01434F54; // "TOC\x01" little-endian
constexpr u8 CodecRaw = 0;
constexpr std::array<u8, 4> ZstdMagic{0x28, 0xB5, 0x2F, 0xFD};
constexpr u8 CodecZstd = 1;
constexpr u8 PackageFlagStandaloneUpdate = 0x01;
constexpr u32 MaxDirDepth = 128;

#pragma pack(push, 1)
struct Superblock {
    std::array<u8, 4> magic;
    u16_le header_size;
    u16_le header_version;
    u64_le toc_offset;
    u64_le toc_size;
    std::array<u8, 0x20> toc_sha256;
    u64_le blob_offset;
    u64_le file_size;
    u32_le package_count;
    u32_le flags;
    std::array<u8, 0x10> uuid;
    std::array<u8, 0x20> reserved;
};
static_assert(sizeof(Superblock) == SuperblockSize);

struct TocHeader {
    u32_le magic;
    u32_le package_count;
    u64_le dir_count;
    u64_le file_count;
    u64_le pool_offset;
    u64_le pool_size;
    u64_le reserved;
};
static_assert(sizeof(TocHeader) == TocHeaderSize);

struct PackageRecord {
    u64_le title_id; // 0x00
    u32_le root_dir; // 0x08
    u32_le exefs_dir; // 0x0C
    u32_le romfs_dir; // 0x10
    u32_le version; // 0x14
    u8 kind; // 0x18
    u8 flags; // 0x19
    u16_le reserved; // 0x1A
    u32_le reserved2; // 0x1C
    u64_le meta_offset; // 0x20
    u32_le meta_size; // 0x28
    std::array<u8, 20> reserved3; // 0x2C
};
static_assert(sizeof(PackageRecord) == PackageRecordSize);
static_assert(offsetof(PackageRecord, title_id) == 0x00);
static_assert(offsetof(PackageRecord, root_dir) == 0x08);
static_assert(offsetof(PackageRecord, exefs_dir) == 0x0C);
static_assert(offsetof(PackageRecord, romfs_dir) == 0x10);
static_assert(offsetof(PackageRecord, version) == 0x14);
static_assert(offsetof(PackageRecord, kind) == 0x18);
static_assert(offsetof(PackageRecord, flags) == 0x19);
static_assert(offsetof(PackageRecord, reserved) == 0x1A);
static_assert(offsetof(PackageRecord, reserved2) == 0x1C);
static_assert(offsetof(PackageRecord, meta_offset) == 0x20);
static_assert(offsetof(PackageRecord, meta_size) == 0x28);

struct DirRecord {
    u32_le name_offset;
    u32_le name_length;
    u32_le first_child;
    u32_le child_count;
    u64_le reserved;
};
static_assert(sizeof(DirRecord) == DirRecordSize);

struct FileRecord {
    u32_le name_offset; // 0x00
    u32_le name_length; // 0x04
    u64_le size; // 0x08
    u64_le extent_offset; // 0x10
    u64_le extent_size; // 0x18
    std::array<u8, 0x20> sha256; // 0x20
    u8 codec; // 0x40
    u8 codec_flags; // 0x41
    u16_le reserved2; // 0x42
    std::array<u8, 12> reserved3; // 0x44
};
static_assert(sizeof(FileRecord) == FileRecordSize);
static_assert(offsetof(FileRecord, name_offset) == 0x00);
static_assert(offsetof(FileRecord, name_length) == 0x04);
static_assert(offsetof(FileRecord, size) == 0x08);
static_assert(offsetof(FileRecord, extent_offset) == 0x10);
static_assert(offsetof(FileRecord, extent_size) == 0x18);
static_assert(offsetof(FileRecord, sha256) == 0x20);
static_assert(offsetof(FileRecord, codec) == 0x40);
static_assert(offsetof(FileRecord, codec_flags) == 0x41);
static_assert(offsetof(FileRecord, reserved2) == 0x42);

struct TlvHeader {
    u16_le type;
    u16_le reserved;
    u32_le size;
};
static_assert(sizeof(TlvHeader) == TlvHeaderSize);
#pragma pack(pop)

enum MetaType : u16 {
    MetaNpdm = 0x0001,
    MetaIcon = 0x0003,
    MetaDisplayVersion = 0x0004,
    MetaTitleName = 0x0005,
    MetaPublisher = 0x0006,
    MetaSourceDump = 0x0008,
    MetaLocalizedTitles = 0x0009,
    MetaLocalizedIcons = 0x000A,
};

bool Sha256(const u8* data, std::size_t size, std::array<u8, 0x20>& out) {
    unsigned int length = 0;
    EVP_Digest(data, size, out.data(), &length, EVP_sha256(), nullptr);
    return length == out.size();
}

/// Bytes read from the extent per decode step; a read decodes at most one step past its end.
constexpr std::size_t DecodeInputChunk = 128 * 1024;
/// How far past the decoded prefix a read may end and still decode only up to its end.
constexpr u64 NearDecode = 1024 * 1024;
/// What a zstd decoder holds while a member is partly decoded (measured: about 221 KiB).
constexpr std::size_t DecoderCharge = 256 * 1024;
/// Partly decoded spilled members that keep their decoder, and with it a window of up to
/// 128 MiB; another one restarts from its first frame when read past its prefix again.
constexpr std::size_t MaxSpillDecoders = 2;

} // namespace

/// The decoded prefix of one zstd member, grown as reads need more of it. Reads below
/// `decoded` never wait on decoding.
struct NupMember {
    ~NupMember() {
        file.Close();
        if (!path.empty()) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    }

    std::atomic<u64> decoded{};

    // Whoever sets decoding owns the decoder state below until it clears it again.
    std::mutex mutex;
    std::condition_variable progress;
    bool decoding{};
    bool failed{};

    std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)> dctx{nullptr, &ZSTD_freeDCtx};
    u64 consumed{};
    u64 produced{};
    std::size_t last_result{};
    bool started{};

    // A member up to half the cache cap, decoded in place.
    std::unique_ptr<u8[]> data;

    // A larger member, decoded into a temporary file.
    std::mutex file_mutex;
    std::filesystem::path path;
    Common::FS::IOFile file;
};

/// Members of one container, shared by all files opened from it. In-memory members are charged
/// their full size, plus DecoderCharge while partly decoded, against `cap` and evicted least
/// recently used first; spilled members last as long as the container.
struct NupDecompressCache {
    struct Entry {
        std::size_t file_index;
        std::size_t charge;
        std::shared_ptr<NupMember> member;
    };

    std::mutex mutex;
    std::list<Entry> lru;
    std::unordered_map<std::size_t, std::list<Entry>::iterator> index;
    std::unordered_map<std::size_t, std::shared_ptr<NupMember>> spills;
    // Spilled members holding a decoder, newest first.
    std::list<NupMember*> spill_decoders;
    std::size_t bytes{};
    std::size_t cap{};
    std::filesystem::path spill_dir;

    bool Spills(u64 size) const {
        return size > cap / 2;
    }

    std::shared_ptr<NupMember> Get(std::size_t file_index, u64 size) {
        std::scoped_lock lock{mutex};
        if (Spills(size)) {
            auto& member = spills[file_index];
            if (!member) {
                member = std::make_shared<NupMember>();
            }
            return member;
        }
        if (const auto it = index.find(file_index); it != index.end()) {
            lru.splice(lru.begin(), lru, it->second);
            return it->second->member;
        }
        const std::size_t charge = static_cast<std::size_t>(size) + DecoderCharge;
        while (bytes + charge > cap && !lru.empty()) {
            bytes -= lru.back().charge;
            index.erase(lru.back().file_index);
            lru.pop_back();
        }
        bytes += charge;
        lru.push_front({file_index, charge, std::make_shared<NupMember>()});
        index.emplace(file_index, lru.begin());
        return lru.front().member;
    }

    /// Registers the new decoder of a spilled member the caller decodes, and frees the decoders of
    /// idle spilled members beyond MaxSpillDecoders.
    void AddSpillDecoder(NupMember& member) {
        std::scoped_lock lock{mutex};
        spill_decoders.push_front(&member);
        auto it = spill_decoders.end();
        while (spill_decoders.size() > MaxSpillDecoders &&
               it != std::next(spill_decoders.begin())) {
            --it;
            NupMember& other = **it;
            std::unique_lock other_lock{other.mutex, std::try_to_lock};
            if (!other_lock || other.decoding) {
                continue;
            }
            other.dctx.reset();
            it = spill_decoders.erase(it);
        }
    }

    void RemoveSpillDecoder(NupMember& member) {
        std::scoped_lock lock{mutex};
        spill_decoders.remove(&member);
    }

    /// Stops charging a member's decoder once it is freed.
    void ReleaseDecoder(std::size_t file_index, const NupMember& member) {
        std::scoped_lock lock{mutex};
        const auto it = index.find(file_index);
        if (it == index.end() || it->second->member.get() != &member ||
            it->second->charge < DecoderCharge) {
            return;
        }
        it->second->charge -= DecoderCharge;
        bytes -= DecoderCharge;
    }
};

namespace {

/// A member file whose extent is zstd-compressed, decoded from its start as reads need it and
/// shared through the container's cache.
class NupVfsFile final : public VfsFile {
public:
    NupVfsFile(VirtualFile backing_, u64 extent_offset_, u64 extent_size_, u64 uncompressed_size_,
               std::string name_, std::shared_ptr<NupDecompressCache> cache_,
               std::size_t cache_index_)
        : backing(std::move(backing_)), extent_offset(extent_offset_), extent_size(extent_size_),
          uncompressed_size(uncompressed_size_), name(std::move(name_)), cache(std::move(cache_)),
          cache_index(cache_index_), spilled(cache->Spills(uncompressed_size)) {}

    std::string GetName() const override {
        return name;
    }

    std::size_t GetSize() const override {
        return static_cast<std::size_t>(uncompressed_size);
    }

    bool Resize(std::size_t) override {
        return false;
    }

    VirtualDir GetContainingDirectory() const override {
        return nullptr;
    }

    bool IsWritable() const override {
        return false;
    }

    bool IsReadable() const override {
        return true;
    }

    std::size_t Read(u8* data, std::size_t length, std::size_t offset) const override {
        if (length == 0 || offset >= uncompressed_size) {
            return 0;
        }
        length = std::min(length, static_cast<std::size_t>(uncompressed_size) - offset);
        const u64 end = static_cast<u64>(offset) + length;

        const auto member = cache->Get(cache_index, uncompressed_size);
        if (const u64 decoded = member->decoded.load(std::memory_order_acquire); decoded < end) {
            // A read just past the decoded prefix, like the next chunk of a stream or the start of
            // a file, decodes only that far. A read further in decodes the whole member, as the
            // old reader did, so a random reader stalls once per member rather than repeatedly.
            const u64 target = end - decoded <= NearDecode ? end : uncompressed_size;
            if (!DecodeTo(*member, end, target)) {
                return 0;
            }
        }
        if (!spilled) {
            std::memcpy(data, member->data.get() + offset, length);
            return length;
        }
        std::scoped_lock lock{member->file_mutex};
        if (!member->file.Seek(static_cast<s64>(offset))) {
            return 0;
        }
        return member->file.ReadSpan(std::span{data, length});
    }

    std::size_t Write(const u8*, std::size_t, std::size_t) override {
        return 0;
    }

    bool Rename(std::string_view new_name) override {
        name = std::string(new_name);
        return true;
    }

private:
    /// Decodes the member up to `target` bytes one step at a time, waiting while another thread
    /// holds the decoder. False if fewer than `end` bytes can be decoded.
    bool DecodeTo(NupMember& member, u64 end, u64 target) const {
        std::unique_lock lock{member.mutex};
        while (member.decoded.load(std::memory_order_acquire) < target) {
            if (member.failed) {
                return member.decoded.load(std::memory_order_acquire) >= end;
            }
            if (member.decoding) {
                member.progress.wait(lock);
                continue;
            }
            member.decoding = true;
            lock.unlock();
            const bool ok = DecodeStep(member);
            lock.lock();
            member.decoding = false;
            member.failed = !ok;
            member.progress.notify_all();
        }
        return true;
    }

    /// Decodes the next input chunk; the caller owns the member's decoder. The last byte is
    /// published only once the extent has ended exactly at the file size. False on failure.
    bool DecodeStep(NupMember& member) const {
        u64 decoded = member.decoded.load(std::memory_order_relaxed);
        const auto finish = [&] {
            member.dctx.reset();
            if (spilled) {
                cache->RemoveSpillDecoder(member);
            } else {
                cache->ReleaseDecoder(cache_index, member);
            }
        };
        const auto fail = [&](const char* reason) {
            LOG_ERROR(Loader, "NUP zstd decompression of {} failed at byte {}: {}", name,
                      member.produced, reason);
            finish();
            return false;
        };
        if (!member.dctx) {
            if (!member.started) {
                member.started = true;
                if (extent_size > backing->GetSize() ||
                    extent_offset > backing->GetSize() - extent_size) {
                    return fail("extent out of range");
                }
                if (spilled) {
                    std::random_device random;
                    std::error_code ec;
                    std::filesystem::create_directories(cache->spill_dir, ec);
                    member.path = cache->spill_dir / fmt::format("{:08x}{:08x}-{}.tmp", random(),
                                                                 random(), cache_index);
                    member.file.Open(member.path, Common::FS::FileAccessMode::Write);
                    member.file.Close();
                    member.file.Open(member.path, Common::FS::FileAccessMode::ReadWrite);
                    if (!member.file.IsOpen()) {
                        return fail("cannot create the temporary file");
                    }
                } else {
                    member.data = std::make_unique_for_overwrite<u8[]>(
                        static_cast<std::size_t>(uncompressed_size));
                }
            }
            // A spilled member whose decoder was freed starts over; output already in its
            // temporary file is decoded again but not rewritten.
            member.dctx.reset(ZSTD_createDCtx());
            if (!member.dctx) {
                return fail("cannot create a decoder");
            }
            member.consumed = 0;
            member.produced = 0;
            if (spilled) {
                cache->AddSpillDecoder(member);
            } else {
                // Decoding straight into the member skips zstd's window buffer and its copy.
                ZSTD_DCtx_setParameter(member.dctx.get(), ZSTD_d_stableOutBuffer, 1);
                ZSTD_DCtx_setParameter(member.dctx.get(), ZSTD_d_windowLogMax, ZSTD_WINDOWLOG_MAX);
            }
        }

        thread_local std::vector<u8> input(DecodeInputChunk);
        thread_local std::vector<u8> output(ZSTD_DStreamOutSize());
        const auto chunk = static_cast<std::size_t>(
            std::min<u64>(DecodeInputChunk, extent_size - member.consumed));
        if (backing->Read(input.data(), chunk,
                          static_cast<std::size_t>(extent_offset + member.consumed)) != chunk) {
            return fail("short read of the compressed data");
        }
        member.consumed += chunk;
        ZSTD_inBuffer in{input.data(), chunk, 0};
        bool flush = false;
        while (in.pos < in.size || flush) {
            ZSTD_outBuffer out =
                spilled
                    ? ZSTD_outBuffer{output.data(), output.size(), 0}
                    : ZSTD_outBuffer{member.data.get(), static_cast<std::size_t>(uncompressed_size),
                                     static_cast<std::size_t>(member.produced)};
            const std::size_t in_before = in.pos;
            const std::size_t out_before = out.pos;
            member.last_result = ZSTD_decompressStream(member.dctx.get(), &out, &in);
            if (ZSTD_isError(member.last_result)) {
                return fail(ZSTD_getErrorName(member.last_result));
            }
            const u64 produced = out.pos - out_before;
            if (produced > uncompressed_size - member.produced) {
                return fail("more data than the file size");
            }
            const u64 from = std::max(member.produced, decoded);
            if (spilled && member.produced + produced > from) {
                const auto count = static_cast<std::size_t>(member.produced + produced - from);
                std::scoped_lock file_lock{member.file_mutex};
                if (!member.file.Seek(static_cast<s64>(from)) ||
                    member.file.WriteSpan(std::span<const u8>{
                        output.data() + (from - member.produced), count}) != count) {
                    return fail("cannot write the temporary file");
                }
            }
            if (produced == 0 && in.pos == in_before) {
                if (in.pos < in.size) {
                    return fail("more data than the file size");
                }
                break;
            }
            member.produced += produced;
            if (const u64 visible = std::min(member.produced, uncompressed_size - 1);
                visible > decoded) {
                decoded = visible;
                member.decoded.store(decoded, std::memory_order_release);
            }
            flush = spilled && out.pos == out.size && member.last_result != 0;
        }
        if (member.consumed < extent_size) {
            return true;
        }
        if (member.produced != uncompressed_size || member.last_result != 0) {
            return fail("the data does not end at the file size");
        }
        member.decoded.store(uncompressed_size, std::memory_order_release);
        finish();
        return true;
    }

    VirtualFile backing;
    u64 extent_offset;
    u64 extent_size;
    u64 uncompressed_size;
    std::string name;
    std::shared_ptr<NupDecompressCache> cache;
    std::size_t cache_index;
    bool spilled;
};

} // namespace

bool NupContainer::IsNupFile(const VirtualFile& file) {
    if (file == nullptr || file->GetSize() < SuperblockSize) {
        return false;
    }
    std::array<u8, 4> magic{};
    file->ReadObject(&magic, 0);
    return magic == HeaderMagic;
}

NupContainer::NupContainer(VirtualFile file_, std::size_t memory_cache_cap,
                           std::filesystem::path spill_dir_)
    : file(std::move(file_)), cache(std::make_shared<NupDecompressCache>()) {
    cache->cap = memory_cache_cap;
    cache->spill_dir = spill_dir_.empty()
                           ? Common::FS::GetEdenPath(Common::FS::EdenPath::CacheDir) / "nup"
                           : std::move(spill_dir_);
    if (file == nullptr) {
        return;
    }
    if (Parse()) {
        status = Loader::ResultStatus::Success;
    } else if (status == Loader::ResultStatus::ErrorNullFile) {
        status = Loader::ResultStatus::ErrorNUPBadFormat;
    }
}

NupContainer::~NupContainer() = default;

bool NupContainer::Parse() {
    if (file->GetSize() < SuperblockSize) {
        status = Loader::ResultStatus::ErrorNUPBadFormat;
        return false;
    }
    Superblock super{};
    if (file->ReadObject(&super, 0) != sizeof(Superblock)) {
        status = Loader::ResultStatus::ErrorNUPBadFormat;
        return false;
    }
    if (super.magic != HeaderMagic || super.header_size < SuperblockSize || super.flags != 0) {
        status = Loader::ResultStatus::ErrorNUPBadFormat;
        return false;
    }
    toc_offset = super.toc_offset;
    toc_size = super.toc_size;
    blob_offset = super.blob_offset;
    file_size = super.file_size;
    if (file_size != file->GetSize() || toc_offset < SuperblockSize || blob_offset > file_size ||
        toc_offset > blob_offset || toc_size > blob_offset - toc_offset ||
        blob_offset % 0x10 != 0) {
        status = Loader::ResultStatus::ErrorNUPBadFormat;
        return false;
    }

    const auto toc = file->ReadBytes(static_cast<std::size_t>(toc_size),
                                     static_cast<std::size_t>(toc_offset));
    if (toc.size() != toc_size) {
        status = Loader::ResultStatus::ErrorNUPBadFormat;
        return false;
    }
    std::array<u8, 0x20> toc_hash{};
    if (!Sha256(toc.data(), toc.size(), toc_hash) || toc_hash != super.toc_sha256) {
        LOG_ERROR(Loader, "NUP TOC SHA-256 mismatch, the file is corrupt or was modified");
        status = Loader::ResultStatus::ErrorNUPBadFormat;
        return false;
    }

    TocHeader header{};
    if (toc.size() < sizeof(header)) {
        status = Loader::ResultStatus::ErrorNUPBadFormat;
        return false;
    }
    std::memcpy(&header, toc.data(), sizeof(header));
    if (header.magic != TocMagic || header.package_count == 0 ||
        static_cast<u32>(header.package_count) != super.package_count) {
        status = Loader::ResultStatus::ErrorNUPBadFormat;
        return false;
    }

    // The counts come from the file, so bound them by the TOC size before multiplying.
    if (header.dir_count > toc.size() / DirRecordSize ||
        header.file_count > toc.size() / FileRecordSize) {
        status = Loader::ResultStatus::ErrorNUPBadFormat;
        return false;
    }
    const u64 needed = TocHeaderSize + static_cast<u64>(header.package_count) * PackageRecordSize +
                       header.dir_count * DirRecordSize + header.file_count * FileRecordSize;
    if (needed > toc.size() || header.pool_offset < needed || header.pool_offset > toc.size() ||
        header.pool_size > toc.size() - header.pool_offset ||
        (header.pool_offset - needed) % sizeof(u32) != 0) {
        status = Loader::ResultStatus::ErrorNUPBadFormat;
        return false;
    }
    const u64 children_offset = needed;
    const u64 children_bytes = header.pool_offset - needed;
    const u8* base = toc.data();

    std::size_t pos = TocHeaderSize;
    packages.reserve(header.package_count);
    for (u32 i = 0; i < header.package_count; ++i) {
        PackageRecord record{};
        std::memcpy(&record, base + pos, sizeof(record));
        pos += PackageRecordSize;
        if (record.kind > static_cast<u8>(PackageKind::Addon) ||
            (record.flags & ~PackageFlagStandaloneUpdate) != 0 ||
            record.root_dir >= header.dir_count ||
            (record.exefs_dir != NoDir && record.exefs_dir >= header.dir_count) ||
            (record.romfs_dir != NoDir && record.romfs_dir >= header.dir_count)) {
            status = Loader::ResultStatus::ErrorNUPBadFormat;
            return false;
        }
        PackageInfo package;
        package.title_id = record.title_id;
        package.version = record.version;
        package.kind = static_cast<PackageKind>(record.kind);
        package.standalone_update = (record.flags & PackageFlagStandaloneUpdate) != 0;
        package.root_dir = record.root_dir;
        package.exefs_dir = record.exefs_dir;
        package.romfs_dir = record.romfs_dir;
        if (record.meta_size > 0 && record.meta_offset != 0 && record.meta_offset <= file_size &&
            record.meta_size <= file_size - record.meta_offset) {
            package.meta_offset = record.meta_offset;
            package.meta_size = record.meta_size;
        }
        packages.push_back(std::move(package));
    }

    dirs.resize(header.dir_count);
    for (u64 i = 0; i < header.dir_count; ++i) {
        DirRecord record{};
        std::memcpy(&record, base + pos, sizeof(record));
        pos += DirRecordSize;
        if (static_cast<u64>(record.first_child) + record.child_count >
                children_bytes / sizeof(u32) ||
            static_cast<u64>(record.name_offset) + record.name_length > header.pool_size) {
            status = Loader::ResultStatus::ErrorNUPBadFormat;
            return false;
        }
        dirs[i].name = std::string(reinterpret_cast<const char*>(base) + header.pool_offset +
                                       record.name_offset,
                                   record.name_length);
        dirs[i].first_child = record.first_child;
        dirs[i].child_count = record.child_count;
    }

    files.resize(header.file_count);
    for (u64 i = 0; i < header.file_count; ++i) {
        FileRecord record{};
        std::memcpy(&record, base + pos, sizeof(record));
        pos += FileRecordSize;
        if (record.codec > CodecZstd || record.codec_flags != 0 ||
            (record.size == 0 && (record.extent_offset != 0 || record.extent_size != 0)) ||
            (record.size != 0 &&
             (record.extent_size == 0 || record.extent_offset < blob_offset ||
              record.extent_offset % 0x10 != 0 || record.extent_offset > file_size ||
              record.extent_size > file_size - record.extent_offset)) ||
            static_cast<u64>(record.name_offset) + record.name_length > header.pool_size) {
            status = Loader::ResultStatus::ErrorNUPBadFormat;
            return false;
        }
        files[i].name = std::string(reinterpret_cast<const char*>(base) + header.pool_offset +
                                        record.name_offset,
                                    record.name_length);
        files[i].size = record.size;
        files[i].sha256 = record.sha256;
        files[i].extent_offset = record.extent_offset;
        files[i].extent_size = record.extent_size;
        files[i].codec = record.codec;
    }

    children.resize(children_bytes / sizeof(u32));
    std::memcpy(children.data(), base + children_offset, children_bytes);
    for (const u32 child : children) {
        if (child >= header.dir_count + header.file_count) {
            status = Loader::ResultStatus::ErrorNUPBadFormat;
            return false;
        }
    }

    if (!ValidateDirTree()) {
        status = Loader::ResultStatus::ErrorNUPBadFormat;
        return false;
    }

    for (auto& package : packages) {
        if (!ParseMeta(package)) {
            status = Loader::ResultStatus::ErrorNUPBadFormat;
            return false;
        }
    }
    return true;
}

bool NupContainer::ValidateDirTree() const {
    enum : u8 { Unvisited, InProgress, Done };
    struct Frame {
        u32 dir;
        u32 next;
    };
    std::vector<u8> state(dirs.size(), Unvisited);
    std::vector<u32> height(dirs.size(), 0);
    std::vector<Frame> stack;
    for (u32 start = 0; start < dirs.size(); ++start) {
        if (state[start] != Unvisited) {
            continue;
        }
        state[start] = InProgress;
        stack.push_back({start, 0});
        while (!stack.empty()) {
            const Frame frame = stack.back();
            const DirNode& node = dirs[frame.dir];
            if (frame.next == node.child_count) {
                state[frame.dir] = Done;
                stack.pop_back();
                if (!stack.empty()) {
                    u32& parent_height = height[stack.back().dir];
                    parent_height = (std::max)(parent_height, height[frame.dir] + 1);
                }
                continue;
            }
            ++stack.back().next;
            const u32 child = children[static_cast<std::size_t>(node.first_child) + frame.next];
            if (child >= dirs.size()) {
                continue;
            }
            if (state[child] == InProgress) {
                LOG_ERROR(Loader, "NUP directory tree contains a cycle");
                return false;
            }
            if (stack.size() + height[child] + 1 > MaxDirDepth) {
                LOG_ERROR(Loader, "NUP directory tree is nested too deeply");
                return false;
            }
            if (state[child] == Done) {
                height[frame.dir] = (std::max)(height[frame.dir], height[child] + 1);
            } else {
                state[child] = InProgress;
                stack.push_back({child, 0});
            }
        }
    }
    return true;
}

bool NupContainer::ParseMeta(PackageInfo& package) {
    if (package.meta_offset == 0) {
        return true;
    }
    const auto block = file->ReadBytes(static_cast<std::size_t>(package.meta_size),
                                       static_cast<std::size_t>(package.meta_offset));
    if (block.size() != package.meta_size) {
        return false;
    }
    std::size_t pos = 0;
    while (pos + TlvHeaderSize <= block.size()) {
        TlvHeader header{};
        std::memcpy(&header, block.data() + pos, sizeof(header));
        pos += TlvHeaderSize;
        if (pos + header.size > block.size()) {
            return false;
        }
        const u8* payload = block.data() + pos;
        pos += header.size;
        switch (header.type) {
        case MetaNpdm:
            package.npdm.assign(payload, payload + header.size);
            break;
        case MetaIcon:
            package.icon.assign(payload, payload + header.size);
            break;
        case MetaTitleName:
            package.title_name = std::string(reinterpret_cast<const char*>(payload), header.size);
            break;
        case MetaPublisher:
            package.publisher = std::string(reinterpret_cast<const char*>(payload), header.size);
            break;
        case MetaSourceDump:
            package.source_dump = std::string(reinterpret_cast<const char*>(payload), header.size);
            break;
        case MetaDisplayVersion:
            package.display_version =
                std::string(reinterpret_cast<const char*>(payload), header.size);
            break;
        case MetaLocalizedTitles:
        case MetaLocalizedIcons: {
            const bool is_title = header.type == MetaLocalizedTitles;
            std::size_t record_pos = 0;
            while (record_pos < header.size) {
                const u8 lang_length = payload[record_pos++];
                if (record_pos + lang_length + sizeof(u32) > header.size) {
                    return false;
                }
                const std::string lang(reinterpret_cast<const char*>(payload) + record_pos,
                                       lang_length);
                record_pos += lang_length;
                u32 data_length = 0;
                std::memcpy(&data_length, payload + record_pos, sizeof(data_length));
                record_pos += sizeof(data_length);
                if (record_pos + data_length > header.size) {
                    return false;
                }
                if (is_title) {
                    package.localized_titles.emplace_back(
                        lang, std::string(reinterpret_cast<const char*>(payload) + record_pos,
                                          data_length));
                } else {
                    package.localized_icons.emplace_back(
                        lang, std::vector<u8>(payload + record_pos,
                                              payload + record_pos + data_length));
                }
                record_pos += data_length;
            }
            break;
        }
        default:
            break;
        }
    }
    return true;
}

std::optional<std::size_t> NupContainer::FindPackage(PackageKind kind,
                                                     std::optional<u64> title_id) const {
    for (std::size_t i = 0; i < packages.size(); ++i) {
        if (packages[i].kind != kind) {
            continue;
        }
        if (title_id && packages[i].title_id != *title_id) {
            continue;
        }
        return i;
    }
    return std::nullopt;
}

std::optional<std::size_t> NupContainer::FindBasePackage() const {
    return FindPackage(PackageKind::Base);
}

std::optional<std::size_t> NupContainer::FindBestUpdatePackage() const {
    std::optional<std::size_t> best;
    for (std::size_t i = 0; i < packages.size(); ++i) {
        if (packages[i].kind != PackageKind::Update) {
            continue;
        }
        if (!best || packages[i].version > packages[*best].version) {
            best = i;
        }
    }
    return best;
}

VirtualFile NupContainer::OpenFile(std::size_t file_index) {
    if (file_index >= files.size()) {
        return nullptr;
    }
    return OpenFileAs(file_index, files[file_index].name);
}

VirtualFile NupContainer::OpenFileAs(std::size_t file_index, const std::string& name) {
    const FileNode& node = files[file_index];
    if (node.size == 0) {
        return std::make_shared<VectorVfsFile>(std::vector<u8>{}, name);
    }
    if (node.codec == CodecRaw) {
        return std::make_shared<OffsetVfsFile>(
            file, static_cast<std::size_t>(node.extent_size),
            static_cast<std::size_t>(node.extent_offset), name);
    }
    return std::make_shared<NupVfsFile>(file, node.extent_offset, node.extent_size, node.size,
                                        name, cache, file_index);
}

std::optional<bool> NupContainer::HasZstdMagic(std::size_t file_index) const {
    const FileNode& node = files[file_index];
    if (node.size < ZstdMagic.size()) {
        return false;
    }
    std::array<u8, 4> head{};
    if (node.codec == CodecRaw) {
        if (file->Read(head.data(), head.size(), static_cast<std::size_t>(node.extent_offset)) !=
            head.size()) {
            return std::nullopt;
        }
        return head == ZstdMagic;
    }
    // The first block decodes the first bytes, and a block takes at most 128 KiB plus headers.
    const std::size_t compressed_size =
        static_cast<std::size_t>(std::min<u64>(node.extent_size, 132 * 1024));
    const auto compressed =
        file->ReadBytes(compressed_size, static_cast<std::size_t>(node.extent_offset));
    std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)> dctx{ZSTD_createDCtx(), &ZSTD_freeDCtx};
    if (compressed.size() != compressed_size || !dctx) {
        return std::nullopt;
    }
    ZSTD_DCtx_setParameter(dctx.get(), ZSTD_d_windowLogMax, ZSTD_WINDOWLOG_MAX);
    ZSTD_inBuffer input{compressed.data(), compressed.size(), 0};
    ZSTD_outBuffer output{head.data(), head.size(), 0};
    while (output.pos < output.size && input.pos < input.size) {
        if (ZSTD_isError(ZSTD_decompressStream(dctx.get(), &output, &input))) {
            return std::nullopt;
        }
    }
    if (output.pos < output.size) {
        return std::nullopt;
    }
    return head == ZstdMagic;
}

std::optional<std::size_t> NupContainer::FindStandIn(std::size_t damaged,
                                                     std::span<const std::size_t> siblings) const {
    // Same directory and same last name token ("Nrm.bntx.zs"), longest common name prefix,
    // then the closest size: a texture of the same kind and, often, the same resolution.
    const std::string& name = files[damaged].name;
    const std::size_t separator = name.rfind('_');
    if (separator == std::string::npos) {
        return std::nullopt;
    }
    const std::string_view suffix = std::string_view{name}.substr(separator + 1);
    std::optional<std::size_t> best;
    std::size_t best_prefix = 0;
    u64 best_gap = 0;
    for (const std::size_t candidate : siblings) {
        const std::string& other = files[candidate].name;
        if (candidate == damaged || other.size() <= suffix.size() || !other.ends_with(suffix) ||
            other[other.size() - suffix.size() - 1] != '_') {
            continue;
        }
        const auto mismatch = std::ranges::mismatch(name, other);
        const auto prefix = static_cast<std::size_t>(mismatch.in1 - name.begin());
        const u64 size = files[candidate].size;
        const u64 gap = size > files[damaged].size ? size - files[damaged].size
                                                   : files[damaged].size - size;
        if (best && (prefix < best_prefix || (prefix == best_prefix && gap >= best_gap))) {
            continue;
        }
        if (HasZstdMagic(candidate) != true) {
            continue;
        }
        best = candidate;
        best_prefix = prefix;
        best_gap = gap;
    }
    return best;
}

void NupContainer::BuildDirTree(u32 dir_index, VectorVfsDirectory& out, bool serve_stand_ins) {
    const DirNode& node = dirs[dir_index];
    const u64 end = static_cast<u64>(node.first_child) + node.child_count;
    std::vector<std::size_t> dir_files;
    std::vector<std::size_t> damaged;
    for (u64 slot = node.first_child; slot < end; ++slot) {
        const u32 child = children[slot];
        if (child >= dirs.size()) {
            const std::size_t index = child - static_cast<u32>(dirs.size());
            dir_files.push_back(index);
            // A game .zs is already compressed, so the NUP only recompresses one that shrinks,
            // which a zeroed range does: only those members need their header checked.
            const FileNode& member = files[index];
            if (serve_stand_ins && member.codec == CodecZstd && member.name.ends_with(".zs") &&
                HasZstdMagic(index) == false) {
                damaged.push_back(index);
            } else {
                out.AddFile(OpenFile(index));
            }
        } else {
            auto sub = std::make_shared<VectorVfsDirectory>(std::vector<VirtualFile>{},
                                                            std::vector<VirtualDir>{},
                                                            dirs[child].name);
            BuildDirTree(child, *sub, serve_stand_ins);
            out.AddDirectory(sub);
        }
    }
    for (const std::size_t index : damaged) {
        const std::optional<std::size_t> stand_in = FindStandIn(index, dir_files);
        if (!stand_in) {
            LOG_WARNING(Loader,
                        "NUP: {} has no zstd header, so this dump is damaged and the game cannot "
                        "load the file. Dump the game again.",
                        files[index].name);
            out.AddFile(OpenFile(index));
            continue;
        }
        LOG_WARNING(Loader,
                    "NUP: {} has no zstd header, so this dump is damaged. Serving {} in its place, "
                    "a different texture; dump the game again to get the original.",
                    files[index].name, files[*stand_in].name);
        out.AddFile(OpenFileAs(*stand_in, files[index].name));
    }
}

VirtualDir NupContainer::BuildPackageDir(u32 dir_index, bool serve_stand_ins) {
    auto out = std::make_shared<VectorVfsDirectory>(std::vector<VirtualFile>{},
                                                    std::vector<VirtualDir>{}, dirs[dir_index].name);
    BuildDirTree(dir_index, *out, serve_stand_ins);
    return out;
}

VirtualDir NupContainer::GetExeFSDir(std::size_t package_index) {
    if (package_index >= packages.size() || packages[package_index].exefs_dir == NoDir) {
        return nullptr;
    }
    if (exefs_dir_cache.size() <= package_index) {
        exefs_dir_cache.resize(packages.size());
    }
    if (exefs_dir_cache[package_index] == nullptr) {
        auto dir = BuildPackageDir(packages[package_index].exefs_dir, false);
        if (dir->GetFiles().empty() && dir->GetSubdirectories().empty()) {
            return nullptr;
        }
        exefs_dir_cache[package_index] = std::move(dir);
    }
    return exefs_dir_cache[package_index];
}

VirtualDir NupContainer::GetRomFSDir(std::size_t package_index) {
    if (package_index >= packages.size() || packages[package_index].romfs_dir == NoDir) {
        return nullptr;
    }
    if (romfs_dir_cache.size() <= package_index) {
        romfs_dir_cache.resize(packages.size());
    }
    if (romfs_dir_cache[package_index] == nullptr) {
        // Only the game sees stand-ins; extraction and other package readers get stored bytes.
        romfs_dir_cache[package_index] = BuildPackageDir(packages[package_index].romfs_dir, true);
    }
    return romfs_dir_cache[package_index];
}

VirtualDir NupContainer::GetPackageDir(std::size_t package_index) {
    if (package_index >= packages.size()) {
        return nullptr;
    }
    return BuildPackageDir(packages[package_index].root_dir, false);
}

VirtualDir NupContainer::MergeShadowDirs(const VirtualDir& base, const VirtualDir& overlay) {
    if (base == nullptr) {
        return overlay;
    }
    if (overlay == nullptr) {
        return base;
    }

    std::map<std::string, VirtualFile, std::less<>> file_map;
    for (const auto& entry : base->GetFiles()) {
        file_map.insert_or_assign(entry->GetName(), entry);
    }
    for (const auto& entry : overlay->GetFiles()) {
        file_map.insert_or_assign(entry->GetName(), entry);
    }

    std::map<std::string, VirtualDir, std::less<>> dir_map;
    for (const auto& entry : base->GetSubdirectories()) {
        dir_map.insert_or_assign(entry->GetName(), entry);
    }
    for (const auto& entry : overlay->GetSubdirectories()) {
        const auto it = dir_map.find(entry->GetName());
        if (it != dir_map.end()) {
            dir_map.insert_or_assign(entry->GetName(), MergeShadowDirs(it->second, entry));
        } else {
            dir_map.insert_or_assign(entry->GetName(), entry);
        }
    }

    std::vector<VirtualFile> merged_files;
    merged_files.reserve(file_map.size());
    for (const auto& [_, entry] : file_map) {
        merged_files.push_back(entry);
    }
    std::vector<VirtualDir> merged_dirs;
    merged_dirs.reserve(dir_map.size());
    for (const auto& [_, entry] : dir_map) {
        merged_dirs.push_back(entry);
    }
    return std::make_shared<VectorVfsDirectory>(std::move(merged_files), std::move(merged_dirs),
                                                base->GetName());
}

std::vector<u8> NupContainer::ReadExtentRaw(const FileNode& node) const {
    return file->ReadBytes(static_cast<std::size_t>(node.extent_size),
                           static_cast<std::size_t>(node.extent_offset));
}

Loader::ResultStatus NupContainer::VerifyIntegrity(
    std::function<bool(std::size_t, std::size_t)> progress_callback) const {
    const std::size_t total = files.size();
    for (std::size_t i = 0; i < total; ++i) {
        const FileNode& node = files[i];
        std::vector<u8> content;
        if (node.size == 0) {
            content = ReadExtentRaw(node);
        } else if (node.codec == CodecRaw) {
            content = ReadExtentRaw(node);
        } else {
            NupVfsFile view(file, node.extent_offset, node.extent_size, node.size, node.name,
                            cache, i);
            content = view.ReadAllBytes();
        }
        if (content.size() != node.size) {
            LOG_ERROR(Loader, "NUP integrity check failed for {}: short read", node.name);
            return Loader::ResultStatus::ErrorIntegrityVerificationFailed;
        }
        std::array<u8, 0x20> hash{};
        if (!Sha256(content.data(), content.size(), hash) || hash != node.sha256) {
            LOG_ERROR(Loader, "NUP integrity check failed for {}: SHA-256 mismatch", node.name);
            return Loader::ResultStatus::ErrorIntegrityVerificationFailed;
        }
        if (progress_callback && !progress_callback(i + 1, total)) {
            return Loader::ResultStatus::ErrorIntegrityVerificationFailed;
        }
    }
    return Loader::ResultStatus::Success;
}

} // namespace FileSys
