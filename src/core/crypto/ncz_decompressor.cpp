// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

#define ZSTD_STATIC_LINKING_ONLY
#include <zstd.h>

#include "common/common_types.h"
#include "common/logging.h"
#include "common/zstd_compression.h"
#include "core/crypto/aes_util.h"
#include "core/crypto/key_manager.h"
#include "core/crypto/ncz_cache.h"
#include "core/crypto/ncz_decompressor.h"
#include "core/file_sys/vfs/vfs.h"

namespace Core::Crypto {

namespace {

constexpr u64 HeaderSize = 0x4000;
constexpr std::size_t SectionWire = 0x40;
constexpr std::size_t BlockFixed = 0x18;
constexpr std::size_t Chunk = 0x100000;
// Upper bound on NCZ section entries (64 bytes each, so the table is at most 4 MiB). A base NCA
// has a handful of sections, but nsz stores an update's patch (BKTR) data one entry per
// subsection, which is thousands of them.
constexpr u64 MaxSectionCount = 0x10000;
constexpr std::array<u8, 8> SectionsMagic{'N', 'C', 'Z', 'S', 'E', 'C', 'T', 'N'};
constexpr std::array<u8, 8> BlockMagic{'N', 'C', 'Z', 'B', 'L', 'O', 'C', 'K'};

struct Section {
    s64 offset, size;
    u64 crypto_type;
    Key128 key, counter;
    bool Ctr() const { return crypto_type == 3 || crypto_type == 4; }
};

struct BlockHeader {
    u32 exponent;
    u64 decompressed_size;
    std::vector<u32> sizes;
};

u64 Le(const u8* p, std::size_t n) {
    u64 v = 0;
    for (std::size_t i = n; i-- > 0;) v = (v << 8) | p[i];
    return v;
}

// Mirrors nsz AESCTR.seek(): upper 8 bytes = stored counter, lower 8 = absolute NCA offset / 16.
void SetIv(Key128& iv, const Key128& base, u64 off) {
    iv = base;
    u64 blk = off >> 4;
    for (std::size_t i = 0; i < 8; ++i) { iv[15 - i] = static_cast<u8>(blk & 0xFF); blk >>= 8; }
}

// Parse header + section table + optional block header. Returns payload_start.
std::optional<std::pair<std::vector<Section>, std::optional<BlockHeader>>> ParseHeader(
    const FileSys::VirtualFile& f, u64& payload_start) {
    const u64 fsize = f->GetSize();
    if (fsize < HeaderSize + 16) return std::nullopt;

    const auto head = f->ReadBytes(16, HeaderSize);
    if (head.size() != 16 || !std::equal(head.begin(), head.begin() + 8, SectionsMagic.begin()))
        return std::nullopt;
    const u64 count = Le(head.data() + 8, 8);
    if (count == 0 || count > MaxSectionCount || count * SectionWire > fsize - HeaderSize - 16)
        return std::nullopt;

    const auto table = f->ReadBytes(count * SectionWire, HeaderSize + 16);
    if (table.size() != count * SectionWire) return std::nullopt;

    std::vector<Section> sections;
    sections.reserve(count + 1);
    for (u64 i = 0; i < count; ++i) {
        const u8* e = table.data() + i * SectionWire;
        Section s{};
        s.offset = static_cast<s64>(Le(e, 8));
        s.size = static_cast<s64>(Le(e + 8, 8));
        s.crypto_type = Le(e + 16, 8);
        std::memcpy(s.key.data(), e + 32, 16);
        std::memcpy(s.counter.data(), e + 48, 16);
        sections.push_back(s);
    }

    const u64 block_start = HeaderSize + 16 + count * SectionWire;
    const auto fixed = f->ReadBytes(BlockFixed, block_start);
    std::optional<BlockHeader> bh;
    if (fixed.size() == BlockFixed &&
        std::equal(fixed.begin(), fixed.begin() + 8, BlockMagic.begin())) {
        BlockHeader b{};
        b.exponent = fixed[11];
        const s32 n = static_cast<s32>(Le(fixed.data() + 12, 4));
        if (n <= 0 || b.exponent < 14 || b.exponent > 32 ||
            static_cast<u64>(n) * 4 > fsize - block_start - BlockFixed)
            return std::nullopt;
        b.decompressed_size = Le(fixed.data() + 16, 8);
        const auto sizes = f->ReadBytes(static_cast<u64>(n) * 4, block_start + BlockFixed);
        if (sizes.size() != static_cast<u64>(n) * 4) return std::nullopt;
        b.sizes.resize(n);
        for (s32 i = 0; i < n; ++i) b.sizes[i] = static_cast<u32>(Le(sizes.data() + i * 4, 4));
        payload_start = block_start + BlockFixed + static_cast<u64>(n) * 4;
        bh = std::move(b);
    } else {
        payload_start = block_start;
    }

    if (!sections.empty() && sections[0].offset > static_cast<s64>(HeaderSize)) {
        Section fake{};
        fake.offset = static_cast<s64>(HeaderSize);
        fake.size = sections[0].offset - static_cast<s64>(HeaderSize);
        sections.insert(sections.begin(), fake);
    }
    return std::make_pair(std::move(sections), std::move(bh));
}

// Sequential reader over independently-compressed blocks, one block cached (like nsz_rs).
class BlockReader {
public:
    BlockReader(FileSys::VirtualFile f, const BlockHeader& bh, u64 payload_start)
        : source{std::move(f)}, block_size{1ULL << bh.exponent}, sizes{bh.sizes},
          decompressed_size{bh.decompressed_size} {
        if (bh.exponent < 14 || bh.exponent > 32 || sizes.empty())
            throw std::runtime_error("bad NCZBLOCK header");
        u64 off = payload_start;
        offsets.reserve(sizes.size());
        for (u32 s : sizes) { offsets.push_back(off); off += s; }
    }

    // Fills `out` from `position`; returns bytes written.
    std::size_t ReadAt(u64 position, std::span<u8> out) {
        std::size_t done = 0;
        while (done < out.size() && position < decompressed_size) {
            const u64 id = position / block_size;
            if (id >= sizes.size()) break;
            if (cache_id != id && !Load(id)) break;
            const u64 in = position - id * block_size;
            const std::size_t n = static_cast<std::size_t>(
                std::min<u64>(out.size() - done, cache.size() - in));
            std::memcpy(out.data() + done, cache.data() + in, n);
            done += n;
            position += n;
        }
        return done;
    }

private:
    bool Load(u64 id) {
        u64 want = block_size;
        if (id + 1 == sizes.size()) {
            const u64 rem = decompressed_size % block_size;
            if (rem > 0) want = rem; // avoid the 0-size bug from the original nsz
        }
        try { cache.resize(static_cast<std::size_t>(want)); }
        catch (const std::bad_alloc&) { return false; }
        const u32 csize = sizes[id];
        if (csize < want) {
            const auto comp = source->ReadBytes(csize, offsets[id]);
            if (comp.size() != csize ||
                Common::Compression::DecompressDataZSTD(comp, cache) != want) return false;
        } else if (source->Read(cache.data(), cache.size(), offsets[id]) != cache.size()) {
            return false;
        }
        cache_id = id;
        return true;
    }

    FileSys::VirtualFile source;
    u64 block_size;
    std::vector<u64> offsets;
    std::vector<u32> sizes;
    u64 decompressed_size;
    u64 cache_id = ~0ULL;
    std::vector<u8> cache;
};

class BlockNczFile final : public FileSys::VfsFile {
public:
    BlockNczFile(std::string name_, std::vector<u8> hdr_, std::vector<Section> sections_,
                 std::unique_ptr<BlockReader> body_)
        : name{std::move(name_)}, hdr{std::move(hdr_)}, sections{std::move(sections_)},
          body{std::move(body_)} {
        nca_size = HeaderSize;
        for (const auto& s : sections)
            nca_size = std::max<u64>(nca_size, static_cast<u64>(s.offset + s.size));
        ciphers.reserve(sections.size());
        for (const auto& s : sections)
            ciphers.push_back(s.Ctr() ? std::make_unique<AESCipher<Key128>>(s.key, Mode::CTR)
                                      : nullptr);
    }

    std::string GetName() const override { return name; }
    std::size_t GetSize() const override { return static_cast<std::size_t>(nca_size); }
    bool Resize(std::size_t) override { return false; }
    FileSys::VirtualDir GetContainingDirectory() const override { return nullptr; }
    bool IsWritable() const override { return false; }
    bool IsReadable() const override { return true; }

    std::size_t Read(u8* data, std::size_t len, std::size_t off) const override {
        if (off >= nca_size) return 0;
        len = static_cast<std::size_t>(std::min<u64>(len, nca_size - off));
        const u64 end = off + len;
        u64 pos = off;
        if (pos < HeaderSize) {
            const u64 n = std::min<u64>(end, HeaderSize) - pos;
            std::memcpy(data, hdr.data() + pos, static_cast<std::size_t>(n));
            pos += n;
        }
        std::scoped_lock lock{mutex};
        for (std::size_t i = 0; i < sections.size() && pos < end; ++i) {
            const auto& s = sections[i];
            const u64 begin = std::max<u64>(static_cast<u64>(s.offset), HeaderSize);
            const u64 stop = std::max<u64>(static_cast<u64>(s.offset + s.size), begin);
            if (stop <= pos) continue;
            if (pos < begin) {
                const u64 g = std::min(end, begin);
                std::memset(data + (pos - off), 0, static_cast<std::size_t>(g - pos));
                pos = g;
                if (pos >= end) break;
            }
            const u64 pe = std::min(end, stop);
            const std::span<u8> piece(data + (pos - off), static_cast<std::size_t>(pe - pos));
            if (body->ReadAt(pos - HeaderSize, piece) != piece.size())
                return static_cast<std::size_t>(pos - off);
            if (ciphers[i]) {
                Key128 iv{};
                SetIv(iv, s.counter, pos);
                ciphers[i]->SetIV(iv);
                ciphers[i]->Transcode(piece.data(), piece.size(), piece.data(), Op::Encrypt);
            }
            pos = pe;
        }
        return len;
    }

    std::size_t Write(const u8*, std::size_t, std::size_t) override { return 0; }
    bool Rename(std::string_view) override { return false; }

private:
    const std::string name;
    const std::vector<u8> hdr;
    const std::vector<Section> sections;
    const std::unique_ptr<BlockReader> body;
    std::vector<std::unique_ptr<AESCipher<Key128>>> ciphers;
    mutable std::mutex mutex;
    u64 nca_size{};
};

// Shared by solid readers' decoders, including Zstandard's internal allocations and I/O buffers.
constexpr std::size_t SolidMemoryLimit = 512ULL * 1024 * 1024;
constexpr int SolidWindowLogMax = 27; // 128 MiB, sufficient for ordinary nsz compression.
std::atomic<std::size_t> solid_memory_used{};
// Size of the last allocation the budget refused on this thread, for the failure's log line.
thread_local std::size_t solid_refused_size{};

struct alignas(std::max_align_t) SolidAllocation {
    std::size_t size;
};

void* AllocateSolid(void*, std::size_t size) {
    if (size > SolidMemoryLimit - sizeof(SolidAllocation)) {
        return nullptr;
    }
    const auto total = size + sizeof(SolidAllocation);
    auto used = solid_memory_used.load(std::memory_order_relaxed);
    do {
        if (total > SolidMemoryLimit - used) {
            solid_refused_size = total;
            return nullptr;
        }
    } while (!solid_memory_used.compare_exchange_weak(used, used + total));
    auto* allocation = static_cast<SolidAllocation*>(std::malloc(total));
    if (!allocation) {
        solid_memory_used.fetch_sub(total);
        return nullptr;
    }
    allocation->size = total;
    return allocation + 1;
}

void FreeSolid(void*, void* address) {
    if (!address) {
        return;
    }
    auto* allocation = static_cast<SolidAllocation*>(address) - 1;
    const auto size = allocation->size;
    std::free(allocation);
    solid_memory_used.fetch_sub(size);
}

struct SolidBufferDeleter {
    void operator()(u8* address) const {
        FreeSolid(nullptr, address);
    }
};

std::optional<u64> SolidSize(const std::vector<Section>& sections) {
    u64 end = HeaderSize;
    for (const auto& s : sections) {
        if (s.offset < 0 || s.size < 0 || s.size > std::numeric_limits<s64>::max() - s.offset ||
            (s.crypto_type != 0 && s.crypto_type != 1 && !s.Ctr())) {
            return std::nullopt;
        }
        const auto section_end = static_cast<u64>(s.offset) + static_cast<u64>(s.size);
        if (s.size == 0 && static_cast<u64>(s.offset) != end) {
            return std::nullopt;
        }
        if (section_end <= HeaderSize || s.size == 0) {
            continue;
        }
        if (std::max<u64>(s.offset, HeaderSize) != end ||
            section_end > std::numeric_limits<std::size_t>::max()) {
            return std::nullopt;
        }
        end = section_end;
    }
    return end > HeaderSize ? std::optional{end} : std::nullopt;
}

bool SupportedSolidFrame(const FileSys::VirtualFile& source, u64 offset, u64 nca_size) {
    if (offset >= source->GetSize())
        return false;
    const auto bytes = source->ReadBytes(
        std::min<u64>(ZSTD_FRAMEHEADERSIZE_MAX, source->GetSize() - offset), offset);
    ZSTD_frameHeader header{};
    const auto result = ZSTD_getFrameHeader(&header, bytes.data(), bytes.size());
    return result == 0 && header.frameType == ZSTD_frame && header.dictID == 0 &&
           header.windowSize <= (1ULL << SolidWindowLogMax) &&
           (header.frameContentSize == ZSTD_CONTENTSIZE_UNKNOWN ||
            header.frameContentSize == nca_size - HeaderSize);
}

class SolidNczFile;
// Live solid readers, so one refused for budget can take the decoders of idle ones (ZOR-286).
std::mutex solid_readers_mutex;
std::vector<const SolidNczFile*> solid_readers;

// A single forward decode populates a private disk file; random/backward reads reuse it.
class SolidNczFile final : public FileSys::VfsFile {
public:
    SolidNczFile(std::string name_, FileSys::VirtualFile source_, std::vector<Section> sections_,
                 u64 payload_start_, u64 size_, std::unique_ptr<NczCache> backing)
        : name{std::move(name_)}, source{std::move(source_)}, sections{std::move(sections_)},
          payload_start{payload_start_}, nca_size{size_}, read_pos{payload_start_} {
        solid_refused_size = 0;
        const auto raw_header = source->ReadBytes(HeaderSize, 0);
        cache = backing ? std::move(backing) : std::make_unique<NczCache>();
        if (raw_header.size() != HeaderSize || !cache->IsOpen() || !cache->Write(raw_header, 0)) {
            Fail("cannot create/write private temporary cache");
        }
        std::scoped_lock registry{solid_readers_mutex};
        solid_readers.push_back(this);
    }

    ~SolidNczFile() override {
        {
            std::scoped_lock registry{solid_readers_mutex};
            std::erase(solid_readers, this);
        }
        ReleaseDecoder();
    }

    std::string GetName() const override { return name; }
    std::size_t GetSize() const override {
        return static_cast<std::size_t>(nca_size);
    }
    bool Resize(std::size_t) override { return false; }
    FileSys::VirtualDir GetContainingDirectory() const override { return nullptr; }
    bool IsWritable() const override { return false; }
    bool IsReadable() const override {
        std::scoped_lock lock{mutex};
        return !failed;
    }

    std::size_t Read(u8* out, std::size_t len, std::size_t off) const override {
        std::scoped_lock lock{mutex};
        if (failed || off >= nca_size || len == 0)
            return 0;
        len = static_cast<std::size_t>(std::min<u64>(len, nca_size - off));
        const u64 needed = off + len;
        if (needed > frontier && !DecodeTo(needed)) return 0;
        if (cache->Read(std::span{out, len}, off) != len) {
            Fail("temporary cache read failed");
            return 0;
        }
        return len;
    }

    std::size_t Write(const u8*, std::size_t, std::size_t) override { return 0; }
    bool Rename(std::string_view) override { return false; }

private:
    void ReleaseDecoder() const {
        ZSTD_freeDStream(stream);
        stream = nullptr;
        input.reset();
        output.reset();
        in = {nullptr, 0, 0};
    }

    // Holding the registry lock keeps the others alive; try_lock never waits on a decoding one.
    void ReleaseIdleReaders() const {
        std::scoped_lock registry{solid_readers_mutex};
        for (const auto* reader : solid_readers) {
            if (reader == this)
                continue;
            std::unique_lock lock{reader->mutex, std::try_to_lock};
            if (lock)
                reader->ReleaseDecoder();
        }
    }

    bool Fail(const char* reason) const {
        if (solid_refused_size != 0 && budget_retry) {
            // DecodeTo takes the memory of idle readers and resumes from the frontier.
            ReleaseDecoder();
            return false;
        }
        if (solid_refused_size != 0) {
            // Idle readers were already released, so the budget is held by decoding ones.
            LOG_ERROR(Crypto,
                      "Solid NCZ {}: {}: the decoder memory budget is used up ({} KiB refused, {} "
                      "of {} MiB held by open solid readers)",
                      name, reason, solid_refused_size >> 10, solid_memory_used.load() >> 20,
                      SolidMemoryLimit >> 20);
            solid_refused_size = 0;
        } else {
            LOG_ERROR(Crypto, "Solid NCZ {}: {}", name, reason);
        }
        failed = true;
        ReleaseDecoder();
        cache.reset();
        return false;
    }

    bool Refill() const {
        if (in.pos < in.size || read_pos == source->GetSize()) {
            return true;
        }
        if (read_pos > source->GetSize()) {
            return Fail("source size changed");
        }
        const auto want =
            static_cast<std::size_t>(std::min<u64>(Chunk, source->GetSize() - read_pos));
        if (source->Read(input.get(), want, read_pos) != want) {
            return Fail("source read failed");
        }
        read_pos += want;
        in = {input.get(), want, 0};
        return true;
    }

    bool EndFrame() const {
        if (in.pos != in.size || read_pos != source->GetSize()) {
            return Fail("trailing compressed data");
        }
        frame_ended = true;
        return true;
    }

    bool FinishFrame() const {
        while (!frame_ended) {
            if (!Refill())
                return false;
            u8 extra{};
            ZSTD_outBuffer out{&extra, 1, 0};
            const auto previous = in.pos;
            const auto result = ZSTD_decompressStream(stream, &out, &in);
            if (ZSTD_isError(result) || out.pos != 0) {
                return Fail("invalid frame or excess decoded data");
            }
            if (result == 0)
                return EndFrame();
            if (in.pos == previous)
                return Fail("truncated frame");
        }
        return true;
    }

    bool DecodeTo(u64 needed) const {
        solid_refused_size = 0;
        budget_retry = true;
        const bool decoded = Decode(needed);
        budget_retry = false;
        if (decoded || failed)
            return decoded;
        ReleaseIdleReaders();
        solid_refused_size = 0;
        return Decode(needed);
    }

    // The payload is one frame, so a new decoder restarts it and skips what the cache holds.
    bool Resume() const {
        read_pos = payload_start;
        for (u64 skipped = HeaderSize; skipped < frontier;) {
            if (!Refill())
                return false;
            const auto want = static_cast<std::size_t>(std::min<u64>(Chunk, frontier - skipped));
            ZSTD_outBuffer out{output.get(), want, 0};
            const auto previous_in = in.pos;
            const auto result = ZSTD_decompressStream(stream, &out, &in);
            if (ZSTD_isError(result))
                return Fail("invalid frame, window limit or memory budget");
            if (result == 0)
                return Fail("frame ended before declared size");
            if (previous_in == in.pos && out.pos == 0)
                return Fail("truncated frame");
            skipped += out.pos;
        }
        return true;
    }

    bool Decode(u64 needed) const {
        if (stream == nullptr) {
            input.reset(static_cast<u8*>(AllocateSolid(nullptr, Chunk)));
            output.reset(static_cast<u8*>(AllocateSolid(nullptr, Chunk)));
            const ZSTD_customMem memory{AllocateSolid, FreeSolid, nullptr};
            stream = ZSTD_createDStream_advanced(memory);
            if (!stream || !input || !output ||
                ZSTD_isError(
                    ZSTD_DCtx_setParameter(stream, ZSTD_d_windowLogMax, SolidWindowLogMax))) {
                return Fail("decoder memory budget exhausted");
            }
            if (!Resume())
                return false;
        }
        while (frontier < needed) {
            while (section_index < sections.size() &&
                   static_cast<u64>(sections[section_index].offset +
                                    sections[section_index].size) <= frontier) {
                ++section_index;
            }
            if (section_index == sections.size())
                return Fail("invalid section layout");
            const auto& section = sections[section_index];
            const auto end = static_cast<u64>(section.offset + section.size);
            const auto prefix = static_cast<std::size_t>(frontier & 15);
            const auto want =
                static_cast<std::size_t>(std::min<u64>(Chunk - prefix, end - frontier));
            std::size_t produced = 0;
            while (produced < want) {
                if (!Refill())
                    return false;
                ZSTD_outBuffer out{output.get() + prefix, want, produced};
                const auto previous_in = in.pos;
                const auto result = ZSTD_decompressStream(stream, &out, &in);
                if (ZSTD_isError(result))
                    return Fail("invalid frame, window limit or memory budget");
                if (result == 0) {
                    if (frontier + out.pos != nca_size)
                        return Fail("frame ended before declared size");
                    if (!EndFrame())
                        return false;
                }
                if (previous_in == in.pos && out.pos == produced)
                    return Fail("truncated frame");
                produced = out.pos;
            }
            if (section.Ctr()) {
                AESCipher<Key128> cipher(section.key, Mode::CTR);
                Key128 iv{};
                SetIv(iv, section.counter, frontier);
                cipher.SetIV(iv);
                std::fill_n(output.get(), prefix, 0);
                cipher.Transcode(output.get(), prefix + produced, output.get(), Op::Encrypt);
            }
            if (!cache->Write(std::span{output.get() + prefix, produced}, frontier)) {
                return Fail("temporary cache write failed (disk full or I/O error)");
            }
            frontier += produced;
            if (frontier == nca_size) {
                if (!FinishFrame())
                    return false;
                ReleaseDecoder();
            }
        }
        return true;
    }

    const std::string name;
    const FileSys::VirtualFile source;
    const std::vector<Section> sections;
    const u64 payload_start;
    const u64 nca_size;
    mutable std::mutex mutex;
    mutable std::unique_ptr<NczCache> cache;
    mutable std::unique_ptr<u8, SolidBufferDeleter> input;
    mutable std::unique_ptr<u8, SolidBufferDeleter> output;
    mutable ZSTD_DStream* stream = nullptr;
    mutable u64 read_pos;
    mutable u64 frontier = HeaderSize;
    mutable std::size_t section_index{};
    mutable bool frame_ended{};
    mutable bool failed{};
    mutable bool budget_retry{};
    mutable ZSTD_inBuffer in{nullptr, 0, 0};
};

FileSys::VirtualFile OpenNcz(const FileSys::VirtualFile& f, std::string_view name,
                             std::unique_ptr<NczCache> cache = {}) {
    if (!f) return nullptr;
    u64 ps = 0;
    auto parsed = ParseHeader(f, ps);
    if (!parsed) return nullptr;
    auto& [sections, block] = *parsed;

    if (block) {
        auto hdr = f->ReadBytes(HeaderSize, 0);
        if (hdr.size() != HeaderSize) return nullptr;
        std::unique_ptr<BlockReader> body;
        try { body = std::make_unique<BlockReader>(f, *block, ps); }
        catch (const std::exception& e) {
            LOG_ERROR(Crypto, "Failed to open NCZBLOCK: {}", e.what());
            return nullptr;
        }
        return std::make_shared<BlockNczFile>(std::string{name}, std::move(hdr),
                                              std::move(sections), std::move(body));
    }

    const auto size = SolidSize(sections);
    if (!size) {
        LOG_ERROR(Crypto, "Invalid solid NCZ section layout");
        return nullptr;
    }
    if (!SupportedSolidFrame(f, ps, *size)) {
        LOG_ERROR(Crypto, "Unsupported solid NCZ frame/header (maximum window: 128 MiB)");
        return nullptr;
    }
    auto solid = std::make_shared<SolidNczFile>(std::string{name}, f, std::move(sections), ps,
                                                *size, std::move(cache));
    return solid->IsReadable() ? solid : nullptr;
}

} // Anonymous namespace

bool IsBlockCompressedNCZ(const FileSys::VirtualFile& f) {
    if (!f) return false;
    u64 ps = 0;
    const auto p = ParseHeader(f, ps);
    return p && p->second.has_value();
}

FileSys::VirtualFile DecompressNCZ(const FileSys::VirtualFile& f, std::string_view name) {
    return OpenNcz(f, name);
}

bool IsSupportedNCZ(const FileSys::VirtualFile& f) {
    if (!f)
        return false;
    u64 payload{};
    const auto parsed = ParseHeader(f, payload);
    if (!parsed)
        return false;
    if (parsed->second)
        return true;
    const auto size = SolidSize(parsed->first);
    return size && SupportedSolidFrame(f, payload, *size);
}

FileSys::VirtualFile DecompressNCZ(const FileSys::VirtualFile& f, std::string_view name,
                                   std::unique_ptr<NczCache> cache) {
    return OpenNcz(f, name, std::move(cache));
}

FileSys::VirtualFile DecompressNCZForInstall(const FileSys::VirtualFile& f,
                                             std::string_view name) {
    return OpenNcz(f, name);
}

} // namespace Core::Crypto
