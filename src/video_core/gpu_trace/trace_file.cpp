// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>
#include <unordered_set>

#define XXH_INLINE_ALL
#include <xxhash.h>
#include <zstd.h>

#include "common/fs/fs.h"
#include "common/logging.h"
#include "video_core/gpu_trace/trace_file.h"

namespace VideoCommon::GpuTrace {
namespace {
struct HashKey {
    size_t operator()(const Hash& hash) const noexcept {
        return static_cast<size_t>(hash[0] ^ (hash[1] * 0x9E3779B97F4A7C15ULL));
    }
};

struct FileCloser {
    void operator()(std::FILE* file) const noexcept {
        std::fclose(file);
    }
};
using File = std::unique_ptr<std::FILE, FileCloser>;

File OpenFile(const std::filesystem::path& path, bool write) {
#ifdef _WIN32
    return File{_wfopen(path.c_str(), write ? L"wb" : L"rb")};
#else
    return File{std::fopen(path.c_str(), write ? "wb" : "rb")};
#endif
}
} // Anonymous namespace

Hash HashBytes(std::span<const u8> bytes) {
    const XXH128_hash_t hash = XXH3_128bits(bytes.data(), bytes.size());
    return {hash.low64, hash.high64};
}

bool HasPayload(const Record& record, size_t size) {
    return record.payload.size() >= size;
}

struct Writer::Impl {
    File file;
    ZSTD_CCtx* context = nullptr;
    std::vector<u8> out_buffer = std::vector<u8>(ZSTD_CStreamOutSize());
    std::unordered_set<Hash, HashKey> blobs;
    bool failed = false;

    ~Impl() {
        ZSTD_freeCCtx(context);
    }

    void Compress(std::span<const u8> data, ZSTD_EndDirective mode) {
        ZSTD_inBuffer input{data.data(), data.size(), 0};
        for (;;) {
            ZSTD_outBuffer output{out_buffer.data(), out_buffer.size(), 0};
            const size_t remaining = ZSTD_compressStream2(context, &output, &input, mode);
            if (ZSTD_isError(remaining)) {
                LOG_ERROR(HW_GPU, "GPU trace: compression failed: {}", ZSTD_getErrorName(remaining));
                failed = true;
                return;
            }
            if (output.pos != 0 && std::fwrite(out_buffer.data(), 1, output.pos, file.get()) !=
                                       output.pos) {
                failed = true;
                return;
            }
            const bool done = mode == ZSTD_e_continue ? input.pos == input.size : remaining == 0;
            if (done) {
                return;
            }
        }
    }
};

Writer::Writer(const std::filesystem::path& path, int compression_level)
    : impl{std::make_unique<Impl>()} {
    impl->file = OpenFile(path, true);
    if (!impl->file) {
        LOG_ERROR(HW_GPU, "GPU trace: can't create {}", path.string());
        return;
    }
    impl->context = ZSTD_createCCtx();
    ZSTD_CCtx_setParameter(impl->context, ZSTD_c_compressionLevel, compression_level);
    ZSTD_CCtx_setParameter(impl->context, ZSTD_c_checksumFlag, 1);
#ifdef ZSTD_MULTITHREAD
    ZSTD_CCtx_setParameter(impl->context, ZSTD_c_nbWorkers, 4);
#endif
    const FileHeader header{};
    if (std::fwrite(&header, sizeof(header), 1, impl->file.get()) != 1) {
        impl->failed = true;
    }
}

Writer::~Writer() = default;

bool Writer::IsOpen() const {
    return impl->file && impl->context && !impl->failed;
}

void Writer::Write(RecordType type, std::span<const std::span<const u8>> parts) {
    if (!IsOpen()) {
        return;
    }
    RecordHeader header{type, 0, 0};
    for (const auto part : parts) {
        header.size += part.size();
    }
    impl->Compress({reinterpret_cast<const u8*>(&header), sizeof(header)}, ZSTD_e_continue);
    for (const auto part : parts) {
        impl->Compress(part, ZSTD_e_continue);
    }
    bytes_in += sizeof(header) + header.size;
}

void Writer::Write(RecordType type, std::span<const u8> payload) {
    const std::array parts{payload};
    Write(type, parts);
}

Hash Writer::WriteBlob(std::span<const u8> bytes) {
    const Hash hash = HashBytes(bytes);
    if (impl->blobs.insert(hash).second) {
        const BlobRecord record{hash};
        const std::array<std::span<const u8>, 2> parts{
            std::span<const u8>{reinterpret_cast<const u8*>(&record), sizeof(record)}, bytes};
        Write(RecordType::Blob, parts);
    }
    return hash;
}

void Writer::WriteMemory(u64 physical, std::span<const u8> bytes) {
    WritePod(RecordType::Memory, MemoryRecord{physical, bytes.size(), WriteBlob(bytes)});
}

bool Writer::Finish() {
    if (!IsOpen()) {
        return false;
    }
    Write(RecordType::End, std::span<const u8>{});
    impl->Compress({}, ZSTD_e_end);
    const bool ok = !impl->failed && std::fflush(impl->file.get()) == 0;
    impl->file.reset();
    return ok;
}

struct Reader::Impl {
    File file;
    ZSTD_DCtx* context = nullptr;
    std::vector<u8> in_buffer = std::vector<u8>(ZSTD_DStreamInSize());
    ZSTD_inBuffer input{in_buffer.data(), 0, 0};
    bool ended = false;

    ~Impl() {
        ZSTD_freeDCtx(context);
    }
};

Reader::Reader(const std::filesystem::path& path) : impl{std::make_unique<Impl>()} {
    impl->file = OpenFile(path, false);
    if (!impl->file) {
        error = "can't open " + path.string();
        return;
    }
    FileHeader header{};
    if (std::fread(&header, sizeof(header), 1, impl->file.get()) != 1 || header.magic != Magic) {
        error = path.string() + " is not a GPU trace";
        return;
    }
    if (header.version != FormatVersion) {
        error = "trace format version " + std::to_string(header.version) + ", this build reads " +
                std::to_string(FormatVersion);
        return;
    }
    if (header.header_size > sizeof(header)) {
        std::fseek(impl->file.get(), static_cast<long>(header.header_size), SEEK_SET);
    }
    impl->context = ZSTD_createDCtx();
}

Reader::~Reader() = default;

bool Reader::Fill(void* out, size_t size) {
    ZSTD_outBuffer output{out, size, 0};
    while (output.pos < output.size) {
        if (impl->input.pos == impl->input.size) {
            const size_t read =
                std::fread(impl->in_buffer.data(), 1, impl->in_buffer.size(), impl->file.get());
            if (read == 0) {
                error = "the trace ends early (truncated file)";
                return false;
            }
            impl->input = {impl->in_buffer.data(), read, 0};
        }
        const size_t result = ZSTD_decompressStream(impl->context, &output, &impl->input);
        if (ZSTD_isError(result)) {
            error = std::string{"decompression failed: "} + ZSTD_getErrorName(result);
            return false;
        }
    }
    return true;
}

std::optional<Record> Reader::Next() {
    if (!error.empty() || impl->ended || !impl->context) {
        return std::nullopt;
    }
    RecordHeader header{};
    if (!Fill(&header, sizeof(header))) {
        return std::nullopt;
    }
    if (header.size > (u64{1} << 34)) {
        error = "corrupt record size";
        return std::nullopt;
    }
    Record record{header.type, std::vector<u8>(static_cast<size_t>(header.size))};
    if (header.size != 0 && !Fill(record.payload.data(), record.payload.size())) {
        return std::nullopt;
    }
    if (record.type == RecordType::End) {
        impl->ended = true;
    }
    return record;
}

} // namespace VideoCommon::GpuTrace
