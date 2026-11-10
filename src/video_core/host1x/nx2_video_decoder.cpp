// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <optional>
#include <set>
#include <type_traits>

#if defined(_MSC_VER)
#include <excpt.h>
#endif

#include "common/logging.h"
#include "common/settings.h"
#include "video_core/host1x/ffmpeg.h"
#include "video_core/host1x/h264_packet.h"
#include "video_core/host1x/nx2_video_decoder.h"

extern "C" {
#include <libavutil/pixdesc.h>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#endif

#include <libswscale/swscale.h>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
}

namespace Tegra::Host1x {

namespace {

std::optional<NvdecCommon::VideoCodec> CodecFromMime(std::string_view mime) {
    if (mime.find("vp9") != std::string_view::npos) {
        return NvdecCommon::VideoCodec::VP9;
    }
    if (mime.find("vp8") != std::string_view::npos) {
        return NvdecCommon::VideoCodec::VP8;
    }
    // "video/av01" or "video/av1". FFmpeg decodes AV1 with libdav1d.
    if (mime.find("av01") != std::string_view::npos || mime.find("av1") != std::string_view::npos) {
        return NvdecCommon::VideoCodec::AV1;
    }
    if (mime.find("avc") != std::string_view::npos || mime.find("264") != std::string_view::npos) {
        return NvdecCommon::VideoCodec::H264;
    }
    return std::nullopt;
}

#if defined(_MSC_VER)
int FaultFilter(unsigned long code, unsigned long* out) {
    switch (code) {
    case 0xC0000005: // access violation
    case 0xC0000006: // in-page error
    case 0xC000001D: // illegal instruction
    case 0xC000008C: // array bounds exceeded
    case 0xC0000094: // integer divide by zero
    case 0xC0000095: // integer overflow
    case 0x80000002: // datatype misalignment
        *out = code;
        return EXCEPTION_EXECUTE_HANDLER;
    default:
        // C++ exceptions, breakpoints and stack overflows go on as before.
        return EXCEPTION_CONTINUE_SEARCH;
    }
}

/// Runs fn(arg); false when it faulted. No objects with destructors in this frame (SEH).
bool RunGuarded(void (*fn)(void*), void* arg, unsigned long* code) {
    __try {
        fn(arg);
    } __except (FaultFilter(GetExceptionCode(), code)) {
        return false;
    }
    return true;
}
#else
bool RunGuarded(void (*fn)(void*), void* arg, unsigned long*) {
    fn(arg);
    return true;
}
#endif

/// Runs f on this thread; false when it faulted (Windows: a decoder or GPU driver reading bad
/// memory). A movie frame must never take the game down.
template <typename F>
bool Guarded(F&& f, unsigned long& code) {
    using Fn = std::remove_reference_t<F>;
    return RunGuarded([](void* p) { (*static_cast<Fn*>(p))(); }, &f, &code);
}

} // namespace

bool IsH264Stream(std::span<const u8> data) {
    if (data.size() >= 4) {
        // Annex-B 3-byte start code: 00 00 01 <nal_header>
        if (data[0] == 0 && data[1] == 0 && data[2] == 1) {
            const u8 nal_type = data[3] & 0x1F;
            if ((data[3] & 0x80) == 0 && nal_type >= 1 && nal_type <= 23) {
                return true;
            }
        }
    }
    if (data.size() >= 5) {
        // Annex-B 4-byte start code: 00 00 00 01 <nal_header>
        if (data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1) {
            const u8 nal_type = data[4] & 0x1F;
            if ((data[4] & 0x80) == 0 && nal_type >= 1 && nal_type <= 23) {
                return true;
            }
        }
        // avcC / MP4 length-prefixed NAL unit: [u32 be length] <nal_header>
        const u32 nalu_len = (static_cast<u32>(data[0]) << 24) |
                             (static_cast<u32>(data[1]) << 16) |
                             (static_cast<u32>(data[2]) << 8) |
                             static_cast<u32>(data[3]);
        if (nalu_len > 0 && nalu_len <= data.size() - 4) {
            const u8 nal_type = data[4] & 0x1F;
            if ((data[4] & 0x80) == 0 && nal_type >= 1 && nal_type <= 23) {
                return true;
            }
        }
    }
    if (data.size() >= 7) {
        // avcC configuration record: version=1, profile, compat, level, lengthSizeMinusOne | 0xFC, numSPS | 0xE0
        if (data[0] == 1 && (data[4] & 0xFC) == 0xFC && (data[5] & 0xE0) == 0xE0 && (data[5] & 0x1F) > 0) {
            return true;
        }
    }
    return false;
}

struct Nx2VideoDecoder::Impl {
    std::string name;
    std::optional<NvdecCommon::VideoCodec> codec;
    // Pointers, not values: a context that faulted is abandoned, never freed (its state is
    // unknown, and freeing it calls into the same driver).
    std::unique_ptr<FFmpeg::Decoder> decoder;
    std::unique_ptr<FFmpeg::HardwareContext> hardware;
    std::unique_ptr<FFmpeg::DecoderContext> context;
    SwsContext* scaler{};
    bool open{};
    bool ever_open{};
    bool using_hardware{};
    bool received_any{};
    u64 errors_seen{};
    u32 hardware_failures{}; ///< Packets in a row that failed on the GPU
    u32 render_faults{};
    H264PacketConverter h264_packets;
    /// Timestamps of submitted frames not yet output, for frames that come out without one.
    std::multiset<u64> pending_pts;
    std::vector<u8> packet; ///< FFmpeg reads up to AV_INPUT_BUFFER_PADDING_SIZE past the data

    ~Impl() {
        sws_freeContext(scaler);
        context.reset();
        hardware.reset();
    }

    bool Open(bool use_gpu) {
        context.reset();
        hardware.reset();
        decoder = std::make_unique<FFmpeg::Decoder>(*codec);
        using_hardware = false;
        received_any = false;
        hardware_failures = 0;
        pending_pts.clear();
        h264_packets.Reset();
        if (decoder->GetCodec() == nullptr) {
            LOG_ERROR(HW_GPU, "nx2 video: FFmpeg has no decoder for \"{}\"", name);
            return false;
        }
        context = std::make_unique<FFmpeg::DecoderContext>(*decoder);
        if (use_gpu) {
            hardware = std::make_unique<FFmpeg::HardwareContext>();
            using_hardware = hardware->InitializeForDecoder(*context, *decoder);
            if (!using_hardware) {
                // No GPU decoder for this codec (libdav1d has none): the same context decodes
                // on the CPU.
                hardware.reset();
            }
        }
        bool opened = false;
        unsigned long code = 0;
        if (!Guarded([&] { opened = context->OpenContext(*decoder); }, code)) {
            LOG_CRITICAL(HW_GPU, "nx2 video: opening \"{}\" faulted ({:#x})", name, code);
            Abandon();
            return false;
        }
        errors_seen = context->ErrorCount();
        return opened;
    }

    /// Leaks the decoder objects after a fault inside them.
    void Abandon() {
        (void)context.release();
        (void)hardware.release();
        (void)decoder.release();
        using_hardware = false;
        open = false;
    }

    void Receive(std::vector<Nx2Picture>& pictures) {
        while (auto frame = context->ReceiveFrame()) {
            const AVFrame* f = frame->GetFrame();
            if (f->width <= 0 || f->height <= 0) {
                continue;
            }
            const s64 stamp = f->best_effort_timestamp != AV_NOPTS_VALUE ? f->best_effort_timestamp
                                                                         : f->pts;
            u64 pts{};
            if (stamp != AV_NOPTS_VALUE) {
                pts = static_cast<u64>(stamp);
                // Frames come out in display order: earlier timestamps belong to frames the
                // decoder dropped (errors, hidden frames). Left here, a frame without a
                // timestamp of its own would take one of them and show too early.
                pending_pts.erase(pending_pts.begin(), pending_pts.upper_bound(pts));
            } else if (!pending_pts.empty()) {
                pts = *pending_pts.begin();
                pending_pts.erase(pending_pts.begin());
            }
            received_any = true;
            pictures.push_back({static_cast<u32>(f->width), static_cast<u32>(f->height), pts,
                                std::move(frame)});
        }
    }

    /// Sends one packet and collects the pictures it completes. False when the decoder was
    /// replaced (a fault, or a failing GPU decoder): the packet should go to the new one.
    bool SendReceive(const FFmpeg::Packet& packet, std::vector<Nx2Picture>& pictures) {
        const size_t first = pictures.size();
        unsigned long code = 0;
        if (!Guarded(
                [&] {
                    context->SendPacket(packet);
                    Receive(pictures);
                },
                code)) {
            pictures.resize(first);
            LOG_CRITICAL(HW_GPU, "nx2 video: decoding \"{}\" on the {} faulted ({:#x})", name,
                         using_hardware ? "GPU" : "CPU", code);
            const bool had_hardware = using_hardware;
            Abandon();
            // A fault on the CPU decoder too: stop decoding this stream rather than risk more.
            if (had_hardware && codec) {
                open = Open(false);
                LOG_INFO(HW_GPU, "nx2 video: CPU decoder for \"{}\" {}", name,
                         open ? "open" : "failed to open, frames stay empty");
            }
            return false;
        }
        if (!using_hardware) {
            return true;
        }
        if (!context->UsingHardware()) {
            // get_format moved this stream to the CPU by itself.
            using_hardware = false;
            return true;
        }
        const u64 errors = context->ErrorCount();
        if (errors == errors_seen) {
            hardware_failures = 0;
            return true;
        }
        errors_seen = errors;
        // One bad packet is skipped; a GPU decoder that fails from the start or keeps failing
        // is replaced.
        if (received_any && ++hardware_failures < 3) {
            return true;
        }
        LOG_WARNING(HW_GPU, "nx2 video: GPU decode of \"{}\" failed ({}), switching to the CPU",
                    name, received_any ? "3 bad packets in a row" : "no picture yet");
        open = Open(false);
        LOG_INFO(HW_GPU, "nx2 video: CPU decoder for \"{}\" {}", name,
                 open ? "open" : "failed to open, frames stay empty");
        return false;
    }
};

Nx2VideoDecoder::Nx2VideoDecoder(std::string_view mime) : impl{std::make_unique<Impl>()} {
    impl->name = std::string{mime};
    impl->codec = CodecFromMime(mime);
    if (!impl->codec) {
        LOG_ERROR(HW_GPU, "nx2 video: no decoder for codec \"{}\"", mime);
        return;
    }
    if (Settings::values.nvdec_emulation.GetValue() == Settings::NvdecEmulation::Gpu) {
        impl->open = impl->Open(true);
        if (!impl->open && impl->decoder && impl->decoder->GetCodec() != nullptr) {
            LOG_WARNING(HW_GPU, "nx2 video: GPU decoder for \"{}\" failed to open, using the CPU",
                        mime);
            impl->open = impl->Open(false);
        }
    } else {
        impl->open = impl->Open(false);
    }
    impl->ever_open = impl->open;
    LOG_INFO(HW_GPU, "nx2 video: decoder for \"{}\" {}{}", mime,
             impl->open ? "open" : "failed to open",
             impl->open ? (impl->using_hardware ? " (GPU)" : " (CPU)") : "");
}

Nx2VideoDecoder::~Nx2VideoDecoder() = default;

bool Nx2VideoDecoder::IsOpen() const {
    return impl->open;
}

bool Nx2VideoDecoder::Failed() const {
    return impl->ever_open && !impl->open;
}

bool Nx2VideoDecoder::UsingHardware() const {
    return impl->open && impl->using_hardware;
}

const std::string& Nx2VideoDecoder::Name() const {
    return impl->name;
}

void Nx2VideoDecoder::Decode(std::span<const u8> data, u64 pts, std::vector<Nx2Picture>& pictures) {
    if (!impl->open || data.empty()) {
        return;
    }
    impl->packet.assign(data.size() + AV_INPUT_BUFFER_PADDING_SIZE, 0);
    std::memcpy(impl->packet.data(), data.data(), data.size());

    // Convert MP4/avcC length-prefixed NALUs to Annex-B start codes for FFmpeg if needed.
    if (impl->codec == NvdecCommon::VideoCodec::H264) {
        impl->h264_packets.Convert(std::span<u8>{impl->packet.data(), data.size()});
    }

    FFmpeg::Packet packet{std::span<const u8>{impl->packet.data(), data.size()}};
    packet.GetPacket()->pts = static_cast<s64>(pts);
    // Hidden VP9 frames never come out: keep the set from growing over a long movie.
    if (impl->pending_pts.size() >= 64) {
        impl->pending_pts.erase(impl->pending_pts.begin());
    }
    impl->pending_pts.insert(pts);
    if (!impl->SendReceive(packet, pictures) && impl->open) {
        // The CPU decoder took over: give it this packet too (Open cleared the timestamps).
        impl->pending_pts.insert(pts);
        impl->SendReceive(packet, pictures);
    }
}

void Nx2VideoDecoder::Drain(std::vector<Nx2Picture>& pictures) {
    if (!impl->open) {
        return;
    }
    unsigned long code = 0;
    if (!Guarded(
            [&] {
                avcodec_send_packet(impl->context->GetCodecContext(), nullptr);
                impl->Receive(pictures);
            },
            code)) {
        LOG_CRITICAL(HW_GPU, "nx2 video: draining \"{}\" faulted ({:#x})", impl->name, code);
        impl->Abandon();
        impl->open = impl->codec && impl->Open(false);
        return;
    }
    Reset();
}

void Nx2VideoDecoder::Reset() {
    if (impl->open) {
        unsigned long code = 0;
        if (!Guarded([&] { avcodec_flush_buffers(impl->context->GetCodecContext()); }, code)) {
            LOG_CRITICAL(HW_GPU, "nx2 video: flushing \"{}\" faulted ({:#x})", impl->name, code);
            impl->Abandon();
            impl->open = impl->codec && impl->Open(false);
        }
    }
    impl->pending_pts.clear();
    impl->h264_packets.Reset();
}

void Nx2VideoDecoder::Render(const Nx2Picture& picture, Nx2PictureFormat format, u32 width,
                             u32 height, u64 pitch, u64 chroma_offset, std::span<u8> out) {
    if (!picture.frame) {
        return;
    }
    unsigned long code = 0;
    if (!Guarded([&] { RenderUnguarded(picture, format, width, height, pitch, chroma_offset, out); },
                 code)) {
        // The image keeps what it held; the next frame tries again.
        if (impl->render_faults++ < 8) {
            LOG_ERROR(HW_GPU, "nx2 video: converting a {}x{} frame of \"{}\" faulted ({:#x})",
                      picture.width, picture.height, impl->name, code);
        }
    }
}

void Nx2VideoDecoder::RenderUnguarded(const Nx2Picture& picture, Nx2PictureFormat format,
                                      u32 width, u32 height, u64 pitch, u64 chroma_offset,
                                      std::span<u8> out) {
    const AVFrame* f = picture.frame->GetFrame();
    const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(f->format));
    if (desc == nullptr || (desc->flags & AV_PIX_FMT_FLAG_HWACCEL) || f->data[0] == nullptr ||
        f->width <= 0 || f->height <= 0) {
        // Never read a GPU surface or an empty frame as memory.
        return;
    }
    const u64 needed = format == Nx2PictureFormat::Nv12
                           ? chroma_offset + pitch * ((u64{height} + 1) / 2)
                           : pitch * height;
    const u64 min_pitch = format == Nx2PictureFormat::Nv12 ? ((u64{width} + 1) / 2) * 2 : 0;
    if (width == 0 || height == 0 || pitch < min_pitch || out.size() < needed) {
        return;
    }
    const AVPixelFormat target = format == Nx2PictureFormat::Nv12    ? AV_PIX_FMT_NV12
                                 : format == Nx2PictureFormat::Rgba8 ? AV_PIX_FMT_RGBA
                                                                     : AV_PIX_FMT_BGRA;
    // Frames copied back from a host GPU decoder are already NV12. Copy them as they are:
    // FFmpeg 6.1's swscale leaves the chroma plane of an NV12 to NV12 copy at zero, which shows
    // the movies green.
    if (target == AV_PIX_FMT_NV12 && f->format == AV_PIX_FMT_NV12 &&
        static_cast<u32>(f->width) == width && static_cast<u32>(f->height) == height) {
        for (u32 y = 0; y < height; ++y) {
            std::memcpy(out.data() + y * pitch, f->data[0] + y * f->linesize[0], width);
        }
        for (u32 y = 0; y < (height + 1) / 2; ++y) {
            std::memcpy(out.data() + chroma_offset + y * pitch, f->data[1] + y * f->linesize[1],
                        min_pitch);
        }
        return;
    }
    if (target == AV_PIX_FMT_NV12 && f->format == AV_PIX_FMT_YUV420P &&
        f->color_range != AVCOL_RANGE_JPEG && static_cast<u32>(f->width) == width &&
        static_cast<u32>(f->height) == height) {
        for (u32 y = 0; y < height; ++y) {
            std::memcpy(out.data() + y * pitch, f->data[0] + y * f->linesize[0], width);
        }
        const u32 chroma_width = static_cast<u32>(min_pitch / 2);
        for (u32 y = 0; y < (height + 1) / 2; ++y) {
            const u8* u = f->data[1] + y * f->linesize[1];
            const u8* v = f->data[2] + y * f->linesize[2];
            u8* dst = out.data() + chroma_offset + y * pitch;
            for (u32 x = 0; x < chroma_width; ++x) {
                dst[x * 2] = u[x];
                dst[x * 2 + 1] = v[x];
            }
        }
        return;
    }
    impl->scaler = sws_getCachedContext(impl->scaler, f->width, f->height,
                                        static_cast<AVPixelFormat>(f->format),
                                        static_cast<int>(width), static_cast<int>(height), target,
                                        SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (impl->scaler == nullptr) {
        LOG_ERROR(HW_GPU, "nx2 video: cannot convert {} to {}",
                  av_get_pix_fmt_name(static_cast<AVPixelFormat>(f->format)),
                  av_get_pix_fmt_name(target));
        return;
    }
    // The stream's matrix and range; untagged HD video is BT.709.
    int space = SWS_CS_ITU601;
    if (f->colorspace == AVCOL_SPC_BT709 ||
        (f->colorspace == AVCOL_SPC_UNSPECIFIED && f->height >= 720)) {
        space = SWS_CS_ITU709;
    } else if (f->colorspace == AVCOL_SPC_BT2020_NCL || f->colorspace == AVCOL_SPC_BT2020_CL) {
        space = SWS_CS_BT2020;
    }
    const bool full_range_source = f->color_range == AVCOL_RANGE_JPEG ||
                                  f->format == AV_PIX_FMT_YUVJ420P ||
                                  f->format == AV_PIX_FMT_YUVJ422P ||
                                  f->format == AV_PIX_FMT_YUVJ444P ||
                                  f->format == AV_PIX_FMT_YUVJ440P ||
                                  f->format == AV_PIX_FMT_YUVJ411P;
    const int* coefficients = sws_getCoefficients(space);
    sws_setColorspaceDetails(impl->scaler, coefficients, full_range_source, coefficients,
                             format != Nx2PictureFormat::Nv12, 0, 1 << 16, 1 << 16);
    std::array<u8*, 4> planes{out.data(), nullptr, nullptr, nullptr};
    std::array<int, 4> strides{static_cast<int>(pitch), 0, 0, 0};
    if (format == Nx2PictureFormat::Nv12) {
        planes[1] = out.data() + chroma_offset;
        strides[1] = static_cast<int>(pitch);
    }
    sws_scale(impl->scaler, f->data, f->linesize, 0, f->height, planes.data(), strides.data());
}

} // namespace Tegra::Host1x
