// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <optional>
#include <span>
#include <vector>

#include "common/fs/path_util.h"
#include "common/logging.h"
#include "video_core/nx2_stutter_trace.h"
#include "video_core/renderer_vulkan/vk_pipeline_binaries.h"
#include "video_core/shader_cache_v2/cache.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {
namespace {

namespace ShaderCacheV2 = VideoCommon::ShaderCacheV2;
namespace StutterTrace = VideoCommon::StutterTrace;
using ShaderCacheV2::RecordKind;

const VkBaseInStructure* FindNext(const void* next, VkStructureType type) {
    for (auto* item = static_cast<const VkBaseInStructure*>(next); item; item = item->pNext) {
        if (item->sType == type) {
            return item;
        }
    }
    return nullptr;
}

/// AMD's proprietary driver keeps no graphics pipeline with dynamic vertex input in its own disk
/// cache, so where the store runs there it is all there is: a shader precache stores every cached
/// pipeline (~600 MB for a game with ~17,000) and the default 384 MiB compaction dropped a third
/// of it. AMD compacts past 1 GiB, and never drops what a session used or wrote below 4 GiB.
ShaderCacheV2::DriverBlobOptions BlobOptions(const Device& device) {
    ShaderCacheV2::DriverBlobOptions options{};
    if (device.GetDriverID() == VK_DRIVER_ID_AMD_PROPRIETARY) {
        options.max_bytes = u64{1} << 30;
        options.target_bytes = options.max_bytes / 8 * 7;
        options.session_cap_bytes = u64{4} << 30;
        options.ceiling_bytes = options.session_cap_bytes;
    }
    return options;
}

/// Libraries, and create infos that already chain binaries or flags of their own, are left alone.
template <typename CreateInfo>
bool TakesBinaries(const CreateInfo& ci) {
    return (ci.flags & VK_PIPELINE_CREATE_LIBRARY_BIT_KHR) == 0 &&
           !FindNext(ci.pNext, VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR) &&
           !FindNext(ci.pNext, VK_STRUCTURE_TYPE_PIPELINE_BINARY_INFO_KHR) &&
           !FindNext(ci.pNext, VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR) &&
           !FindNext(ci.pNext, VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT);
}

VkResult CreateRaw(const vk::DeviceDispatch& dld, VkDevice device,
                   const VkGraphicsPipelineCreateInfo& ci, VkPipelineCache cache,
                   VkPipeline* pipeline) {
    return dld.vkCreateGraphicsPipelines(device, cache, 1, &ci, nullptr, pipeline);
}

VkResult CreateRaw(const vk::DeviceDispatch& dld, VkDevice device,
                   const VkComputePipelineCreateInfo& ci, VkPipelineCache cache,
                   VkPipeline* pipeline) {
    return dld.vkCreateComputePipelines(device, cache, 1, &ci, nullptr, pipeline);
}

vk::Pipeline CreateNormally(const vk::Device& logical, const VkGraphicsPipelineCreateInfo& ci,
                            VkPipelineCache cache) {
    return logical.CreateGraphicsPipeline(ci, cache);
}

vk::Pipeline CreateNormally(const vk::Device& logical, const VkComputePipelineCreateInfo& ci,
                            VkPipelineCache cache) {
    return logical.CreateComputePipeline(ci, cache);
}

std::optional<std::vector<u8>> KeyBytes(const VkPipelineBinaryKeyKHR& key) {
    if (key.keySize == 0 || key.keySize > VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR) {
        return std::nullopt;
    }
    return std::vector<u8>(key.key, key.key + key.keySize);
}

void DestroyBinaries(const vk::DeviceDispatch& dld, VkDevice device,
                     std::span<const VkPipelineBinaryKHR> binaries) {
    for (const VkPipelineBinaryKHR binary : binaries) {
        if (binary != VK_NULL_HANDLE) {
            dld.vkDestroyPipelineBinaryKHR(device, binary, nullptr);
        }
    }
}

/// The pipeline built from stored binaries, or null when the driver refuses them
template <typename CreateInfo>
VkPipeline CreateFromBinaries(const vk::DeviceDispatch& dld, VkDevice device,
                              const ShaderCacheV2::PipelineBinaries& stored, const CreateInfo& ci) {
    const u32 count = static_cast<u32>(stored.binaries.size());
    std::vector<VkPipelineBinaryKeyKHR> keys(count);
    std::vector<VkPipelineBinaryDataKHR> data(count);
    for (u32 i = 0; i < count; ++i) {
        const ShaderCacheV2::PipelineBinaries::Binary& binary = stored.binaries[i];
        keys[i] = {.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR,
                   .keySize = static_cast<u32>(binary.key.size())};
        std::copy(binary.key.begin(), binary.key.end(), keys[i].key);
        data[i] = {.dataSize = binary.data.size(), .pData = const_cast<u8*>(binary.data.data())};
    }
    const VkPipelineBinaryKeysAndDataKHR keys_and_data{
        .binaryCount = count,
        .pPipelineBinaryKeys = keys.data(),
        .pPipelineBinaryData = data.data(),
    };
    const VkPipelineBinaryCreateInfoKHR binary_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR,
        .pKeysAndDataInfo = &keys_and_data,
    };
    std::vector<VkPipelineBinaryKHR> binaries(count, VK_NULL_HANDLE);
    VkPipelineBinaryHandlesInfoKHR handles{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR,
        .pipelineBinaryCount = count,
        .pPipelineBinaries = binaries.data(),
    };
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (dld.vkCreatePipelineBinariesKHR(device, &binary_ci, nullptr, &handles) == VK_SUCCESS) {
        const VkPipelineBinaryInfoKHR binary_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_INFO_KHR,
            .pNext = ci.pNext,
            .binaryCount = count,
            .pPipelineBinaries = binaries.data(),
        };
        CreateInfo from_binaries = ci;
        from_binaries.pNext = &binary_info;
        // A pipeline built from binaries takes no pipeline cache.
        if (CreateRaw(dld, device, from_binaries, VK_NULL_HANDLE, &pipeline) != VK_SUCCESS) {
            pipeline = VK_NULL_HANDLE;
        }
    }
    DestroyBinaries(dld, device, binaries);
    return pipeline;
}

/// The binaries of a pipeline compiled with VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR
std::optional<ShaderCacheV2::PipelineBinaries> Capture(const vk::DeviceDispatch& dld,
                                                       VkDevice device, VkPipeline pipeline) {
    const VkPipelineBinaryCreateInfoKHR binary_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR,
        .pipeline = pipeline,
    };
    VkPipelineBinaryHandlesInfoKHR handles{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR,
    };
    if (dld.vkCreatePipelineBinariesKHR(device, &binary_ci, nullptr, &handles) != VK_SUCCESS ||
        handles.pipelineBinaryCount == 0 ||
        handles.pipelineBinaryCount > ShaderCacheV2::MaxPipelineBinaries) {
        return std::nullopt;
    }
    std::vector<VkPipelineBinaryKHR> binaries(handles.pipelineBinaryCount, VK_NULL_HANDLE);
    handles.pPipelineBinaries = binaries.data();
    std::optional<ShaderCacheV2::PipelineBinaries> captured;
    if (dld.vkCreatePipelineBinariesKHR(device, &binary_ci, nullptr, &handles) == VK_SUCCESS) {
        captured.emplace();
        for (const VkPipelineBinaryKHR binary : binaries) {
            const VkPipelineBinaryDataInfoKHR data_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_DATA_INFO_KHR,
                .pipelineBinary = binary,
            };
            VkPipelineBinaryKeyKHR binary_key{.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
            size_t size = 0;
            std::optional<std::vector<u8>> key_bytes;
            if (dld.vkGetPipelineBinaryDataKHR(device, &data_info, &binary_key, &size, nullptr) !=
                    VK_SUCCESS ||
                size == 0 || !(key_bytes = KeyBytes(binary_key))) {
                captured.reset();
                break;
            }
            std::vector<u8> data(size);
            if (dld.vkGetPipelineBinaryDataKHR(device, &data_info, &binary_key, &size,
                                               data.data()) != VK_SUCCESS) {
                captured.reset();
                break;
            }
            data.resize(size);
            captured->binaries.push_back({.key = std::move(*key_bytes), .data = std::move(data)});
        }
    }
    DestroyBinaries(dld, device, binaries);
    return captured;
}

} // Anonymous namespace

std::unique_ptr<PipelineBinaries> PipelineBinaries::Open(const Device& device,
                                                         const std::filesystem::path& directory) {
    const vk::DeviceDispatch& dld = device.GetDispatchLoader();
    if (!device.IsKhrPipelineBinarySupported() || !dld.vkGetPipelineKeyKHR ||
        !dld.vkCreatePipelineBinariesKHR || !dld.vkDestroyPipelineBinaryKHR ||
        !dld.vkGetPipelineBinaryDataKHR || !dld.vkReleaseCapturedPipelineDataKHR) {
        return nullptr;
    }
    // No global key (vkGetPipelineKeyKHR without a create info): NVIDIA 610.88 overwrites the low
    // half of RDI there, a register the caller keeps. The driver version and cache UUID name the
    // driver, and each blob is checked against its pipeline's key.
    ShaderCacheV2::DeviceIdentity identity{
        .vendor_id = device.GetVendorID(),
        .device_id = device.GetDeviceID(),
        .driver_id = static_cast<u32>(device.GetDriverID()),
        .driver_version = device.GetDriverVersion(),
    };
    std::ranges::copy(device.GetPipelineCacheUUID(), identity.pipeline_cache_uuid.begin());
    auto binaries = std::make_unique<PipelineBinaries>(
        device, directory / ShaderCacheV2::DriverFileName, identity);
    if (binaries->store.Discarded() == ShaderCacheV2::DriverBlobDiscard::OtherIdentity) {
        LOG_INFO(Render_Vulkan, "Driver blobs: the device or driver changed, starting over");
    } else if (binaries->store.Discarded() == ShaderCacheV2::DriverBlobDiscard::NoIdentity) {
        LOG_WARNING(Render_Vulkan, "Driver blobs: the file names no device, starting over");
    }
    if (!binaries->store.Active()) {
        return nullptr;
    }
    const ShaderCacheV2::DriverBlobCounts counts = binaries->store.Counts();
    LOG_INFO(Render_Vulkan,
             "Driver blobs (VK_KHR_pipeline_binary): {} pipelines stored for this driver ({} "
             "bytes) in {}",
             counts.loaded, counts.loaded_bytes,
             Common::FS::PathToUTF8String(directory / ShaderCacheV2::DriverFileName));
    return binaries;
}

PipelineBinaries::PipelineBinaries(const Device& device_, std::filesystem::path file,
                                   const ShaderCacheV2::DeviceIdentity& identity)
    : device{device_}, store{std::move(file), identity, BlobOptions(device_)} {}

vk::Pipeline PipelineBinaries::CreateGraphicsPipeline(const VkGraphicsPipelineCreateInfo& ci,
                                                      VkPipelineCache cache, u64 key) {
    return Create(ci, cache, RecordKind::GraphicsBinary, key);
}

vk::Pipeline PipelineBinaries::CreateComputePipeline(const VkComputePipelineCreateInfo& ci,
                                                     VkPipelineCache cache, u64 key) {
    return Create(ci, cache, RecordKind::ComputeBinary, key);
}

void PipelineBinaries::LogCounts(std::string_view when) const {
    const ShaderCacheV2::DriverBlobCounts counts = Counts();
    LOG_INFO(Render_Vulkan,
             "Driver blobs {}: {} loaded ({} bytes), {} used, {} rejected, {} written ({} bytes)",
             when, counts.loaded, counts.loaded_bytes, counts.used, counts.rejected, counts.written,
             counts.written_bytes);
}

template <typename CreateInfo>
vk::Pipeline PipelineBinaries::Create(const CreateInfo& ci, VkPipelineCache cache, RecordKind kind,
                                      u64 key) {
    const vk::Device& logical = device.GetLogical();
    const vk::DeviceDispatch& dld = device.GetDispatchLoader();
    const VkDevice handle = *logical;
    if (!TakesBinaries(ci)) {
        return CreateNormally(logical, ci, cache);
    }
    const VkPipelineCreateInfoKHR key_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_INFO_KHR,
        .pNext = const_cast<CreateInfo*>(&ci),
    };
    VkPipelineBinaryKeyKHR pipeline_key{.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
    std::optional<std::vector<u8>> driver_key;
    if (dld.vkGetPipelineKeyKHR(handle, &key_info, &pipeline_key) != VK_SUCCESS ||
        !(driver_key = KeyBytes(pipeline_key))) {
        return CreateNormally(logical, ci, cache);
    }
    if (const auto payload = store.Find(kind, key)) {
        const auto stored = ShaderCacheV2::DecodeBinaries(*payload);
        if (stored && stored->pipeline_key == *driver_key) {
            if (const VkPipeline pipeline = CreateFromBinaries(dld, handle, *stored, ci)) {
                store.Used(kind, key);
                StutterTrace::CountDriverBuild(StutterTrace::DriverBuild::Binary);
                return vk::Pipeline(pipeline, handle, dld);
            }
        }
        // Stale (new SPIR-V for this key) or refused: compiled below, and replaced
        store.Reject(kind, key);
        StutterTrace::CountDriverBuild(StutterTrace::DriverBuild::Rejected);
    }
    if (store.Full()) {
        return CreateNormally(logical, ci, cache);
    }

    const VkPipelineCreateFlags2CreateInfoKHR capture_flags{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR,
        .pNext = ci.pNext,
        .flags = static_cast<VkPipelineCreateFlags2KHR>(ci.flags) |
                 VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR,
    };
    CreateInfo capture_ci = ci;
    capture_ci.pNext = &capture_flags;
    VkPipeline pipeline = VK_NULL_HANDLE;
    // A capturing build takes no pipeline cache either
    // (VUID-vkCreateGraphicsPipelines-pNext-09617).
    if (CreateRaw(dld, handle, capture_ci, VK_NULL_HANDLE, &pipeline) != VK_SUCCESS) {
        if (pipeline != VK_NULL_HANDLE) {
            dld.vkDestroyPipeline(handle, pipeline, nullptr);
        }
        // Compiled as without the store, which reports its own errors
        return CreateNormally(logical, ci, cache);
    }
    StutterTrace::CountDriverBuild(StutterTrace::DriverBuild::Compiled);
    if (auto captured = Capture(dld, handle, pipeline)) {
        captured->pipeline_key = std::move(*driver_key);
        store.Store(kind, key, ShaderCacheV2::EncodeBinaries(*captured));
    }
    const VkReleaseCapturedPipelineDataInfoKHR release{
        .sType = VK_STRUCTURE_TYPE_RELEASE_CAPTURED_PIPELINE_DATA_INFO_KHR,
        .pipeline = pipeline,
    };
    dld.vkReleaseCapturedPipelineDataKHR(handle, &release, nullptr);
    return vk::Pipeline(pipeline, handle, dld);
}

} // namespace Vulkan
