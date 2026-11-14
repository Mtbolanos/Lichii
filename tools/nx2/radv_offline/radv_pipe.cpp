// SPDX-License-Identifier: GPL-3.0-or-later

// Builds one Vulkan pipeline on the first RADV device and writes what the driver made of it:
// every pipeline executable's statistics and internal representations (NIR, ACO IR, assembly).
// Run it under the amdgpu drm-shim (LD_PRELOAD=$RADV_OFFLINE_SHIM AMDGPU_GPU_ID=<device>) to get
// a RADV device for a chip without the GPU.
//
//   radv_pipe <description.txt> <output dir>
//   radv_pipe --profile    prints the device's shader-sweep --device-profile JSON
//
// compile.py writes the description; see README.md for its lines.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

PFN_vkGetInstanceProcAddr get_instance_proc;
VkInstance instance;
VkDevice device;

#define FN(name) reinterpret_cast<PFN_##name>(get_instance_proc(instance, #name))

std::string JsonString(const std::string& text) {
    std::string out = "\"";
    for (const char c : text) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            char buffer[8];
            std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
            out += buffer;
        } else {
            out += c;
        }
    }
    return out + "\"";
}

[[noreturn]] void Fail(const std::string& message) {
    std::printf("{\"status\": \"error\", \"error\": %s}\n", JsonString(message).c_str());
    std::exit(1);
}

void Check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) {
        Fail(std::string(what) + " returned VkResult " + std::to_string(result));
    }
}

struct Stage {
    VkShaderStageFlagBits stage;
    std::string path;
    std::string entry;
    uint32_t subgroup_size = 0;
    bool full_subgroups = false;
    std::vector<VkSpecializationMapEntry> spec_entries;
    std::vector<uint32_t> spec_data;
};

struct Description {
    bool compute = false;
    std::vector<Stage> stages;
    std::map<uint32_t, std::vector<VkDescriptorSetLayoutBinding>> sets;
    VkPushConstantRange push{};
    std::vector<VkFormat> colors;
    VkFormat depth = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    uint32_t patch_points = 3;
    bool dynamic_vertex_input = true;
    std::vector<VkVertexInputAttributeDescription> attributes;
    bool robust_buffer_access = true;
};

VkShaderStageFlagBits ParseStage(const std::string& name) {
    static const std::map<std::string, VkShaderStageFlagBits> stages{
        {"vertex", VK_SHADER_STAGE_VERTEX_BIT},
        {"tess-control", VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT},
        {"tess-eval", VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT},
        {"geometry", VK_SHADER_STAGE_GEOMETRY_BIT},
        {"fragment", VK_SHADER_STAGE_FRAGMENT_BIT},
        {"compute", VK_SHADER_STAGE_COMPUTE_BIT},
        {"task", VK_SHADER_STAGE_TASK_BIT_EXT},
        {"mesh", VK_SHADER_STAGE_MESH_BIT_EXT},
    };
    const auto it = stages.find(name);
    if (it == stages.end()) {
        Fail("unknown stage " + name);
    }
    return it->second;
}

Description ReadDescription(const char* path) {
    std::ifstream file(path);
    if (!file) {
        Fail(std::string("cannot read ") + path);
    }
    Description desc;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream in(line);
        std::string word;
        if (!(in >> word) || word[0] == '#') {
            continue;
        }
        if (word == "kind") {
            in >> word;
            desc.compute = word == "compute";
        } else if (word == "stage") {
            Stage stage;
            in >> word >> stage.path >> stage.entry;
            stage.stage = ParseStage(word);
            while (in >> word) {
                if (word == "full") {
                    stage.full_subgroups = true;
                } else if (word.rfind("subgroup=", 0) == 0) {
                    stage.subgroup_size = static_cast<uint32_t>(std::stoul(word.substr(9)));
                }
            }
            desc.stages.push_back(stage);
        } else if (word == "spec") {
            size_t index;
            uint32_t id, value;
            in >> index >> id >> value;
            if (index >= desc.stages.size()) {
                Fail("spec line before its stage");
            }
            Stage& stage = desc.stages[index];
            stage.spec_entries.push_back({id, static_cast<uint32_t>(stage.spec_data.size() * 4), 4});
            stage.spec_data.push_back(value);
        } else if (word == "binding") {
            uint32_t set, binding, type, count, stages;
            in >> set >> binding >> type >> count >> std::hex >> stages;
            desc.sets[set].push_back({binding, static_cast<VkDescriptorType>(type), count,
                                      static_cast<VkShaderStageFlags>(stages), nullptr});
        } else if (word == "push") {
            in >> std::hex >> desc.push.stageFlags >> std::dec >> desc.push.size;
        } else if (word == "color") {
            uint32_t format;
            in >> format;
            desc.colors.push_back(static_cast<VkFormat>(format));
        } else if (word == "depth") {
            uint32_t format;
            in >> format;
            desc.depth = static_cast<VkFormat>(format);
        } else if (word == "samples") {
            uint32_t samples;
            in >> samples;
            desc.samples = static_cast<VkSampleCountFlagBits>(samples);
        } else if (word == "topology") {
            uint32_t topology;
            in >> topology;
            desc.topology = static_cast<VkPrimitiveTopology>(topology);
        } else if (word == "patch") {
            in >> desc.patch_points;
        } else if (word == "vertex_dynamic") {
            int value;
            in >> value;
            desc.dynamic_vertex_input = value != 0;
        } else if (word == "attribute") {
            VkVertexInputAttributeDescription attribute{};
            uint32_t format;
            in >> attribute.location >> attribute.binding >> format >> attribute.offset;
            attribute.format = static_cast<VkFormat>(format);
            desc.attributes.push_back(attribute);
        } else if (word == "robust") {
            in >> word;
            desc.robust_buffer_access = word != "none";
        } else {
            Fail("unknown description line: " + line);
        }
    }
    if (desc.stages.empty()) {
        Fail("the description has no stage");
    }
    return desc;
}

std::vector<uint32_t> ReadSpirv(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        Fail("cannot read " + path);
    }
    const auto size = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> words(size / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(words.size() * 4));
    return words;
}

struct FeatureStruct {
    VkStructureType type;
    const char* extension;
};

// Lichii enables these when the device has them. Promoted features come from the Vulkan 1.1-1.3
// structures, which must not be chained next to the extension structures they replace.
constexpr FeatureStruct feature_structs[] = {
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, nullptr},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, nullptr},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, nullptr},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT,
     VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT,
     VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT,
     VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT,
     VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT, VK_EXT_MESH_SHADER_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT,
     VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT,
     VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR,
     VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
     VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT,
     VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT,
     VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT,
     VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT,
     VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_EXT,
     VK_EXT_LINE_RASTERIZATION_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT,
     VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVE_TOPOLOGY_LIST_RESTART_FEATURES_EXT,
     VK_EXT_PRIMITIVE_TOPOLOGY_LIST_RESTART_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
     VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_MAXIMAL_RECONVERGENCE_FEATURES_KHR,
     VK_KHR_SHADER_MAXIMAL_RECONVERGENCE_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR,
     VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT,
     VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR,
     VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COLOR_WRITE_ENABLE_FEATURES_EXT,
     VK_EXT_COLOR_WRITE_ENABLE_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_QUAD_CONTROL_FEATURES_KHR,
     VK_KHR_SHADER_QUAD_CONTROL_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES_KHR,
     VK_KHR_SHADER_FLOAT_CONTROLS_2_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_KHR,
     VK_KHR_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_KHR, VK_KHR_INDEX_TYPE_UINT8_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR, VK_KHR_MAINTENANCE_5_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_2D_VIEW_OF_3D_FEATURES_EXT,
     VK_EXT_IMAGE_2D_VIEW_OF_3D_EXTENSION_NAME},
    {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT,
     VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME},
};

// Extensions that change what a pipeline may contain but have no feature structure.
constexpr const char* plain_extensions[] = {
    VK_EXT_SHADER_STENCIL_EXPORT_EXTENSION_NAME,     VK_EXT_SHADER_VIEWPORT_INDEX_LAYER_EXTENSION_NAME,
    VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,  VK_EXT_SAMPLER_FILTER_MINMAX_EXTENSION_NAME,
    VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
};

std::string FileName(std::string text) {
    for (char& c : text) {
        if (!std::isalnum(static_cast<unsigned char>(c))) {
            c = '-';
        }
    }
    return text;
}

// The Shader::HostTranslateInfo and Shader::Profile fields PipelineCache's constructor
// (vk_pipeline_cache.cpp) and Vulkan::Device take from this device, as shader-sweep's
// --device-profile JSON. Lichii's other device workarounds don't touch RADV.
template <typename HasExtension>
void PrintProfile(VkPhysicalDevice physical, const HasExtension& has_extension) {
    VkPhysicalDeviceShaderQuadControlFeaturesKHR quad_control{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_QUAD_CONTROL_FEATURES_KHR};
    VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR workgroup{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR, &quad_control};
    VkPhysicalDeviceTransformFeedbackFeaturesEXT xfb{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT, &workgroup};
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &xfb};
    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &v12};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v11};
    FN(vkGetPhysicalDeviceFeatures2)(physical, &features);

    VkPhysicalDeviceVulkan13Properties v13p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    VkPhysicalDeviceVulkan12Properties v12p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES, &v13p};
    VkPhysicalDeviceVulkan11Properties v11p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES, &v12p};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &v11p};
    FN(vkGetPhysicalDeviceProperties2)(physical, &properties);
    const VkPhysicalDeviceLimits& limits = properties.properties.limits;

    const auto stage_bit = [&](VkShaderStageFlags flag, unsigned stage) {
        return (v11p.subgroupSupportedStages & flag) ? 1u << stage : 0u;
    };
    // Shader::Stage: VertexB 0, TessellationControl 1, TessellationEval 2, Geometry 3, Fragment 4,
    // Compute 5, VertexA 6, Task 7, Mesh 8.
    const unsigned subgroup_stages =
        stage_bit(VK_SHADER_STAGE_VERTEX_BIT, 6) | stage_bit(VK_SHADER_STAGE_VERTEX_BIT, 0) |
        stage_bit(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, 1) |
        stage_bit(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, 2) | stage_bit(VK_SHADER_STAGE_GEOMETRY_BIT, 3) |
        stage_bit(VK_SHADER_STAGE_FRAGMENT_BIT, 4) | stage_bit(VK_SHADER_STAGE_COMPUTE_BIT, 5) |
        stage_bit(VK_SHADER_STAGE_TASK_BIT_EXT, 7) | stage_bit(VK_SHADER_STAGE_MESH_BIT_EXT, 8);
    const unsigned max_subgroup = std::max(v11p.subgroupSize, v13p.maxSubgroupSize);
    const bool workgroup_layout = has_extension(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME);
    const bool atomic_int64 = has_extension(VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME);
    const bool ops_quad = v11p.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT;
    const bool ops_vote = v11p.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_VOTE_BIT;
    const bool ops_relative = v11p.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT;
    const bool layer = has_extension(VK_EXT_SHADER_VIEWPORT_INDEX_LAYER_EXTENSION_NAME);
    const std::vector<std::pair<const char*, unsigned long long>> host = {
        {"min_ssbo_alignment", limits.minStorageBufferOffsetAlignment},
        {"max_per_stage_descriptor_sampled_images", limits.maxPerStageDescriptorSampledImages},
        {"max_per_stage_resources", limits.maxPerStageResources},
        {"max_per_stage_descriptor_storage_buffers", limits.maxPerStageDescriptorStorageBuffers},
        {"max_per_stage_descriptor_samplers", limits.maxPerStageDescriptorSamplers},
        {"max_descriptor_set_samplers", limits.maxDescriptorSetSamplers},
        {"max_descriptor_set_uniform_buffers", limits.maxDescriptorSetUniformBuffers},
        {"max_descriptor_set_uniform_buffers_dynamic", limits.maxDescriptorSetUniformBuffersDynamic},
        {"max_descriptor_set_storage_buffers", limits.maxDescriptorSetStorageBuffers},
        {"max_descriptor_set_storage_buffers_dynamic", limits.maxDescriptorSetStorageBuffersDynamic},
        {"max_descriptor_set_sampled_images", limits.maxDescriptorSetSampledImages},
        {"max_descriptor_set_storage_images", limits.maxDescriptorSetStorageImages},
        {"max_descriptor_set_input_attachements", limits.maxDescriptorSetInputAttachments},
        {"support_float64", features.features.shaderFloat64},
        {"support_float16", v12.shaderFloat16},
        {"support_int64", features.features.shaderInt64},
        {"needs_demote_reorder", 0},
        {"support_snorm_render_buffer", 1},
        {"support_viewport_index_layer", layer},
        {"support_viewport_mask", 0},
        {"support_geometry_shader_passthrough", 0},
        {"support_conditional_barrier", 1},
        {"single_lane_geometry_subgroups", !(subgroup_stages & (1u << 3))},
        {"support_byte_addressed_r16", 1},
        {"support_handle_table_arrays", v12.shaderSampledImageArrayNonUniformIndexing},
    };
    const std::vector<std::pair<const char*, unsigned long long>> profile = {
        {"supported_spirv", properties.properties.apiVersion >= VK_API_VERSION_1_3 ? 0x00010600u : 0x00010400u},
        {"support_descriptor_aliasing", 1},
        {"support_int8", v12.shaderInt8},
        {"support_uniform_and_storage_buffer_8bit", v12.uniformAndStorageBuffer8BitAccess},
        {"support_storage_buffer_8bit", v12.storageBuffer8BitAccess},
        {"support_int16", features.features.shaderInt16},
        {"support_uniform_and_storage_buffer_16bit", v11.uniformAndStorageBuffer16BitAccess},
        {"support_storage_buffer_16bit", v11.storageBuffer16BitAccess},
        {"support_int64", features.features.shaderInt64},
        {"support_float_controls", has_extension(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME)},
        {"support_separate_denorm_behavior",
         v12p.denormBehaviorIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL},
        {"support_separate_rounding_mode",
         v12p.roundingModeIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL},
        {"support_fp16_denorm_preserve", v12p.shaderDenormPreserveFloat16},
        {"support_fp32_denorm_preserve", v12p.shaderDenormPreserveFloat32},
        {"support_fp16_denorm_flush", v12p.shaderDenormFlushToZeroFloat16},
        {"support_fp32_denorm_flush", v12p.shaderDenormFlushToZeroFloat32},
        {"support_fp16_signed_zero_nan_preserve", v12p.shaderSignedZeroInfNanPreserveFloat16},
        {"support_fp32_signed_zero_nan_preserve", v12p.shaderSignedZeroInfNanPreserveFloat32},
        {"support_fp64_signed_zero_nan_preserve", v12p.shaderSignedZeroInfNanPreserveFloat64},
        {"support_explicit_workgroup_layout", workgroup_layout},
        {"support_workgroup_layout_8bit_access",
         workgroup_layout && workgroup.workgroupMemoryExplicitLayout8BitAccess && v12.shaderInt8},
        {"support_workgroup_layout_16bit_access",
         workgroup_layout && workgroup.workgroupMemoryExplicitLayout16BitAccess},
        {"support_shader_quad_control",
         has_extension(VK_KHR_SHADER_QUAD_CONTROL_EXTENSION_NAME) && quad_control.shaderQuadControl},
        {"support_quad_shuffles", ops_quad},
        {"support_vote", ops_vote},
        {"support_shuffle_relative", ops_relative},
        {"supported_subgroup_stages", subgroup_stages},
        {"support_viewport_index_layer_non_geometry", layer},
        {"support_viewport_mask", 0},
        {"support_typeless_image_loads", features.features.shaderStorageImageReadWithoutFormat},
        {"support_demote_to_helper_invocation",
         has_extension(VK_EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_EXTENSION_NAME)},
        {"support_int64_atomics", atomic_int64},
        {"support_shared_int64_atomics", atomic_int64 && v12.shaderSharedInt64Atomics},
        {"support_derivative_control", 1},
        {"support_geometry_shader_passthrough", 0},
        {"support_native_ndc", has_extension(VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME)},
        {"support_scaled_attributes", 1},
        {"support_multi_viewport", features.features.multiViewport},
        {"support_geometry_streams", xfb.geometryStreams},
        {"support_sampled_image_array_nonuniform_indexing", v12.shaderSampledImageArrayNonUniformIndexing},
        {"support_storage_image_array_nonuniform_indexing", v12.shaderStorageImageArrayNonUniformIndexing},
        {"support_uniform_texel_buffer_array_nonuniform_indexing",
         v12.shaderUniformTexelBufferArrayNonUniformIndexing},
        {"support_storage_texel_buffer_array_nonuniform_indexing",
         v12.shaderStorageTexelBufferArrayNonUniformIndexing},
        {"warp_size_potentially_larger_than_guest",
         !has_extension(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME) || v13p.maxSubgroupSize > 32},
        {"need_gather_subpixel_offset", 1},
        {"has_broken_spirv_fragment_shuffle", 0},
        {"max_compute_shared_memory_size", limits.maxComputeSharedMemorySize},
        {"max_subgroup_size", max_subgroup},
        {"max_user_clip_distances", limits.maxClipDistances},
    };
    const auto print = [](const char* name, const auto& fields, const char* end) {
        std::printf("  \"%s\": {\n", name);
        for (size_t i = 0; i < fields.size(); ++i) {
            const auto& [key, value] = fields[i];
            const bool number = std::strncmp(key, "max_", 4) == 0 || std::strncmp(key, "min_", 4) == 0 ||
                                std::strncmp(key, "supported_", 10) == 0;
            std::printf("    \"%s\": %s%s\n", key,
                        number ? std::to_string(value).c_str() : (value ? "true" : "false"),
                        i + 1 < fields.size() ? "," : "");
        }
        std::printf("  }%s\n", end);
    };
    std::printf("{\n");
    print("host", host, ",");
    print("profile", profile, "");
    std::printf("}\n");
}

} // namespace

int main(int argc, char** argv) {
    const bool profile_only = argc == 2 && std::strcmp(argv[1], "--profile") == 0;
    if (argc != 3 && !profile_only) {
        std::fprintf(stderr, "usage: %s <description.txt> <output dir>\n       %s --profile\n", argv[0],
                     argv[0]);
        return 2;
    }
    const Description desc = profile_only ? Description{} : ReadDescription(argv[1]);
    const std::filesystem::path out_dir = profile_only ? "" : argv[2];
    if (!profile_only) {
        std::filesystem::create_directories(out_dir);
    }

    void* loader = dlopen("libvulkan.so.1", RTLD_NOW);
    if (!loader) {
        Fail("cannot load libvulkan.so.1");
    }
    get_instance_proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(loader, "vkGetInstanceProcAddr"));

    const VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "radv_pipe", 1, "lichii", 1,
                                VK_API_VERSION_1_3};
    const VkInstanceCreateInfo instance_ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &app, 0,
                                           nullptr, 0, nullptr};
    Check(FN(vkCreateInstance)(&instance_ci, nullptr, &instance), "vkCreateInstance");

    uint32_t count = 0;
    FN(vkEnumeratePhysicalDevices)(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> physicals(count);
    FN(vkEnumeratePhysicalDevices)(instance, &count, physicals.data());
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceSubgroupSizeControlProperties subgroup_control{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES, &driver};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &subgroup_control};
    for (const VkPhysicalDevice candidate : physicals) {
        FN(vkGetPhysicalDeviceProperties2)(candidate, &properties);
        if (driver.driverID == VK_DRIVER_ID_MESA_RADV) {
            physical = candidate;
            break;
        }
    }
    if (!physical) {
        Fail("no RADV device; source env.sh and run under the drm-shim (see README.md)");
    }

    FN(vkEnumerateDeviceExtensionProperties)(physical, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    FN(vkEnumerateDeviceExtensionProperties)(physical, nullptr, &count, available.data());
    const auto has_extension = [&](const char* name) {
        for (const auto& extension : available) {
            if (std::strcmp(extension.extensionName, name) == 0) {
                return true;
            }
        }
        return false;
    };
    if (profile_only) {
        PrintProfile(physical, has_extension);
        return 0;
    }
    if (!has_extension(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)) {
        Fail("the device lacks VK_KHR_pipeline_executable_properties");
    }

    std::vector<const char*> extensions;
    std::vector<std::vector<uint8_t>> feature_storage;
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    void** next = &features.pNext;
    for (const FeatureStruct& feature : feature_structs) {
        if (feature.extension) {
            if (!has_extension(feature.extension)) {
                continue;
            }
            extensions.push_back(feature.extension);
        }
        auto& storage = feature_storage.emplace_back(1024);
        auto* base = reinterpret_cast<VkBaseOutStructure*>(storage.data());
        base->sType = feature.type;
        *next = base;
        next = reinterpret_cast<void**>(&base->pNext);
    }
    for (const char* extension : plain_extensions) {
        if (has_extension(extension)) {
            extensions.push_back(extension);
        }
    }
    FN(vkGetPhysicalDeviceFeatures2)(physical, &features);

    // Lichii keeps robustBufferAccess and nullDescriptor but turns robustness2's access checks
    // off (Device::Device, VK_EXT_robustness2).
    if (!desc.robust_buffer_access) {
        features.features.robustBufferAccess = VK_FALSE;
    }
    for (auto* base = static_cast<VkBaseOutStructure*>(features.pNext); base; base = base->pNext) {
        if (base->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT) {
            auto* robustness2 = reinterpret_cast<VkPhysicalDeviceRobustness2FeaturesEXT*>(base);
            robustness2->robustBufferAccess2 = VK_FALSE;
            robustness2->robustImageAccess2 = VK_FALSE;
        }
    }

    const float priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_ci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, 0, 1,
                                           &priority};
    const VkDeviceCreateInfo device_ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                       &features,
                                       0,
                                       1,
                                       &queue_ci,
                                       0,
                                       nullptr,
                                       static_cast<uint32_t>(extensions.size()),
                                       extensions.data(),
                                       nullptr};
    Check(FN(vkCreateDevice)(physical, &device_ci, nullptr, &device), "vkCreateDevice");

    std::vector<VkShaderModule> modules;
    std::vector<VkSpecializationInfo> spec_infos(desc.stages.size());
    std::vector<VkPipelineShaderStageRequiredSubgroupSizeCreateInfo> subgroup_infos(desc.stages.size());
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    VkShaderStageFlags unsized_stages = 0;
    for (size_t i = 0; i < desc.stages.size(); ++i) {
        const Stage& stage = desc.stages[i];
        const std::vector<uint32_t> code = ReadSpirv(stage.path);
        const VkShaderModuleCreateInfo module_ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                                 code.size() * 4, code.data()};
        VkShaderModule module;
        Check(FN(vkCreateShaderModule)(device, &module_ci, nullptr, &module), "vkCreateShaderModule");
        modules.push_back(module);
        spec_infos[i] = {static_cast<uint32_t>(stage.spec_entries.size()), stage.spec_entries.data(),
                         stage.spec_data.size() * 4, stage.spec_data.data()};
        subgroup_infos[i] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, nullptr,
                             stage.subgroup_size};
        // Like Device::IsGuestWarpSizeSupported, a stage the driver can't size runs at its default.
        const bool sized = stage.subgroup_size && (subgroup_control.requiredSubgroupSizeStages & stage.stage);
        if (stage.subgroup_size && !sized) {
            unsized_stages |= stage.stage;
        }
        stages.push_back({VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, sized ? &subgroup_infos[i] : nullptr,
                          stage.full_subgroups ? VkPipelineShaderStageCreateFlags{
                                                     VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT}
                                               : VkPipelineShaderStageCreateFlags{0},
                          stage.stage, module, stage.entry.c_str(),
                          stage.spec_entries.empty() ? nullptr : &spec_infos[i]});
    }

    std::vector<VkDescriptorSetLayout> set_layouts;
    const uint32_t set_count = desc.sets.empty() ? 0 : desc.sets.rbegin()->first + 1;
    for (uint32_t set = 0; set < set_count; ++set) {
        const auto it = desc.sets.find(set);
        const auto* bindings = it == desc.sets.end() ? nullptr : &it->second;
        const VkDescriptorSetLayoutCreateInfo set_ci{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0,
            bindings ? static_cast<uint32_t>(bindings->size()) : 0, bindings ? bindings->data() : nullptr};
        VkDescriptorSetLayout set_layout;
        Check(FN(vkCreateDescriptorSetLayout)(device, &set_ci, nullptr, &set_layout),
              "vkCreateDescriptorSetLayout");
        set_layouts.push_back(set_layout);
    }
    const VkPipelineLayoutCreateInfo layout_ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                               nullptr,
                                               0,
                                               static_cast<uint32_t>(set_layouts.size()),
                                               set_layouts.data(),
                                               desc.push.size ? 1u : 0u,
                                               &desc.push};
    VkPipelineLayout layout;
    Check(FN(vkCreatePipelineLayout)(device, &layout_ci, nullptr, &layout), "vkCreatePipelineLayout");

    const VkPipelineCreateFlags flags = VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR |
                                        VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    VkPipeline pipeline;
    if (desc.compute) {
        const VkComputePipelineCreateInfo pipeline_ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                                      nullptr,
                                                      flags,
                                                      stages[0],
                                                      layout,
                                                      VK_NULL_HANDLE,
                                                      0};
        Check(FN(vkCreateComputePipelines)(device, VK_NULL_HANDLE, 1, &pipeline_ci, nullptr, &pipeline),
              "vkCreateComputePipelines");
    } else {
        bool has_mesh = false, has_tess = false, has_vertex = false;
        for (const Stage& stage : desc.stages) {
            has_mesh |= stage.stage == VK_SHADER_STAGE_MESH_BIT_EXT;
            has_tess |= stage.stage == VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
            has_vertex |= stage.stage == VK_SHADER_STAGE_VERTEX_BIT;
        }
        const bool dynamic_vertex =
            desc.dynamic_vertex_input && has_extension(VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME);
        std::vector<VkVertexInputBindingDescription> bindings;
        if (!dynamic_vertex) {
            for (const auto& attribute : desc.attributes) {
                bool known = false;
                for (const auto& binding : bindings) {
                    known |= binding.binding == attribute.binding;
                }
                if (!known) {
                    bindings.push_back({attribute.binding, 0, VK_VERTEX_INPUT_RATE_VERTEX});
                }
            }
        }
        const VkPipelineVertexInputStateCreateInfo vertex_input{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            nullptr,
            0,
            static_cast<uint32_t>(bindings.size()),
            bindings.data(),
            dynamic_vertex ? 0u : static_cast<uint32_t>(desc.attributes.size()),
            desc.attributes.data()};
        const VkPipelineInputAssemblyStateCreateInfo input_assembly{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0,
            has_tess ? VK_PRIMITIVE_TOPOLOGY_PATCH_LIST : desc.topology, VK_FALSE};
        const VkPipelineTessellationStateCreateInfo tessellation{
            VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO, nullptr, 0, desc.patch_points};
        const VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                                         nullptr, 0, 1, nullptr, 1, nullptr};
        const VkPipelineRasterizationStateCreateInfo rasterization{
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            nullptr,
            0,
            VK_FALSE,
            VK_FALSE,
            VK_POLYGON_MODE_FILL,
            VK_CULL_MODE_NONE,
            VK_FRONT_FACE_COUNTER_CLOCKWISE,
            VK_FALSE,
            0.0f,
            0.0f,
            0.0f,
            1.0f};
        const VkPipelineMultisampleStateCreateInfo multisample{
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, nullptr, 0, desc.samples, VK_FALSE, 0.0f,
            nullptr, VK_FALSE, VK_FALSE};
        const VkPipelineDepthStencilStateCreateInfo depth_stencil{
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        const std::vector<VkPipelineColorBlendAttachmentState> attachments(
            desc.colors.size(), VkPipelineColorBlendAttachmentState{
                                    VK_FALSE, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD,
                                    VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD, 0xf});
        const VkPipelineColorBlendStateCreateInfo color_blend{
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            nullptr,
            0,
            VK_FALSE,
            VK_LOGIC_OP_COPY,
            static_cast<uint32_t>(attachments.size()),
            attachments.data(),
            {}};

        // The dynamic states GraphicsPipeline::MakePipeline sets on a device with every extended
        // dynamic state extension, which RADV has.
        std::vector<VkDynamicState> dynamic_states{
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
            VK_DYNAMIC_STATE_DEPTH_BIAS,
            VK_DYNAMIC_STATE_BLEND_CONSTANTS,
            VK_DYNAMIC_STATE_DEPTH_BOUNDS,
            VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
            VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
            VK_DYNAMIC_STATE_STENCIL_REFERENCE,
            VK_DYNAMIC_STATE_LINE_WIDTH,
            VK_DYNAMIC_STATE_CULL_MODE,
            VK_DYNAMIC_STATE_FRONT_FACE,
            VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
            VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
            VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
            VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE,
            VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
            VK_DYNAMIC_STATE_STENCIL_OP,
            VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE,
            VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE,
        };
        if (!has_mesh) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY);
            dynamic_states.push_back(VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE);
            if (has_vertex) {
                dynamic_states.push_back(dynamic_vertex ? VK_DYNAMIC_STATE_VERTEX_INPUT_EXT
                                                        : VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE);
            }
        }
        if (has_extension(VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME)) {
            dynamic_states.push_back(VK_DYNAMIC_STATE_LOGIC_OP_EXT);
        }
        if (has_extension(VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME)) {
            for (const VkDynamicState state :
                 {VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT, VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT,
                  VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT, VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT,
                  VK_DYNAMIC_STATE_LOGIC_OP_ENABLE_EXT, VK_DYNAMIC_STATE_LINE_RASTERIZATION_MODE_EXT,
                  VK_DYNAMIC_STATE_CONSERVATIVE_RASTERIZATION_MODE_EXT, VK_DYNAMIC_STATE_LINE_STIPPLE_ENABLE_EXT,
                  VK_DYNAMIC_STATE_ALPHA_TO_COVERAGE_ENABLE_EXT, VK_DYNAMIC_STATE_ALPHA_TO_ONE_ENABLE_EXT}) {
                dynamic_states.push_back(state);
            }
        }
        const VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                                       nullptr, 0, static_cast<uint32_t>(dynamic_states.size()),
                                                       dynamic_states.data()};
        const VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                                                      nullptr,
                                                      0,
                                                      static_cast<uint32_t>(desc.colors.size()),
                                                      desc.colors.data(),
                                                      desc.depth,
                                                      VK_FORMAT_UNDEFINED};
        const VkGraphicsPipelineCreateInfo pipeline_ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                                                       &rendering,
                                                       flags,
                                                       static_cast<uint32_t>(stages.size()),
                                                       stages.data(),
                                                       has_mesh ? nullptr : &vertex_input,
                                                       has_mesh ? nullptr : &input_assembly,
                                                       has_tess ? &tessellation : nullptr,
                                                       &viewport,
                                                       &rasterization,
                                                       &multisample,
                                                       &depth_stencil,
                                                       &color_blend,
                                                       &dynamic,
                                                       layout,
                                                       VK_NULL_HANDLE,
                                                       0,
                                                       VK_NULL_HANDLE,
                                                       0};
        Check(FN(vkCreateGraphicsPipelines)(device, VK_NULL_HANDLE, 1, &pipeline_ci, nullptr, &pipeline),
              "vkCreateGraphicsPipelines");
    }

    const auto get_device_proc = FN(vkGetDeviceProcAddr);
    const auto get_executables = reinterpret_cast<PFN_vkGetPipelineExecutablePropertiesKHR>(
        get_device_proc(device, "vkGetPipelineExecutablePropertiesKHR"));
    const auto get_statistics = reinterpret_cast<PFN_vkGetPipelineExecutableStatisticsKHR>(
        get_device_proc(device, "vkGetPipelineExecutableStatisticsKHR"));
    const auto get_representations = reinterpret_cast<PFN_vkGetPipelineExecutableInternalRepresentationsKHR>(
        get_device_proc(device, "vkGetPipelineExecutableInternalRepresentationsKHR"));

    const VkPipelineInfoKHR pipeline_info{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, nullptr, pipeline};
    get_executables(device, &pipeline_info, &count, nullptr);
    std::vector<VkPipelineExecutablePropertiesKHR> executables(
        count, VkPipelineExecutablePropertiesKHR{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
    Check(get_executables(device, &pipeline_info, &count, executables.data()),
          "vkGetPipelineExecutablePropertiesKHR");

    std::string json = "{\"status\": \"ok\", \"device\": " + JsonString(properties.properties.deviceName) +
                       ", \"driver\": " + JsonString(driver.driverInfo) + ", \"subgroup_sizes\": [" +
                       std::to_string(subgroup_control.minSubgroupSize) + ", " +
                       std::to_string(subgroup_control.maxSubgroupSize) + "], \"sizable_stages\": " +
                       std::to_string(subgroup_control.requiredSubgroupSizeStages) +
                       ", \"unsized_stages\": " + std::to_string(unsized_stages) + ", \"executables\": [";
    for (uint32_t index = 0; index < count; ++index) {
        const auto& executable = executables[index];
        const std::string prefix = std::to_string(index) + "-" + FileName(executable.name);
        const VkPipelineExecutableInfoKHR info{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, nullptr, pipeline,
                                               index};
        json += std::string(index ? ", " : "") + "{\"name\": " + JsonString(executable.name) +
                ", \"stages\": " + std::to_string(executable.stages) +
                ", \"subgroup_size\": " + std::to_string(executable.subgroupSize) + ", \"statistics\": {";

        uint32_t stat_count = 0;
        get_statistics(device, &info, &stat_count, nullptr);
        std::vector<VkPipelineExecutableStatisticKHR> statistics(
            stat_count, VkPipelineExecutableStatisticKHR{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
        get_statistics(device, &info, &stat_count, statistics.data());
        for (uint32_t i = 0; i < stat_count; ++i) {
            const auto& stat = statistics[i];
            std::string value;
            switch (stat.format) {
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
                value = stat.value.b32 ? "true" : "false";
                break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
                value = std::to_string(stat.value.i64);
                break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
                value = std::to_string(stat.value.u64);
                break;
            default:
                value = std::to_string(stat.value.f64);
            }
            json += std::string(i ? ", " : "") + JsonString(stat.name) + ": " + value;
        }
        json += "}, \"files\": [";

        uint32_t repr_count = 0;
        get_representations(device, &info, &repr_count, nullptr);
        std::vector<VkPipelineExecutableInternalRepresentationKHR> representations(
            repr_count, VkPipelineExecutableInternalRepresentationKHR{
                            VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR});
        get_representations(device, &info, &repr_count, representations.data());
        std::vector<std::vector<char>> data(repr_count);
        for (uint32_t i = 0; i < repr_count; ++i) {
            data[i].resize(representations[i].dataSize);
            representations[i].pData = data[i].data();
        }
        get_representations(device, &info, &repr_count, representations.data());
        for (uint32_t i = 0; i < repr_count; ++i) {
            const auto& repr = representations[i];
            const std::string name = prefix + "." + FileName(repr.name) + (repr.isText ? ".txt" : ".bin");
            size_t size = repr.dataSize;
            if (repr.isText && size && data[i][size - 1] == '\0') {
                --size;
            }
            std::ofstream(out_dir / name, std::ios::binary)
                .write(data[i].data(), static_cast<std::streamsize>(size));
            json += std::string(i ? ", " : "") + JsonString(name);
        }
        json += "]}";
    }
    json += "]}";
    std::printf("%s\n", json.c_str());

    FN(vkDestroyPipeline)(device, pipeline, nullptr);
    FN(vkDestroyPipelineLayout)(device, layout, nullptr);
    for (const auto set_layout : set_layouts) {
        FN(vkDestroyDescriptorSetLayout)(device, set_layout, nullptr);
    }
    for (const auto module : modules) {
        FN(vkDestroyShaderModule)(device, module, nullptr);
    }
    FN(vkDestroyDevice)(device, nullptr);
    FN(vkDestroyInstance)(instance, nullptr);
    return 0;
}
