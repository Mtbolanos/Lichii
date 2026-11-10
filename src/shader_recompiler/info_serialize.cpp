// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "shader_recompiler/info_serialize.h"

#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace Shader {
namespace {

// Every field of Info is listed in VisitInfo below. An info that grows a field has to be listed
// there too: this size check fails the build until it is (update the size with it).
// Bump Magic in vk_precompiled_kept.cpp when this layout changes.
#if defined(_MSC_VER) && defined(_WIN64)
static_assert(sizeof(Info) == 6960,
              "Shader::Info changed: list the new fields in VisitInfo, then update this size");
// The descriptors are held in vectors, so a field they grow does not change sizeof(Info).
static_assert(sizeof(ConstantBufferDescriptor) == 8 && sizeof(StorageBufferDescriptor) == 80 &&
                  sizeof(TextureBufferDescriptor) == 40 && sizeof(ImageBufferDescriptor) == 28 &&
                  sizeof(TextureDescriptor) == 44 && sizeof(ImageDescriptor) == 36,
              "A Shader::Info descriptor changed: list the new fields in its Visit function below, "
              "then update these sizes and Magic (vk_precompiled_kept.cpp)");
static_assert(sizeof(StorageWriteExtent) == 60,
              "StorageWriteExtent changed: list the new fields in VisitWriteExtent below");
#endif

struct Writer {
    std::vector<u8>& out;

    template <typename T>
        requires(std::is_arithmetic_v<T> || std::is_enum_v<T>)
    void operator()(const T& value) {
        const auto* const bytes = reinterpret_cast<const u8*>(&value);
        out.insert(out.end(), bytes, bytes + sizeof(T));
    }
    template <typename T, size_t N>
    void operator()(const std::array<T, N>& values) {
        for (const T& value : values) {
            (*this)(value);
        }
    }
    template <size_t N>
    void operator()(const std::bitset<N>& bits) {
        for (size_t word = 0; word < (N + 63) / 64; ++word) {
            u64 value = 0;
            for (size_t bit = 0; bit < 64 && word * 64 + bit < N; ++bit) {
                value |= u64{bits[word * 64 + bit]} << bit;
            }
            (*this)(value);
        }
    }
    void operator()(const VaryingState& state) {
        (*this)(state.mask);
    }
    template <typename K, typename V>
    void operator()(const std::map<K, V>& map) {
        (*this)(static_cast<u32>(map.size()));
        for (const auto& [key, value] : map) {
            (*this)(key);
            (*this)(value);
        }
    }
    void operator()(const std::vector<std::vector<u32>>& lists) {
        (*this)(static_cast<u32>(lists.size()));
        for (const auto& list : lists) {
            (*this)(static_cast<u32>(list.size()));
            for (const u32 value : list) {
                (*this)(value);
            }
        }
    }
    template <typename Container>
        requires requires(const Container& c) { c.begin()->operator<=>(*c.begin()); }
    void operator()(const Container& descriptors) {
        (*this)(static_cast<u32>(descriptors.size()));
        for (const auto& descriptor : descriptors) {
            VisitDescriptor(*this, descriptor);
        }
    }
    template <typename Descriptor>
    static void VisitDescriptor(Writer& io, const Descriptor& descriptor);
};

struct Reader {
    std::span<const u8>& in;

    template <typename T>
        requires(std::is_arithmetic_v<T> || std::is_enum_v<T>)
    void operator()(T& value) {
        if (in.size() < sizeof(T)) {
            throw std::out_of_range{"shader info"};
        }
        std::memcpy(&value, in.data(), sizeof(T));
        in = in.subspan(sizeof(T));
    }
    template <typename T, size_t N>
    void operator()(std::array<T, N>& values) {
        for (T& value : values) {
            (*this)(value);
        }
    }
    template <size_t N>
    void operator()(std::bitset<N>& bits) {
        for (size_t word = 0; word < (N + 63) / 64; ++word) {
            u64 value = 0;
            (*this)(value);
            for (size_t bit = 0; bit < 64 && word * 64 + bit < N; ++bit) {
                bits[word * 64 + bit] = ((value >> bit) & 1) != 0;
            }
        }
    }
    void operator()(VaryingState& state) {
        (*this)(state.mask);
    }
    template <typename K, typename V>
    void operator()(std::map<K, V>& map) {
        u32 count{};
        (*this)(count);
        map.clear();
        for (u32 i = 0; i < count; ++i) {
            K key{};
            V value{};
            (*this)(key);
            (*this)(value);
            map.emplace(key, value);
        }
    }
    void operator()(std::vector<std::vector<u32>>& lists) {
        u32 count{};
        (*this)(count);
        if (count > in.size()) {
            throw std::out_of_range{"shader info"};
        }
        lists.assign(count, {});
        for (auto& list : lists) {
            u32 size{};
            (*this)(size);
            if (size > in.size() / sizeof(u32)) {
                throw std::out_of_range{"shader info"};
            }
            list.resize(size);
            for (u32& value : list) {
                (*this)(value);
            }
        }
    }
    template <typename Container>
        requires requires(Container& c) { c.begin()->operator<=>(*c.begin()); }
    void operator()(Container& descriptors) {
        u32 count{};
        (*this)(count);
        if (count > descriptors.max_size() || count > in.size()) {
            throw std::out_of_range{"shader info"};
        }
        descriptors.clear();
        for (u32 i = 0; i < count; ++i) {
            auto& descriptor = descriptors.emplace_back();
            VisitDescriptor(*this, descriptor);
        }
    }
    template <typename Descriptor>
    static void VisitDescriptor(Reader& io, Descriptor& descriptor);
};

template <typename Io, typename D>
void VisitConstantBuffer(Io& io, D& d) {
    io(d.index);
    io(d.count);
}
template <typename Io, typename E>
void VisitWriteExtent(Io& io, E& e) {
    io(e.bounded);
    io(e.guarded);
    io(e.const_end);
    io(e.local_stride);
    io(e.group_stride);
    io(e.guard_const);
    io(e.guard_local_stride);
    io(e.guard_group_stride);
}
template <typename Io, typename D>
void VisitStorageBuffer(Io& io, D& d) {
    io(d.cbuf_index);
    io(d.cbuf_offset);
    io(d.count);
    io(d.is_written);
    io(d.indirect_offset);
    VisitWriteExtent(io, d.write_extent);
}
template <typename Io, typename D>
void VisitTextureBuffer(Io& io, D& d) {
    io(d.has_secondary);
    io(d.cbuf_index);
    io(d.cbuf_offset);
    io(d.shift_left);
    io(d.secondary_cbuf_index);
    io(d.secondary_cbuf_offset);
    io(d.secondary_shift_left);
    io(d.count);
    io(d.size_shift);
    io(d.handle_program);
}
template <typename Io, typename D>
void VisitImageBuffer(Io& io, D& d) {
    io(d.format);
    io(d.is_written);
    io(d.is_read);
    io(d.is_integer);
    io(d.cbuf_index);
    io(d.cbuf_offset);
    io(d.count);
    io(d.size_shift);
    io(d.handle_program);
}
template <typename Io, typename D>
void VisitTexture(Io& io, D& d) {
    io(d.type);
    io(d.is_depth);
    io(d.is_multisample);
    io(d.is_integer);
    io(d.has_secondary);
    io(d.cbuf_index);
    io(d.cbuf_offset);
    io(d.shift_left);
    io(d.secondary_cbuf_index);
    io(d.secondary_cbuf_offset);
    io(d.secondary_shift_left);
    io(d.count);
    io(d.size_shift);
    io(d.handle_program);
}
template <typename Io, typename D>
void VisitImage(Io& io, D& d) {
    io(d.type);
    io(d.format);
    io(d.is_written);
    io(d.is_read);
    io(d.is_integer);
    io(d.cbuf_index);
    io(d.cbuf_offset);
    io(d.count);
    io(d.size_shift);
    io(d.is_byte_addressed);
    io(d.handle_program);
}

template <typename Io, typename D>
void VisitAnyDescriptor(Io& io, D& d) {
    using T = std::remove_const_t<D>;
    if constexpr (std::is_same_v<T, ConstantBufferDescriptor>) {
        VisitConstantBuffer(io, d);
    } else if constexpr (std::is_same_v<T, StorageBufferDescriptor>) {
        VisitStorageBuffer(io, d);
    } else if constexpr (std::is_same_v<T, StorageWriteExtent>) {
        VisitWriteExtent(io, d);
    } else if constexpr (std::is_same_v<T, TextureBufferDescriptor>) {
        VisitTextureBuffer(io, d);
    } else if constexpr (std::is_same_v<T, ImageBufferDescriptor>) {
        VisitImageBuffer(io, d);
    } else if constexpr (std::is_same_v<T, TextureDescriptor>) {
        VisitTexture(io, d);
    } else {
        static_assert(std::is_same_v<T, ImageDescriptor>);
        VisitImage(io, d);
    }
}

template <typename Descriptor>
void Writer::VisitDescriptor(Writer& io, const Descriptor& descriptor) {
    VisitAnyDescriptor(io, descriptor);
}
template <typename Descriptor>
void Reader::VisitDescriptor(Reader& io, Descriptor& descriptor) {
    VisitAnyDescriptor(io, descriptor);
}

template <typename Io, typename I>
void VisitInfo(Io& io, I& info) {
    io(info.is_task);
    io(info.is_mesh);
    io(info.uses_workgroup_id);
    io(info.uses_local_invocation_id);
    io(info.uses_invocation_id);
    io(info.uses_invocation_info);
    io(info.uses_sample_id);
    io(info.uses_is_helper_invocation);
    io(info.uses_subgroup_invocation_id);
    io(info.uses_subgroup_shuffles);
    io(info.uses_quad_shuffles);
    io(info.uses_patches);
    io(info.interpolation);
    io(info.loads);
    io(info.stores);
    io(info.passthrough);
    io(info.legacy_stores_mapping);
    io(info.loads_indexed_attributes);
    io(info.stores_frag_color);
    io(info.frag_color_components);
    io(info.stores_sample_mask);
    io(info.stores_frag_depth);
    io(info.stores_tess_level_outer);
    io(info.stores_tess_level_inner);
    io(info.stores_indexed_attributes);
    io(info.stores_global_memory);
    io(info.uses_local_memory);
    io(info.uses_fp16);
    io(info.uses_fp64);
    io(info.uses_fp16_denorms_flush);
    io(info.uses_fp16_denorms_preserve);
    io(info.uses_fp32_denorms_flush);
    io(info.uses_fp32_denorms_preserve);
    io(info.uses_int8);
    io(info.uses_int16);
    io(info.uses_int64);
    io(info.uses_image_1d);
    io(info.uses_sampled_1d);
    io(info.uses_sparse_residency);
    io(info.uses_demote_to_helper_invocation);
    io(info.uses_subgroup_vote);
    io(info.uses_subgroup_mask);
    io(info.uses_fswzadd);
    io(info.uses_derivatives);
    io(info.uses_typeless_image_reads);
    io(info.uses_typeless_image_writes);
    io(info.uses_image_buffers);
    io(info.uses_shared_increment);
    io(info.uses_shared_decrement);
    io(info.uses_global_increment);
    io(info.uses_global_decrement);
    io(info.uses_atomic_f32_add);
    io(info.uses_atomic_f16x2_add);
    io(info.uses_atomic_f16x2_min);
    io(info.uses_atomic_f16x2_max);
    io(info.uses_atomic_f32x2_add);
    io(info.uses_atomic_f32x2_min);
    io(info.uses_atomic_f32x2_max);
    io(info.uses_atomic_s32_min);
    io(info.uses_atomic_s32_max);
    io(info.uses_int64_bit_atomics);
    io(info.uses_global_memory);
    io(info.requires_coherent_global_memory);
    io(info.requires_coherent_image_memory);
    io(info.uses_cooperative_matrix);
    io(info.native_global_bounds);
    io(info.uses_atomic_image_u32);
    io(info.uses_shadow_lod);
    io(info.uses_rescaling_uniform);
    io(info.uses_cbuf_indirect);
    io(info.uses_render_area);
    io(info.used_constant_buffer_types);
    io(info.used_storage_buffer_types);
    io(info.used_indirect_cbuf_types);
    io(info.constant_buffer_mask);
    io(info.constant_buffer_used_sizes);
    io(info.nvn_buffer_base);
    io(info.nvn_buffer_used);
    io(info.requires_layer_emulation);
    io(info.emulated_layer);
    io(info.used_clip_distances);
    io(info.constant_buffer_descriptors);
    io(info.storage_buffers_descriptors);
    io(info.storage_read_extents);
    io(info.texture_buffer_descriptors);
    io(info.image_buffer_descriptors);
    io(info.texture_descriptors);
    io(info.handle_programs);
    io(info.image_descriptors);
}

} // Anonymous namespace

void SerializeInfo(const Info& info, std::vector<u8>& out) {
    Writer writer{out};
    VisitInfo(writer, info);
}

bool DeserializeInfo(std::span<const u8>& in, Info& info) {
    try {
        Reader reader{in};
        VisitInfo(reader, info);
        return true;
    } catch (const std::out_of_range&) {
        return false;
    }
}

} // namespace Shader
