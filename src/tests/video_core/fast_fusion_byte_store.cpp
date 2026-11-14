// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#ifdef HAS_OPENGL
#include "shader_recompiler/backend/glasm/emit_glasm.h"
#include "shader_recompiler/backend/glsl/emit_glsl.h"
#endif
#include "shader_recompiler/byte_addressed_image.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/impl.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace {
using namespace Shader;
using Ampere::Instruction;
constexpr Instruction Store{{0x06007b9d, 0x60009917, 0x0010a900, 0x0001e200}};
constexpr Instruction Exit{{0x794d, 0, 0x03800000, 0}};
constexpr u64 MaxwellStore{0xeb38099600c70617ULL};

class ByteStoreEnvironment final : public Environment {
public:
    explicit ByteStoreEnvironment(Stage stage_ = Stage::Compute) { stage = stage_; }
    bool native{true};
    TexturePixelFormat format{TexturePixelFormat::R32_UINT};
    TextureType type{TextureType::Color2D};

    u64 ReadInstruction(u32) override { throw LogicError("Unexpected instruction read"); }
    u32 ReadCbufValue(u32 index, u32 offset) override {
        REQUIRE(index == 0);
        REQUIRE(offset == 0x264);
        return 0x99;
    }
    TextureType ReadTextureType(u32) override { return type; }
    TexturePixelFormat ReadTexturePixelFormat(u32) override { return format; }
    bool IsTexturePixelFormatInteger(u32) override { return format == TexturePixelFormat::R32_UINT; }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return 0; }
    u32 LocalMemorySize() const override { return 0; }
    u32 SharedMemorySize() const override { return 0; }
    std::array<u32, 3> WorkgroupSize() const override { return {32, 1, 1}; }
    bool HasHLEMacroState() const override { return false; }
    bool HasNativeInstructions() const override { return native; }
    std::optional<ReplaceConstant> GetReplaceConstBuffer(u32, u32) override { return std::nullopt; }
    void Dump(u64, u64) override {}
};

struct ByteStoreProgram {
    ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::Program program;
    ByteStoreEnvironment env;
    Profile profile{};

    explicit ByteStoreProgram(Stage stage = Stage::Compute) : env{stage} {
        program.stage = Stage::Compute;
        program.workgroup_size = {32, 1, 1};
        program.blocks = {&block};
        program.post_order_blocks = {&block};
        program.syntax_list = {{{.block = &block}, IR::AbstractSyntaxNode::Type::Block},
                               {{}, IR::AbstractSyntaxNode::Type::Return}};
        profile.supported_spirv = 0x00010300;
    }

    IR::Inst& Write(u32 byte_x = 0xfffffffc, bool coherent = true) {
        IR::IREmitter ir{block};
        IR::TextureInstInfo flags{};
        flags.type.Assign(TextureType::Color2D);
        flags.image_format.Assign(ImageFormat::R32_UINT);
        flags.image_byte_address.Assign(1);
        flags.image_coherent.Assign(coherent);
        const auto coords{ir.CompositeConstruct(ir.Imm32(byte_x), ir.Imm32(7))};
        const auto color{ir.CompositeConstruct(ir.Imm32(0x7fc01234U), ir.Imm32(0), ir.Imm32(0),
                                               ir.Imm32(0))};
        ir.ImageWrite(ir.Imm32(0x264), coords, color, flags);
        return block.back();
    }

    void Resolve() {
        HostTranslateInfo host;
        host.ApplyDescriptorLimitPolicy();
        Optimization::TexturePass(env, program, host);
        Optimization::CollectShaderInfoPass(env, program);
    }
};

std::vector<std::vector<u32>> Instructions(const std::vector<u32>& code, u32 opcode) {
    std::vector<std::vector<u32>> result;
    for (size_t i = 5; i < code.size();) {
        const u32 count{code[i] >> 16};
        REQUIRE(count > 0);
        REQUIRE(i + count <= code.size());
        if ((code[i] & 0xffff) == opcode) {
            result.emplace_back(code.begin() + i + 1, code.begin() + i + count);
        }
        i += count;
    }
    return result;
}
} // namespace

TEST_CASE("R16 BA translation requires an explicit native compute backend capability",
          "[video_core][packed_surface]") {
    for (const bool capability : {false, true}) {
        for (const bool native : {false, true}) {
            for (const auto stage : {Stage::VertexB, Stage::Fragment, Stage::Compute}) {
                CAPTURE(capability, native, stage);
                ByteStoreProgram fixture{stage};
                fixture.program.stage = stage;
                fixture.env.native = native;
                fixture.env.format = TexturePixelFormat::R16_FLOAT;
                auto& store{fixture.Write()};
                HostTranslateInfo host;
                CHECK_FALSE(host.support_byte_addressed_r16);
                host.support_byte_addressed_r16 = capability;
                host.ApplyDescriptorLimitPolicy();
                if (!capability || !native || stage != Stage::Compute) {
                    CHECK_THROWS_AS(Optimization::TexturePass(fixture.env, fixture.program, host),
                                    NotImplementedException);
                    continue;
                }
                REQUIRE_NOTHROW(Optimization::TexturePass(fixture.env, fixture.program, host));
                REQUIRE(fixture.program.info.image_descriptors.size() == 1);
                const auto& desc{fixture.program.info.image_descriptors.front()};
                CHECK(desc.is_byte_addressed);
                CHECK(desc.is_written);
                CHECK_FALSE(desc.is_read);
                CHECK(desc.format == ImageFormat::R32_UINT);
                const auto* x{store.Arg(1).Inst()->Arg(0).Inst()};
                REQUIRE(x->GetOpcode() == IR::Opcode::ShiftRightArithmetic32);
                CHECK(x->Arg(1).U32() == 2);
                CHECK(store.Arg(2).Inst()->Arg(0).U32() == 0x7fc01234U);
            }
        }
    }
}

TEST_CASE("Logged SUST.D.BA lowers without becoming a formatted surface store",
          "[video_core][fast_fusion_byte_store]") {
    for (u32 predicate = 0; predicate < 16; ++predicate) {
        CAPTURE(predicate);
        auto store{Store};
        store.words[0] = (store.words[0] & ~0xf000U) | (predicate << 12);
        const auto lowered{Ampere::LowerToMaxwell(std::array{store, Exit})};
        REQUIRE(lowered);
        if (predicate == 15) {
            REQUIRE(lowered.instructions.size() == 1);
            continue;
        }
        REQUIRE(lowered.required_texture_buffer == 0);
        REQUIRE(lowered.instructions.size() == 2);
        CHECK(lowered.instructions.front() ==
              ((MaxwellStore & ~(u64{15} << 16)) | (u64{predicate} << 16)));
    }
}

TEST_CASE("Byte-addressed native lowering explicitly rejects unproven modes",
          "[video_core][fast_fusion_byte_store]") {
    for (u32 bit = 0; bit < 32; ++bit) {
        auto store{Store};
        store.words[2] ^= 1U << bit;
        const auto lowered{Ampere::LowerToMaxwell(std::array{store, Exit})};
        CAPTURE(bit);
        CHECK_FALSE(lowered);
        CHECK(lowered.instructions.empty());
    }
    for (u32 bit : {16U, 53U, 59U, 60U, 61U, 62U, 63U}) {
        auto store{Store};
        store.words[bit / 32] ^= 1U << (bit % 32);
        CAPTURE(bit);
        CHECK_FALSE(Ampere::LowerToMaxwell(std::array{store, Exit}));
    }
    auto binding{Store};
    binding.words[1] |= 18U << 22;
    CHECK_FALSE(Ampere::LowerToMaxwell(std::array{binding, Exit}));
    for (u32 reg : {246U, 254U, 255U}) {
        auto store{Store};
        store.words[1] = (store.words[1] & ~255U) | reg;
        CHECK_FALSE(Ampere::LowerToMaxwell(std::array{store, Exit}));
    }
}

TEST_CASE("BA translation is native compute B32 only and reads exactly one raw register",
          "[video_core][fast_fusion_byte_store]") {
    ByteStoreProgram fixture;
    Maxwell::TranslatorVisitor visitor{fixture.env, fixture.block};
    REQUIRE_NOTHROW(visitor.SUST(MaxwellStore));
    const auto& store{fixture.block.back()};
    REQUIRE(store.GetOpcode() == IR::Opcode::BoundImageWrite);
    CHECK(store.Arg(0).U32() == 0x264);
    const auto flags{store.Flags<IR::TextureInstInfo>()};
    CHECK(flags.image_byte_address == 1);
    CHECK(flags.image_coherent == 1);
    CHECK(flags.image_format == ImageFormat::R32_UINT);
    const auto* color{store.Arg(2).Inst()};
    REQUIRE(color->GetOpcode() == IR::Opcode::CompositeConstructU32x4);
    REQUIRE(color->Arg(0).Inst()->GetOpcode() == IR::Opcode::GetRegister);
    CHECK(color->Arg(0).Inst()->Arg(0).Reg() == IR::Reg::R23);
    for (size_t i = 1; i < 4; ++i) CHECK(color->Arg(i).U32() == 0);
    size_t reads{};
    for (const auto& inst : fixture.block) {
        if (inst.GetOpcode() == IR::Opcode::GetRegister) ++reads;
        CHECK(inst.GetOpcode() != IR::Opcode::BoundImageRead);
        CHECK(inst.GetOpcode() != IR::Opcode::BindlessImageRead);
    }
    CHECK(reads == 3);
    for (const bool native : {false, true}) {
        for (const auto stage : {Stage::Compute, Stage::Fragment}) {
            if (native && stage == Stage::Compute) continue;
            ByteStoreProgram rejected{stage};
            rejected.env.native = native;
            Maxwell::TranslatorVisitor rejected_visitor{rejected.env, rejected.block};
            CHECK_THROWS_AS(rejected_visitor.SUST(MaxwellStore), NotImplementedException);
        }
    }
    for (u64 bit : {24U, 25U, 20U, 21U, 22U, 33U, 34U, 35U, 49U, 50U, 51U}) {
        ByteStoreProgram rejected;
        Maxwell::TranslatorVisitor rejected_visitor{rejected.env, rejected.block};
        CHECK_THROWS_AS(rejected_visitor.SUST(MaxwellStore ^ (u64{1} << bit)),
                        NotImplementedException);
    }
}

TEST_CASE("BA descriptors require a compatible 32-bit color image before signed coordinate scaling",
          "[video_core][fast_fusion_byte_store]") {
    constexpr std::array supported_formats{
        TexturePixelFormat::R32_UINT, TexturePixelFormat::R32_SINT, TexturePixelFormat::R32_FLOAT,
        TexturePixelFormat::R16G16_UNORM, TexturePixelFormat::R16G16_FLOAT,
        TexturePixelFormat::R16G16_UINT, TexturePixelFormat::R16G16_SINT,
        TexturePixelFormat::R16G16_SNORM, TexturePixelFormat::A8B8G8R8_UNORM,
        TexturePixelFormat::A8B8G8R8_SNORM, TexturePixelFormat::A8B8G8R8_SINT,
        TexturePixelFormat::A8B8G8R8_UINT,
    };
    for (u32 value = 0; value <= static_cast<u32>(TexturePixelFormat::D32_FLOAT_S8_UINT); ++value) {
        const auto format{static_cast<TexturePixelFormat>(value)};
        const bool supported{std::ranges::find(supported_formats, format) != supported_formats.end()};
        CAPTURE(value);
        CHECK(IsByteAddressed32BitColorFormat(format) == supported);
        ByteStoreProgram fixture;
        fixture.env.format = format;
        auto& store{fixture.Write()};
        if (!supported) {
            CHECK_THROWS_AS(fixture.Resolve(), NotImplementedException);
            continue;
        }
        REQUIRE_NOTHROW(fixture.Resolve());
        REQUIRE(fixture.program.info.image_descriptors.size() == 1);
        const auto& desc{fixture.program.info.image_descriptors.front()};
        CHECK(desc.is_byte_addressed);
        CHECK(desc.is_integer);
        CHECK(desc.format == ImageFormat::R32_UINT);
        CHECK(fixture.program.info.requires_coherent_image_memory);
        CHECK_FALSE(fixture.program.info.requires_coherent_global_memory);
        const auto* coords{store.Arg(1).Inst()};
        REQUIRE(coords->GetOpcode() == IR::Opcode::CompositeConstructU32x2);
        const auto* x{coords->Arg(0).Inst()};
        REQUIRE(x->GetOpcode() == IR::Opcode::ShiftRightArithmetic32);
        CHECK(x->Arg(1).U32() == 2);
        CHECK(store.Arg(2).Inst()->Arg(0).U32() == 0x7fc01234U);
        Optimization::ConstantPropagationPass(fixture.env, fixture.program);
        CHECK(store.Arg(1).Inst()->Arg(0).U32() == 0xffffffffU);
        CHECK(store.Arg(1).Inst()->Arg(1).U32() == 7);
    }
    for (const auto type : {TextureType::Buffer, TextureType::Color1D, TextureType::ColorArray2D,
                            TextureType::Color3D, TextureType::ColorCube}) {
        ByteStoreProgram fixture;
        fixture.env.type = type;
        fixture.Write();
        CHECK_THROWS_AS(fixture.Resolve(), NotImplementedException);
    }
}

TEST_CASE("Image descriptor merging preserves BA validation and coherent memory independently",
          "[video_core][fast_fusion_byte_store]") {
    for (const bool first : {false, true}) {
        for (const bool second : {false, true}) {
            Info a{}, b{};
            a.requires_coherent_image_memory = first;
            b.requires_coherent_image_memory = second;
            a.requires_coherent_global_memory = true;
            a.image_descriptors.push_back({.type = TextureType::Color2D,
                                           .format = ImageFormat::R32_UINT,
                                           .is_read = true,
                                           .count = 1,
                                           .is_byte_addressed = first});
            b.image_descriptors.push_back({.type = TextureType::Color2D,
                                           .format = ImageFormat::R32_UINT,
                                           .is_written = true,
                                           .count = 1,
                                           .is_byte_addressed = second});
            Optimization::JoinTextureInfo(a, b);
            REQUIRE(a.image_descriptors.size() == 1);
            CHECK(a.image_descriptors.front().is_byte_addressed == (first || second));
            CHECK(a.image_descriptors.front().is_read);
            CHECK(a.image_descriptors.front().is_written);
            CHECK(a.requires_coherent_image_memory == (first || second));
            CHECK(a.requires_coherent_global_memory);
        }
    }
}

TEST_CASE("Image backends preserve raw B32 storage and add one conditional image fence",
          "[video_core][fast_fusion_byte_store]") {
#ifdef HAS_OPENGL
    constexpr std::array<unsigned, 3> backends{0U, 1U, 2U};
#else
    constexpr std::array<unsigned, 1> backends{0U};
#endif
    for (const bool image_coherent : {false, true}) {
        for (const bool global_coherent : {false, true}) {
            for (const unsigned backend : backends) {
                ByteStoreProgram fixture;
                IR::IREmitter ir{fixture.block};
                IR::TextureInstInfo read_flags{};
                read_flags.type.Assign(TextureType::Color2D);
                read_flags.image_format.Assign(ImageFormat::R32_UINT);
                read_flags.image_coherent.Assign(image_coherent);
                const auto read{ir.ImageRead(ir.Imm32(0x264),
                                             ir.CompositeConstruct(ir.Imm32(0), ir.Imm32(0)),
                                             read_flags)};
                auto& store{fixture.Write(4, false)};
                store.SetArg(2, read);
                fixture.Resolve();
                CHECK(fixture.program.info.requires_coherent_image_memory == image_coherent);
                fixture.program.info.requires_coherent_global_memory = global_coherent;
                ir.Barrier();
                if (backend == 0) {
                    const auto spirv{Backend::SPIRV::EmitSPIRV(fixture.profile, fixture.program)};
                    std::set<u32> coherent_variables;
                    for (const auto& op : Instructions(spirv, 71)) {
                        if (op.size() >= 2 && op[1] == 23) coherent_variables.insert(op[0]);
                    }
                    CHECK(coherent_variables.size() == (image_coherent ? 1U : 0U));
                    REQUIRE(Instructions(spirv, 99).size() == 1);
                    std::map<u32, u32> constants;
                    for (const auto& op : Instructions(spirv, 43)) {
                        if (op.size() >= 3) constants[op[1]] = op[2];
                    }
                    const auto barriers{Instructions(spirv, 224)};
                    REQUIRE(barriers.size() == 1);
                    REQUIRE(barriers.front().size() == 3);
                    CHECK(constants.at(barriers.front()[2]) ==
                          (0x108U | (image_coherent ? 0x800U : 0U) |
                           (global_coherent ? 0x40U : 0U)));
                }
#ifdef HAS_OPENGL
                else if (backend == 1) {
                    const auto glsl{Backend::GLSL::EmitGLSL(fixture.profile, fixture.program)};
                    CHECK((glsl.find("coherent ") != std::string::npos) == image_coherent);
                    CHECK(glsl.find("r32ui") != std::string::npos);
                    CHECK(glsl.find("uimage2D") != std::string::npos);
                    const auto fence{glsl.find("groupMemoryBarrier();")};
                    CHECK((fence != std::string::npos) == (image_coherent || global_coherent));
                    if (fence != std::string::npos) {
                        CHECK(glsl.find("groupMemoryBarrier();", fence + 1) == std::string::npos);
                    }
                } else {
                    const auto glasm{Backend::GLASM::EmitGLASM(fixture.profile, {}, fixture.program)};
                    CHECK(glasm.find(image_coherent ? "STOREIM.COH.U32" : "STOREIM.U32") !=
                          std::string::npos);
                    CHECK(glasm.find(image_coherent ? "LOADIM.COH.U32" : "LOADIM.U32") !=
                          std::string::npos);
                    CHECK(glasm.find(",2D;") != std::string::npos);
                    CHECK(glasm.find("SHADOW") == std::string::npos);
                    const auto membar{glasm.find("MEMBAR.CTA;")};
                    CHECK((membar != std::string::npos) == (image_coherent || global_coherent));
                    if (membar != std::string::npos) {
                        CHECK(glasm.find("MEMBAR.CTA;", membar + 1) == std::string::npos);
                    }
                }
#endif
            }
        }
    }
}
