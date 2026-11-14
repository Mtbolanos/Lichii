#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""The SPIR-V checks in compile.py, on small hand-written modules. Needs no Mesa build."""
import importlib.util
import os
import unittest

spec = importlib.util.spec_from_file_location(
    "radv_compile", os.path.join(os.path.dirname(os.path.abspath(__file__)), "compile.py"))
radv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(radv)

HEADER = """
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint {model} %main "main" %in %out
{modes}
OpDecorate %in Location 0
OpDecorate %out Location 0
OpDecorate %tex DescriptorSet 0
OpDecorate %tex Binding 0
OpDecorate %pc Block
OpMemberDecorate %pcs 0 Offset 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%float = OpTypeFloat 32
%half = OpTypeFloat 16
%uint = OpTypeInt 32 0
%v2 = OpTypeVector %float 2
%v4 = OpTypeVector %float 4
%uv4 = OpTypeVector %uint 4
%zero = OpConstant %float 0
%one = OpConstant %float 1
%u0 = OpConstant %uint 0
%u3 = OpConstant %uint 3
%u4 = OpConstant %uint 4
%pin = OpTypePointer Input %float
%pout = OpTypePointer Output %v4
%pf = OpTypePointer Function %float
%pwg = OpTypePointer Workgroup %float
%img = OpTypeImage %float 2D 0 0 0 1 Unknown
%si = OpTypeSampledImage %img
%arr = OpTypeArray %si %u4
%parr = OpTypePointer UniformConstant %arr
%psi = OpTypePointer UniformConstant %si
%pcs = OpTypeStruct %uint
%ppcs = OpTypePointer PushConstant %pcs
%ppcu = OpTypePointer PushConstant %uint
%in = OpVariable %pin Input
%out = OpVariable %pout Output
%tex = OpVariable %parr UniformConstant
%pc = OpVariable %ppcs PushConstant
%wg = OpVariable %pwg Workgroup
%main = OpFunction %void None %fn
%entry = OpLabel
%local = OpVariable %pf Function
%x = OpLoad %float %in
"""

FOOTER = """
%res = OpCompositeConstruct %v4 %x %x %x %x
OpStore %out %res
OpReturn
OpFunctionEnd
"""


def checks(body, model="Fragment", modes="OpExecutionMode %main OriginUpperLeft"):
    module = radv.Module(HEADER.format(model=model, modes=modes) + body + FOOTER, "test")
    return {(f["check"], f["severity"]) for f in radv.check_module(module)}


class Checks(unittest.TestCase):
    def test_fmin_is_nan_unsafe(self):
        self.assertIn(("nan-minmax", "warn"), checks("%m = OpExtInst %float %glsl FMin %x %one\n"))
        self.assertNotIn(("nan-minmax", "warn"), checks("%m = OpExtInst %float %glsl NMin %x %one\n"))

    def test_divergent_descriptor_index(self):
        body = """
%i = OpConvertFToU %uint %x
%p = OpAccessChain %psi %tex %i
%t = OpLoad %si %p
%c = OpCompositeConstruct %v2 %x %x
%s = OpImageSampleExplicitLod %v4 %t %c Lod %zero
"""
        found = checks(body)
        self.assertIn(("descriptor-divergent", "error"), found)
        self.assertIn(("descriptor-unclamped", "warn"), found)

    def test_uniform_clamped_descriptor_index(self):
        body = """
%pi = OpAccessChain %ppcu %pc %u0
%raw = OpLoad %uint %pi
%i = OpExtInst %uint %glsl UMin %raw %u3
%p = OpAccessChain %psi %tex %i
"""
        found = checks(body)
        self.assertNotIn(("descriptor-divergent", "error"), found)
        self.assertNotIn(("descriptor-unclamped", "warn"), found)

    def test_derivative_after_divergent_kill(self):
        body = """
%c = OpFOrdGreaterThan %bool %x %one
OpSelectionMerge %merge None
OpBranchConditional %c %kill %merge
%kill = OpLabel
OpKill
%merge = OpLabel
%d = OpDPdx %float %x
"""
        self.assertIn(("derivative-divergent", "error"), checks(body))

    def test_derivative_in_divergent_branch(self):
        body = """
%c = OpFOrdGreaterThan %bool %x %one
OpSelectionMerge %merge None
OpBranchConditional %c %then %merge
%then = OpLabel
%d = OpDPdx %float %x
OpBranch %merge
%merge = OpLabel
"""
        self.assertIn(("derivative-divergent", "warn"), checks(body))

    def test_derivative_in_uniform_flow(self):
        body = """
%pi = OpAccessChain %ppcu %pc %u0
%u = OpLoad %uint %pi
%c = OpIEqual %bool %u %u0
OpSelectionMerge %merge None
OpBranchConditional %c %then %merge
%then = OpLabel
%d = OpDPdx %float %x
OpBranch %merge
%merge = OpLabel
"""
        self.assertFalse({c for c in checks(body) if c[0] == "derivative-divergent"})

    def test_ballot_low_bits(self):
        body = """
%b = OpGroupNonUniformBallot %uv4 %u3 %true
%lo = OpCompositeExtract %uint %b 0
"""
        found = checks(body)
        self.assertIn(("subgroup-32-bits", "warn"), found)
        self.assertIn(("helper-subgroup", "warn"), found)

    def test_read_before_write(self):
        body = """
%early = OpLoad %float %local
OpStore %local %x
%late = OpLoad %float %local
"""
        self.assertIn(("uninit-local", "warn"), checks(body))
        self.assertNotIn(("uninit-local", "warn"), checks("OpStore %local %x\n%late = OpLoad %float %local\n"))

    def test_undef_and_fp16(self):
        found = checks("%u = OpUndef %float\n%s = OpFAdd %float %u %x\n%h = OpFConvert %half %s\n")
        self.assertIn(("undef", "warn"), found)
        self.assertIn(("fp16-overflow", "warn"), found)
        clamped = checks("%k = OpExtInst %float %glsl NClamp %x %zero %one\n%h = OpFConvert %half %k\n")
        self.assertNotIn(("fp16-overflow", "warn"), clamped)

    def test_shared_memory_never_written(self):
        found = checks("%w = OpLoad %float %wg\n", model="GLCompute",
                       modes="OpExecutionMode %main LocalSize 64 1 1")
        self.assertIn(("uninit-shared", "error"), found)
        aliased = checks("OpDecorate %wg Aliased\nOpDecorate %wg2 Aliased\n%wg2 = OpVariable %pwg Workgroup\n"
                         "OpStore %wg2 %x\n%w = OpLoad %float %wg\n", model="GLCompute",
                         modes="OpExecutionMode %main LocalSize 64 1 1")
        self.assertNotIn(("uninit-shared", "error"), aliased)
        self.assertIn(("shared-without-barrier", "warn"), aliased)

    def test_float_controls(self):
        body = "%q = OpFDiv %float %one %x\n%n = OpIsNan %bool %q\n"
        self.assertIn(("nan-test-without-preserve", "warn"), checks(body))
        preserved = checks(body, modes="OpExecutionMode %main OriginUpperLeft\n"
                                       "OpExecutionMode %main SignedZeroInfNanPreserve 32")
        self.assertNotIn(("nan-test-without-preserve", "warn"), preserved)


if __name__ == "__main__":
    unittest.main()
