#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compile Lichii's shaders with Mesa RADV (ACO) for AMD chips without an AMD GPU, and check
their SPIR-V for constructs whose results differ between drivers.

    compile.py spirv  MODULE.spv[:entry] ...    compile one pipeline from SPIR-V modules
    compile.py record PIPELINE.record           compile one Lichii pipeline cache record
    compile.py cache  vulkan.fork-vNN.bin       check every record of a cache, compile some
    compile.py foz    DATABASE.foz              compile a Fossilize capture
    compile.py check  MODULE.spv ...            only the SPIR-V checks

Run setup.sh first. See README.md.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HOME = Path(os.environ.get("RADV_OFFLINE_HOME", Path.home() / ".cache/lichii-radv-offline"))

# The chips the brief names, and the drm-shim device that stands in for each.
CHIPS = {
    "gfx1030": "navi21",  # RX 6800/6900
    "gfx1032": "navi23",  # RX 6600/6650 XT
    "gfx1100": "navi31",  # RX 7900
    "gfx1102": "navi33",  # RX 7600
    "gfx1200": "gfx1200",  # RX 9060
    "gfx1201": "gfx1201",  # RX 9070
}
SHIM_NAMES = {shim: chip for chip, shim in CHIPS.items()}

STAGES = {
    "Vertex": ("vertex", 0x1),
    "TessellationControl": ("tess-control", 0x2),
    "TessellationEvaluation": ("tess-eval", 0x4),
    "Geometry": ("geometry", 0x8),
    "Fragment": ("fragment", 0x10),
    "GLCompute": ("compute", 0x20),
    "TaskEXT": ("task", 0x40),
    "MeshEXT": ("mesh", 0x80),
}

FORMATS = {
    "R8G8B8A8_UNORM": 37,
    "B8G8R8A8_UNORM": 44,
    "A2B10G10R10_UNORM_PACK32": 64,
    "R16G16B16A16_SFLOAT": 97,
    "R32G32B32A32_UINT": 107,
    "R32G32B32A32_SINT": 108,
    "R32G32B32A32_SFLOAT": 109,
    "B10G11R11_UFLOAT_PACK32": 122,
    "D16_UNORM": 124,
    "D32_SFLOAT": 126,
    "D24_UNORM_S8_UINT": 129,
    "D32_SFLOAT_S8_UINT": 130,
}

# --------------------------------------------------------------------------------------------
# SPIR-V, read from spirv-dis text

TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|\S+')
TERMINATORS = {"OpBranch", "OpBranchConditional", "OpSwitch", "OpReturn", "OpReturnValue", "OpKill",
               "OpUnreachable", "OpTerminateInvocation", "OpEmitMeshTasksEXT"}
EXITS = {"OpKill", "OpTerminateInvocation"}
UNTYPED = {"OpLabel", "OpExtInstImport", "OpString", "OpDecorationGroup"}
UNIFORM_BUILTINS = {"WorkgroupId", "NumWorkgroups", "WorkgroupSize", "SubgroupSize", "NumSubgroups",
                    "DrawIndex", "BaseVertex", "BaseInstance", "ViewIndex", "DeviceIndex"}
DERIVATIVES = {"OpDPdx", "OpDPdy", "OpFwidth", "OpDPdxFine", "OpDPdyFine", "OpFwidthFine",
               "OpDPdxCoarse", "OpDPdyCoarse", "OpFwidthCoarse", "OpImageSampleImplicitLod",
               "OpImageSampleDrefImplicitLod", "OpImageSampleProjImplicitLod",
               "OpImageSampleProjDrefImplicitLod", "OpImageQueryLod", "OpImageSparseSampleImplicitLod",
               "OpImageSparseSampleDrefImplicitLod"}
SUBGROUP_UNIFORM_RESULT = ("Broadcast", "BroadcastFirst", "All", "Any", "AllEqual", "Ballot", "BallotBitCount")
LANE_BUILTINS = {"SubgroupLocalInvocationId", "SubgroupEqMask", "SubgroupGeMask", "SubgroupGtMask",
                 "SubgroupLeMask", "SubgroupLtMask", "SubgroupSize", "SubgroupId", "NumSubgroups"}


class Inst:
    __slots__ = ("result", "op", "args", "block", "func", "line")

    def __init__(self, result, op, args, line):
        self.result, self.op, self.args, self.line = result, op, args, line
        self.block = self.func = None

    def ids(self):
        return [a for a in self.args if a.startswith("%")]


class Function:
    def __init__(self, inst):
        self.id = inst.result
        self.params = []
        self.blocks = {}
        self.order = []
        self.idom, self.reachable = {}, set()

    def successors(self, label):
        term = self.blocks[label][-1] if self.blocks[label] else None
        if term is None:
            return []
        if term.op == "OpBranch":
            return [term.args[0]]
        if term.op == "OpBranchConditional":
            return [term.args[1], term.args[2]]
        if term.op == "OpSwitch":
            return [term.args[1]] + term.args[3::2]
        return []

    def dominators(self):
        preds = defaultdict(list)
        for label in self.order:
            for succ in self.successors(label):
                preds[succ].append(label)
        postorder, seen = [], {self.order[0]}
        stack = [(self.order[0], iter(self.successors(self.order[0])))]
        while stack:
            node, it = stack[-1]
            nxt = next((s for s in it if s not in seen and s in self.blocks), None)
            if nxt is None:
                postorder.append(node)
                stack.pop()
            else:
                seen.add(nxt)
                stack.append((nxt, iter(self.successors(nxt))))
        rpo = postorder[::-1]
        index = {label: i for i, label in enumerate(rpo)}
        idom = {rpo[0]: rpo[0]}
        changed = True
        while changed:
            changed = False
            for label in rpo[1:]:
                done = [p for p in preds[label] if p in idom]
                if not done:
                    continue
                new = done[0]
                for p in done[1:]:
                    a, b = p, new
                    while a != b:
                        while index[a] > index[b]:
                            a = idom[a]
                        while index[b] > index[a]:
                            b = idom[b]
                    new = a
                if idom.get(label) != new:
                    idom[label] = new
                    changed = True
        self.idom, self.reachable = idom, set(rpo)
        # Pre/post numbers on the dominator tree make a dominance test O(1).
        children = defaultdict(list)
        for label in rpo[1:]:
            if label in idom:
                children[idom[label]].append(label)
        self.pre, self.post, self.tree_order = {}, {}, []
        stack = [(rpo[0], False)]
        while stack:
            node, done = stack.pop()
            if done:
                self.post[node] = len(self.tree_order)
                continue
            self.pre[node] = len(self.tree_order)
            self.tree_order.append(node)
            stack.append((node, True))
            stack += [(child, False) for child in reversed(children[node])]

    def dominates(self, a, b):
        return a in self.pre and b in self.pre and self.pre[a] <= self.pre[b] and self.post[b] <= self.post[a]

    def dominated(self, header):
        return self.tree_order[self.pre[header]:self.post[header]] if header in self.pre else []

    def construct(self, header, merge):
        return {b for b in self.dominated(header) if b != header and not self.dominates(merge, b)}


class Module:
    def __init__(self, text, name):
        self.name = name
        self.insts, self.defs, self.uses = [], {}, defaultdict(list)
        self.decor, self.member_decor = defaultdict(list), defaultdict(list)
        self.entries, self.modes, self.functions = [], defaultdict(list), []
        func = block = None
        for number, line in enumerate(text.splitlines(), 1):
            tokens = TOKEN.findall(line)
            if ";" in tokens:
                tokens = tokens[:tokens.index(";")]
            if not tokens:
                continue
            result = None
            if len(tokens) > 2 and tokens[1] == "=":
                result, tokens = tokens[0], tokens[2:]
            if not tokens[0].startswith("Op"):
                continue
            inst = Inst(result, tokens[0], tokens[1:], number)
            self.insts.append(inst)
            if result:
                self.defs[result] = inst
            for a in inst.ids():
                self.uses[a].append(inst)
            op = inst.op
            if op == "OpDecorate":
                self.decor[inst.args[0]].append(inst.args[1:])
            elif op == "OpMemberDecorate":
                self.member_decor[(inst.args[0], int(inst.args[1]))].append(inst.args[2:])
            elif op == "OpEntryPoint":
                self.entries.append((inst.args[0], inst.args[1], inst.args[2].strip('"'), inst.args[3:]))
            elif op in ("OpExecutionMode", "OpExecutionModeId"):
                self.modes[inst.args[0]].append(inst.args[1:])
            elif op == "OpFunction":
                func = Function(inst)
                self.functions.append(func)
            elif op == "OpFunctionParameter" and func:
                func.params.append(result)
            elif op == "OpLabel" and func:
                block = result
                func.blocks[block] = []
                func.order.append(block)
            elif op == "OpFunctionEnd":
                func = block = None
            if block is not None and op != "OpLabel":
                inst.block, inst.func = block, func
                func.blocks[block].append(inst)
                if op in TERMINATORS:
                    block = None
        for f in self.functions:
            if f.order:
                f.dominators()

    def type_of(self, value):
        inst = self.defs.get(value)
        return inst.args[0] if inst and inst.op not in UNTYPED and not inst.op.startswith("OpType") else None

    def strip(self, type_id, ops=("OpTypePointer",)):
        while type_id in self.defs and self.defs[type_id].op in ops:
            t = self.defs[type_id]
            type_id = t.args[1] if t.op == "OpTypePointer" else t.args[0]
        return type_id

    def float_width(self, type_id):
        t = self.defs.get(type_id)
        if t and t.op in ("OpTypeVector", "OpTypeMatrix"):
            t = self.defs.get(t.args[0])
            if t and t.op == "OpTypeVector":
                t = self.defs.get(t.args[0])
        return int(t.args[0]) if t and t.op == "OpTypeFloat" else None

    def constant(self, value):
        inst = self.defs.get(value)
        if inst and inst.op == "OpConstant":
            try:
                text = inst.args[1]
                return float(text) if any(c in text for c in ".eEn") and not text.startswith("0x") else int(text, 0)
            except ValueError:
                return None
        return None

    def builtin(self, value):
        for d in self.decor.get(value, []):
            if d[0] == "BuiltIn":
                return d[1]
        return None

    def has_decor(self, value, name):
        return any(d[0] == name for d in self.decor.get(value, []))

    def root(self, pointer):
        for _ in range(64):
            inst = self.defs.get(pointer)
            if inst is None:
                return None
            if inst.op in ("OpAccessChain", "OpInBoundsAccessChain", "OpPtrAccessChain", "OpCopyObject"):
                pointer = inst.args[1]
            else:
                return inst
        return None


# --------------------------------------------------------------------------------------------
# Uniformity: which values and blocks can differ between the invocations of a subgroup


class Uniformity:
    def __init__(self, m: Module):
        self.m = m
        self.values, self.vars, self.params, self.returns, self.merges = set(), set(), set(), set(), set()
        self.blocks = {}
        self.callees = {f.id: f for f in m.functions}
        while self._pass():
            pass

    def _value(self, inst):
        m, op = self.m, inst.op
        if op == "OpLoad":
            root = m.root(inst.args[1])
            if root is None or root.op != "OpVariable":
                return root is not None and root.result in self.params
            storage = root.args[1]
            if storage == "Input":
                return m.builtin(root.result) not in UNIFORM_BUILTINS
            if storage in ("Output", "Workgroup", "TaskPayloadWorkgroupEXT"):
                return True
            if storage in ("Function", "Private"):
                return root.result in self.vars or inst.args[1] in self.values
            return inst.args[1] in self.values
        if op == "OpVariable" or op.startswith(("OpType", "OpConstant", "OpSpecConstant")):
            return False
        if op == "OpUndef" or op.startswith("OpAtomic"):
            return True
        if op == "OpFunctionParameter":
            return inst.result in self.params
        if op == "OpPhi":
            return inst.block in self.merges or any(a in self.values for a in inst.args[1::2])
        if op == "OpFunctionCall":
            return inst.args[1] in self.returns or any(a in self.values for a in inst.args[2:])
        if op.startswith("OpGroupNonUniform"):
            if op == "OpGroupNonUniformElect":
                return True
            return not (op[len("OpGroupNonUniform"):] in SUBGROUP_UNIFORM_RESULT or "Reduce" in inst.args)
        return any(a in self.values for a in inst.args[1:] if a.startswith("%"))

    def _pass(self):
        m = self.m
        before = (len(self.values), len(self.vars), len(self.blocks), len(self.params), len(self.returns),
                  len(self.merges))
        for f in m.functions:
            for label in f.order:
                for inst in f.blocks[label]:
                    if inst.result and inst.result not in self.values and self._value(inst):
                        self.values.add(inst.result)
                    if inst.op == "OpStore":
                        root = m.root(inst.args[0])
                        if root is not None and root.op == "OpVariable" and root.args[1] in ("Function", "Private") \
                                and (inst.args[1] in self.values or inst.args[0] in self.values or label in self.blocks):
                            self.vars.add(root.result)
                    elif inst.op == "OpFunctionCall" and inst.args[1] in self.callees:
                        for param, arg in zip(self.callees[inst.args[1]].params, inst.args[2:]):
                            if arg in self.values:
                                self.params.add(param)
                    elif inst.op == "OpReturnValue" and (inst.args[0] in self.values or label in self.blocks):
                        self.returns.add(f.id)
            for label in f.order:
                insts = f.blocks[label]
                if label not in f.reachable or not insts:
                    continue
                merge = next((i for i in insts if i.op in ("OpSelectionMerge", "OpLoopMerge")), None)
                if merge is None:
                    continue
                term = insts[-1]
                construct = f.construct(label, merge.args[0])
                if merge.op == "OpSelectionMerge":
                    cond = term.args[0] if term.op in ("OpBranchConditional", "OpSwitch") else None
                    divergent = cond in self.values
                    reason = f"branch on non-uniform {cond}"
                else:
                    construct |= {label}
                    exits = [b for b in construct if f.blocks[b] and f.blocks[b][-1].op == "OpBranchConditional"
                             and merge.args[0] in f.blocks[b][-1].args[1:3]]
                    divergent = any(f.blocks[b][-1].args[0] in self.values for b in exits)
                    reason = "loop with a non-uniform exit"
                if not divergent:
                    continue
                self.merges.add(merge.args[0])
                for b in construct:
                    self.blocks.setdefault(b, reason)
                exits = [f.blocks[b][-1].op for b in construct if f.blocks[b]]
                if any(op in EXITS or op in ("OpReturn", "OpReturnValue") for op in exits):
                    after = f"after {'a kill' if any(op in EXITS for op in exits) else 'a return'} in a {reason}"
                    for b in f.dominated(merge.args[0]):
                        self.blocks.setdefault(b, after)
        return before != (len(self.values), len(self.vars), len(self.blocks), len(self.params),
                          len(self.returns), len(self.merges))


# --------------------------------------------------------------------------------------------
# Checks. Each finding: check, severity (error, warn, info), message, count, where.


def finding(check, severity, message, where=(), count=None):
    where = list(where)
    return {"check": check, "severity": severity, "message": message,
            "count": count if count is not None else max(1, len(where)), "where": where[:8]}


def loc(inst):
    return f"line {inst.line}: {inst.result + ' = ' if inst.result else ''}{inst.op}" + \
        (f" {inst.args[2]}" if inst.op == "OpExtInst" else "")


def check_module(m: Module):
    out = []
    uni = Uniformity(m)
    stages = {STAGES.get(model, (model,))[0] for model, *_ in m.entries}

    # NaN, Inf and signed zero
    by_ext = defaultdict(list)
    for inst in m.insts:
        if inst.op == "OpExtInst" and len(inst.args) > 2:
            by_ext[inst.args[2]].append(inst)
    fminmax = by_ext["FMin"] + by_ext["FMax"] + by_ext["FClamp"]
    if fminmax:
        out.append(finding("nan-minmax", "warn",
                           "FMin/FMax/FClamp: the result is undefined when an operand is NaN, and "
                           "min(-0, +0) may return either zero (GLSL.std.450). Maxwell FMNMX returns "
                           "the non-NaN operand, which is NMin/NMax/NClamp.", map(loc, fminmax)))
    controls = defaultdict(set)
    for modes in m.modes.values():
        for mode in modes:
            if mode[0] in ("SignedZeroInfNanPreserve", "DenormPreserve", "DenormFlushToZero"):
                controls[mode[0]].add(int(mode[1]))
    arith = defaultdict(list)
    for inst in m.insts:
        if inst.op in ("OpFAdd", "OpFSub", "OpFMul", "OpFDiv", "OpFRem", "OpFMod", "OpFNegate") or \
                (inst.op == "OpExtInst" and inst.args[2] in ("Fma", "Sqrt", "InverseSqrt", "Exp2", "Log2")):
            width = m.float_width(inst.args[0])
            if width:
                arith[width].append(inst)
    for width, insts in sorted(arith.items()):
        if width not in controls["SignedZeroInfNanPreserve"]:
            nan_tests = [i for i in m.insts if i.op in ("OpIsNan", "OpIsInf", "OpOrdered", "OpUnordered")
                         and m.float_width(m.type_of(i.args[1])) == width]
            sensitive = [i for i in insts if i.op in ("OpFDiv", "OpFRem", "OpFMod")] + \
                [i for i in by_ext["FSign"] + by_ext["Atan2"] + by_ext["Pow"] if m.float_width(i.args[0]) == width]
            if nan_tests:
                out.append(finding("nan-test-without-preserve", "warn",
                                   f"IsNan/IsInf on fp{width} without SignedZeroInfNanPreserve {width}: "
                                   "a driver may assume no NaN or Inf and fold the test to false.",
                                   map(loc, nan_tests)))
            out.append(finding("float-controls", "info",
                               f"fp{width} arithmetic without SignedZeroInfNanPreserve {width}: the driver "
                               f"may drop -0 and assume finite values. {len(sensitive)} divisions, sign, "
                               "atan2 or pow results depend on it.", map(loc, sensitive), count=len(insts)))
        if width not in controls["DenormPreserve"] | controls["DenormFlushToZero"]:
            out.append(finding("denormals", "info",
                               f"fp{width} arithmetic with no DenormPreserve or DenormFlushToZero {width}: "
                               "denormals follow the driver's default, which differs between vendors, while "
                               "Maxwell flushes only where the instruction's FTZ bit says so.", count=len(insts)))

    # float16 overflow
    narrowing = [i for i in m.insts if i.op in ("OpFConvert", "OpQuantizeToF16") and m.float_width(i.args[0]) == 16]
    narrowing += by_ext["PackHalf2x16"]
    unclamped = []
    for inst in narrowing:
        source = m.defs.get(inst.args[-1])
        bounded = source is not None and source.op == "OpExtInst" and \
            source.args[2] in ("FClamp", "NClamp", "FMin", "NMin")
        if not bounded:
            unclamped.append(inst)
    if unclamped:
        out.append(finding("fp16-overflow", "warn",
                           "fp32 to fp16 without a clamp: values over 65504 become Inf (or 65504 where the "
                           "hardware rounds toward zero), and an Inf later turns into NaN.", map(loc, unclamped)))
    if 16 in arith:
        out.append(finding("fp16-arithmetic", "info",
                           "fp16 arithmetic: intermediates overflow at 65504.", map(loc, arith[16])))
    relaxed = [v for v, ds in m.decor.items() if any(d[0] == "RelaxedPrecision" for d in ds)]
    if relaxed:
        out.append(finding("relaxed-precision", "info",
                           "RelaxedPrecision values: a driver may compute them in fp16.", relaxed))

    # uninitialized values and memory
    undef_uses = [u for inst in m.insts if inst.op == "OpUndef" for u in m.uses[inst.result] if u.op != "OpPhi"]
    if undef_uses:
        out.append(finding("undef", "warn",
                           "OpUndef used as an operand: drivers pick different values (RADV often whatever "
                           "the register held).", map(loc, undef_uses)))
    # With an explicit workgroup layout, Aliased Workgroup blocks are views of the same memory.
    aliased = [i.result for i in m.insts if i.op == "OpVariable" and len(i.args) == 2
               and i.args[1] == "Workgroup" and m.has_decor(i.result, "Aliased")]
    memory = {var: aliased[0] for var in aliased}
    stores, loads = defaultdict(list), defaultdict(list)

    def access(table, pointer, inst):
        root = m.root(pointer)
        if root is not None:
            table[memory.get(root.result, root.result)].append(inst)

    for inst in m.insts:
        if inst.op in ("OpStore", "OpCopyMemory"):
            access(stores, inst.args[0], inst)
        elif inst.op == "OpLoad":
            access(loads, inst.args[1], inst)
        elif inst.op.startswith("OpAtomic") and len(inst.args) > 2:
            access(stores, inst.args[2] if inst.result else inst.args[0], inst)
            access(loads, inst.args[2] if inst.result else inst.args[0], inst)
    early = []
    for var in (i for i in m.insts if i.op == "OpVariable" and len(i.args) == 2):
        storage = var.args[1]
        if storage == "Function":
            for load in loads.get(var.result, []):
                f = load.func
                if f is None or load.block not in f.reachable:
                    continue
                covered = any(store.func is f and store.block in f.reachable and
                              (store.line < load.line if store.block == load.block
                               else f.dominates(store.block, load.block))
                              for store in stores.get(var.result, []))
                if not covered:
                    early.append(load)
        elif storage == "Private" and loads.get(var.result) and not stores.get(var.result):
            early.extend(loads[var.result])
        elif storage == "Workgroup" and loads.get(var.result):
            if not stores.get(var.result):
                out.append(finding("uninit-shared", "error",
                                   f"Workgroup variable {var.result} is read but never written.",
                                   map(loc, loads[var.result])))
            elif not any(i.op == "OpControlBarrier" for i in m.insts):
                out.append(finding("shared-without-barrier", "warn",
                                   f"Workgroup variable {var.result} is written and read with no "
                                   "OpControlBarrier: the result depends on how the driver schedules waves.",
                                   map(loc, loads[var.result])))
    if early:
        out.append(finding("uninit-local", "warn",
                           "Function or Private variable read where no store dominates the load: the value "
                           "is undefined on some paths.", map(loc, early)))
    for model, fid, name, interface in m.entries:
        for var in interface:
            inst = m.defs.get(var)
            if inst is None or inst.op != "OpVariable" or inst.args[1] != "Output":
                continue
            builtin = m.builtin(var)
            if builtin not in (None, "Position", "FragDepth") or stores.get(var):
                continue
            out.append(finding("output-never-written", "warn",
                               f"{STAGES.get(model, (model,))[0]} output {var}"
                               f"{' (' + builtin + ')' if builtin else ''} is never written: the next stage "
                               "or the attachment gets an undefined value.", [var]))

    # descriptor indices
    for inst in m.insts:
        if inst.op not in ("OpAccessChain", "OpInBoundsAccessChain") or len(inst.args) < 3:
            continue
        base = m.defs.get(inst.args[1])
        if base is None or base.op != "OpVariable" or \
                base.args[1] not in ("UniformConstant", "Uniform", "StorageBuffer"):
            continue
        array = m.defs.get(m.strip(base.args[0]))
        if array is None or array.op not in ("OpTypeArray", "OpTypeRuntimeArray"):
            continue
        index = inst.args[2]
        value = m.constant(index)
        length = m.constant(array.args[1]) if array.op == "OpTypeArray" else None
        if value is not None:
            if length is not None and value >= length:
                out.append(finding("descriptor-oob", "error",
                                   f"constant descriptor index {value} into {base.result}[{length}].", [loc(inst)]))
            continue
        marked = m.has_decor(index, "NonUniform") or m.has_decor(inst.result, "NonUniform") or \
            any(m.has_decor(u.result, "NonUniform") for u in m.uses[inst.result] if u.result)
        if index in uni.values and not marked:
            out.append(finding("descriptor-divergent", "error",
                               f"non-uniform index into descriptor array {base.result} without NonUniform: "
                               "AMD loads one descriptor per wave (readfirstlane), so other lanes use the first "
                               "lane's resource.", [loc(inst)]))
        source = m.defs.get(index)
        clamped = source is not None and (
            (source.op == "OpExtInst" and source.args[2] in ("UMin", "SMin", "UClamp", "SClamp")) or
            source.op in ("OpBitwiseAnd", "OpSelect", "OpUMod"))
        if not clamped:
            out.append(finding("descriptor-unclamped", "warn",
                               f"dynamic index into descriptor array {base.result}"
                               f"{'[' + str(length) + ']' if length else '[]'} is not clamped: an out-of-range "
                               "index is undefined even with robustness.", [loc(inst)]))

    # helper invocations and derivatives
    if "fragment" in stages:
        divergent = [(i, uni.blocks[label]) for f in m.functions for label in f.order if label in uni.blocks
                     for i in f.blocks[label] if i.op in DERIVATIVES]
        if divergent:
            killed = any("kill" in reason for _, reason in divergent)
            out.append(finding("derivative-divergent", "error" if killed else "warn",
                               "implicit derivatives (texture sampling with implicit LOD, dFdx/dFdy) in "
                               "non-uniform control flow: inactive quad lanes give undefined derivatives, and "
                               "AMD reads whatever their registers hold.",
                               (f"{loc(i)} ({reason})" for i, reason in divergent)))
        group = [i for i in m.insts if i.op.startswith("OpGroupNonUniform") and "Quad" not in i.op]
        if group:
            demote = any(i.op == "OpDemoteToHelperInvocation" for i in m.insts)
            out.append(finding("helper-subgroup", "warn",
                               "subgroup operations in a fragment shader: whether helper invocations take part "
                               f"is implementation-defined{' (and this shader demotes)' if demote else ''}.",
                               map(loc, group)))

    # subgroup size assumptions
    lanes = []
    for inst in m.insts:
        if inst.op == "OpGroupNonUniformBallot":
            lanes.append((inst, "ballot"))
        elif inst.op == "OpLoad":
            root = m.root(inst.args[1])
            builtin = m.builtin(root.result) if root is not None else None
            if builtin and builtin.startswith("Subgroup") and builtin.endswith("Mask"):
                lanes.append((inst, builtin))
    low_only = []
    for inst, what in lanes:
        users = m.uses[inst.result]
        extracts = {u.args[2] for u in users if u.op == "OpCompositeExtract" and len(u.args) > 2}
        if users and all(u.op == "OpCompositeExtract" for u in users) and extracts == {"0"}:
            low_only.append(f"{loc(inst)} ({what})")
    if low_only:
        out.append(finding("subgroup-32-bits", "warn",
                           "only the low 32 bits of a ballot or lane mask are used: invocations 32-63 of a "
                           "64-wide subgroup are lost.", low_only))
    masked = []
    for inst in m.insts:
        if inst.op != "OpLoad":
            continue
        root = m.root(inst.args[1])
        if root is not None and m.builtin(root.result) == "SubgroupLocalInvocationId":
            masked += [loc(u) for u in m.uses[inst.result]
                       if u.op in ("OpBitwiseAnd", "OpUMod") and any(m.constant(a) in (31, 32) for a in u.args[2:])]
    if masked:
        out.append(finding("subgroup-lane-mask", "warn", "the subgroup lane id is masked to 0-31.", masked))
    used = sorted({m.builtin(v) for v in m.decor if m.builtin(v) in LANE_BUILTINS})
    ops = Counter(i.op for i in m.insts if i.op.startswith("OpGroupNonUniform"))
    if used or ops:
        out.append(finding("subgroup-ops", "info",
                           "uses " + ", ".join(used + [f"{op} x{n}" for op, n in sorted(ops.items())]) +
                           ": results depend on the subgroup size, which Lichii sets to 32 where the driver "
                           "lets it.", count=sum(ops.values()) + len(used)))
    return out


def uses_subgroups(m: Module):
    return any(i.op.startswith("OpGroupNonUniform") for i in m.insts) or \
        any(m.builtin(v) in LANE_BUILTINS for v in m.decor)


# --------------------------------------------------------------------------------------------
# Reflection: the pipeline layout and attachments radv_pipe needs


def descriptor_type(m: Module, var):
    storage = var.args[1]
    pointee = m.strip(var.args[0])
    count = 1
    t = m.defs.get(pointee)
    if t is not None and t.op in ("OpTypeArray", "OpTypeRuntimeArray"):
        count = (m.constant(t.args[1]) or 1) if t.op == "OpTypeArray" else 64
        pointee = t.args[0]
        t = m.defs.get(pointee)
    if storage == "StorageBuffer" or (storage == "Uniform" and m.has_decor(pointee, "BufferBlock")):
        return 7, count
    if storage == "Uniform":
        return 6, count
    if t is None:
        return None, count
    if t.op == "OpTypeSampler":
        return 0, count
    if t.op == "OpTypeSampledImage":
        return 1, count
    if t.op == "OpTypeAccelerationStructureKHR":
        return 1000150000, count
    if t.op == "OpTypeImage":
        dim, sampled = t.args[1], t.args[5]
        if dim == "Buffer":
            return (4 if sampled == "1" else 5), count
        if dim == "SubpassData":
            return 10, count
        return (2 if sampled == "1" else 3), count
    return None, count


def type_size(m: Module, type_id):
    t = m.defs[type_id]
    if t.op in ("OpTypeInt", "OpTypeFloat"):
        return int(t.args[0]) // 8
    if t.op == "OpTypeVector":
        return type_size(m, t.args[0]) * int(t.args[1])
    if t.op == "OpTypeMatrix":
        stride = next((int(d[1]) for d in m.decor.get(type_id, []) if d[0] == "MatrixStride"), None)
        return (stride or type_size(m, t.args[0])) * int(t.args[1])
    if t.op == "OpTypeArray":
        stride = next((int(d[1]) for d in m.decor.get(type_id, []) if d[0] == "ArrayStride"), None)
        return (stride or type_size(m, t.args[0])) * (m.constant(t.args[1]) or 1)
    if t.op == "OpTypeStruct":
        end = 0
        for i, member in enumerate(t.args):
            offset = next((int(d[1]) for d in m.member_decor.get((type_id, i), []) if d[0] == "Offset"), end)
            end = max(end, offset + type_size(m, member))
        return end
    return 4


def scalar_kind(m: Module, type_id):
    t = m.defs.get(m.strip(type_id, ("OpTypePointer", "OpTypeVector", "OpTypeArray")))
    if t is not None and t.op == "OpTypeInt":
        return "sint" if t.args[1] == "1" else "uint"
    return "float"


# --------------------------------------------------------------------------------------------
# Running tools


def tool(name):
    for candidate in (HOME / "bin" / name, HOME / "fossilize/bin" / name):
        if candidate.exists():
            return str(candidate)
    found = shutil.which(name)
    if not found:
        sys.exit(f"{name} not found; run tools/nx2/radv_offline/setup.sh")
    return found


def driver_env(chip, extra):
    icd = next((HOME / "mesa/share/vulkan/icd.d").glob("radeon_icd.*.json"), None)
    if icd is None:
        sys.exit("RADV is not built; run tools/nx2/radv_offline/setup.sh")
    env = dict(os.environ)
    libs = str(HOME / "mesa/lib")
    env.update({
        "VK_DRIVER_FILES": str(icd), "VK_ICD_FILENAMES": str(icd),
        "LD_LIBRARY_PATH": libs + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""),
        "LD_PRELOAD": str(HOME / "mesa/lib/libamdgpu_noop_drm_shim.so"),
        "AMDGPU_GPU_ID": CHIPS[chip], "MESA_SHADER_CACHE_DISABLE": "true",
    })
    env.update(extra)
    return env


def disassemble(path):
    result = subprocess.run([tool("spirv-dis"), "--raw-id", "--no-indent", "--no-header", "--no-color", str(path)],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{path}: {result.stderr.strip()}")
    return result.stdout


def validate(path):
    # Lichii's device requires uniformBufferStandardLayout (vulkan_device.h).
    result = subprocess.run([tool("spirv-val"), "--target-env", "vulkan1.3", "--uniform-buffer-standard-layout",
                             str(path)], capture_output=True, text=True)
    return None if result.returncode == 0 else (result.stderr or result.stdout).strip()


ISA_NOTES = [
    ("v_cvt_pkrtz_f16_f32", "packs 16-bit colour or data rounding toward zero: overflow gives 65504, not Inf"),
    ("v_readfirstlane_b32", "takes one lane's value for the whole wave (scalarized descriptor or branch)"),
    ("_legacy_f32", "legacy min/max/mul: NaN and 0*Inf follow D3D9 rules"),
    ("v_rcp_f32", "approximate reciprocal (1 ULP) used for division"),
    ("v_rsq_f32", "approximate inverse square root"),
]


def isa_notes(text):
    return [f"{pattern} x{text.count(pattern)}: {meaning}" for pattern, meaning in ISA_NOTES if pattern in text]


# --------------------------------------------------------------------------------------------
# Pipelines


class Stage:
    def __init__(self, path, entry=None, shader_hash=None):
        self.path = Path(path)
        self.module = Module(disassemble(self.path), self.path.name)
        entries = self.module.entries
        if not entries:
            raise RuntimeError(f"{path}: no entry point")
        chosen = next((e for e in entries if e[2] == entry), entries[0])
        self.model, self.entry = chosen[0], chosen[2]
        self.stage, self.flag = STAGES[self.model]
        found = re.search(r"[0-9a-f]{16}", self.path.name)
        self.hash = shader_hash or (found.group(0) if found else
                                    hashlib.sha256(self.path.read_bytes()).hexdigest()[:16])


def describe(stages, args, spec):
    lines = [f"kind {'compute' if stages[0].stage == 'compute' else 'graphics'}"]
    bindings, push, colors = {}, (0, 0), {}
    for index, s in enumerate(stages):
        m = s.module
        extra = ""
        if args.subgroup == "32" or (args.subgroup == "lichii" and uses_subgroups(m)):
            extra = " subgroup=32"
        if any(i.op.startswith("OpCooperativeMatrix") for i in m.insts):
            extra += " full"
        lines.append(f"stage {s.stage} {s.path.resolve()} {s.entry}{extra}")
        lines += [f"spec {index} {spec_id} {value}" for stage_index, spec_id, value in spec if stage_index == index]
        for inst in m.insts:
            if inst.op != "OpVariable":
                continue
            decor = {d[0]: d[1:] for d in m.decor.get(inst.result, [])}
            if "Binding" in decor:
                kind, count = descriptor_type(m, inst)
                if kind is None:
                    continue
                key = (int(decor.get("DescriptorSet", ["0"])[0]), int(decor["Binding"][0]))
                old = bindings.get(key, (kind, count, 0))
                bindings[key] = (kind, max(count, old[1]), old[2] | s.flag)
            elif inst.args[1] == "PushConstant":
                push = (push[0] | s.flag, max(push[1], type_size(m, m.strip(inst.args[0]))))
            elif inst.args[1] == "Output" and s.stage == "fragment" and "Location" in decor and \
                    m.builtin(inst.result) is None:
                colors[int(decor["Location"][0])] = scalar_kind(m, inst.args[0])
    for (set_, binding), (kind, count, flags) in sorted(bindings.items()):
        lines.append(f"binding {set_} {binding} {kind} {count} {flags:x}")
    if push[1]:
        lines.append(f"push {push[0]:x} {push[1]}")
    given = args.color or []
    defaults = {"float": FORMATS["R16G16B16A16_SFLOAT"], "uint": FORMATS["R32G32B32A32_UINT"],
                "sint": FORMATS["R32G32B32A32_SINT"]}
    for location in range(max(colors) + 1 if colors else 0):
        fmt = given[location] if location < len(given) else defaults[colors.get(location, "float")]
        lines.append(f"color {FORMATS.get(fmt, fmt)}")
    if args.depth:
        lines.append(f"depth {FORMATS.get(args.depth, args.depth)}")
    if args.robust == "none":
        lines.append("robust none")
    return "\n".join(lines) + "\n"


def parse_variants(values):
    variants = [("default", {})]
    for value in values or []:
        name, _, settings = value.partition(":")
        env, last = {}, None
        for item in filter(None, settings.split(",")):
            if "=" in item:
                last, setting = item.split("=", 1)
                env[last] = setting
            elif last is None:
                sys.exit(f"--variant {value}: expected VAR=VALUE before {item}")
            else:
                env[last] += "," + item
        variants.append((name, env))
    return variants


def compile_pipeline(stages, args, out, spec=()):
    out.mkdir(parents=True, exist_ok=True)
    description = out / "pipeline.txt"
    description.write_text(describe(stages, args, spec))
    results = []
    for chip in args.chips:
        for variant, env in parse_variants(args.variant):
            target = out / (chip if variant == "default" else f"{chip}+{variant}")
            if target.exists():
                shutil.rmtree(target)
            try:
                run = subprocess.run([tool("radv_pipe"), str(description), str(target)], capture_output=True,
                                     text=True, env=driver_env(chip, env), timeout=args.timeout)
            except subprocess.TimeoutExpired:
                results.append({"status": "error", "error": f"timed out after {args.timeout} s",
                                "chip": chip, "variant": variant})
                continue
            line = next((l for l in run.stdout.splitlines() if l.startswith("{")), None)
            result = json.loads(line) if line else {"status": "error", "error": (run.stderr or "no output")[-2000:]}
            if run.returncode < 0:
                result = {"status": "error", "error": f"radv_pipe died with signal {-run.returncode}: "
                          f"{run.stderr[-2000:]}"}
            result.update(chip=chip, variant=variant)
            for executable in result.get("executables", []):
                asm = next((f for f in executable["files"] if "Assembly" in f), None)
                executable["isa_notes"] = isa_notes((target / asm).read_text(errors="replace")) if asm else []
            results.append(result)
    (out / "compile.json").write_text(json.dumps(results, indent=1))
    return results


def stage_findings(stages, results):
    """SPIR-V findings per stage, made errors where a chip runs the stage 64 wide."""
    findings = []
    for s in stages:
        error = validate(s.path)
        if error:
            findings.append({"hash": s.hash, "stage": s.stage, **finding("spirv-val", "error", error[:4000])})
        waves = {e["subgroup_size"] for r in results for e in r.get("executables", []) if e["stages"] & s.flag}
        for f in check_module(s.module):
            if f["check"] in ("subgroup-32-bits", "subgroup-lane-mask") and 64 in waves:
                f["severity"] = "error"
                f["message"] += " RADV runs this stage 64 wide here."
            findings.append({"hash": s.hash, "stage": s.stage, **f})
    return findings


STATS = ("Instructions", "VGPRs", "SGPRs", "Spilled VGPRs", "Spilled SGPRs", "Scratch size", "Code size",
         "Subgroups per SIMD")


def write_report(out, title, rows, findings):
    order = {"error": 0, "warn": 1, "info": 2}
    findings.sort(key=lambda f: (order[f["severity"]], f["hash"], f["check"]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "findings.json").write_text(json.dumps(findings, indent=1))
    md = [f"# {title}", ""]
    if rows:
        md += ["## Compile", "", "| chip | variant | executable | wave | " + " | ".join(STATS) + " |",
               "|" + "---|" * (4 + len(STATS))]
        for r in rows:
            if r.get("status") != "ok":
                md.append(f"| {r['chip']} | {r['variant']} | failed: {r.get('error', '')[:300]} |")
                continue
            for e in r["executables"]:
                md.append(f"| {r['chip']} | {r['variant']} | {e['name']} | {e['subgroup_size']} | " +
                          " | ".join(str(e["statistics"].get(s, "")) for s in STATS) + " |")
        notes = sorted({(r["chip"], e["name"], n) for r in rows for e in r.get("executables", [])
                        for n in e.get("isa_notes", [])})
        if notes:
            md += ["", "ISA notes:", ""] + [f"- {chip} {name}: {n}" for chip, name, n in notes]
        unsized = sorted({r["unsized_stages"] for r in rows if r.get("unsized_stages")})
        if unsized:
            md += ["", f"Subgroup size 32 was asked for stage mask {unsized}, which RADV can't size: those "
                   "stages run at RADV's default, as they do in Lichii."]
        md.append("")
    counts = Counter(f["severity"] for f in findings)
    md += ["## Findings", "", f"{counts['error']} errors, {counts['warn']} warnings, {counts['info']} notes.", ""]
    for f in findings:
        md.append(f"- **{f['severity']}** \x60{f['hash']}\x60 {f['stage']} {f['check']} (x{f['count']}): {f['message']}")
        md += [f"  - {w}" for w in f["where"]]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    return counts


def parse_spec(values):
    spec = []
    for value in values or []:
        stage, _, rest = value.partition(":")
        spec_id, _, number = rest.partition("=")
        spec.append((int(stage), int(spec_id), int(number, 0) & 0xFFFFFFFF))
    return spec


def failed(results):
    return 1 if any(r["status"] != "ok" for r in results) else 0


def cmd_spirv(args):
    stages = []
    for item in args.modules:
        path, entry = (item, None) if Path(item).exists() else item.rsplit(":", 1)
        stages.append(Stage(path, entry))
    out = Path(args.output)
    results = compile_pipeline(stages, args, out, parse_spec(args.spec))
    counts = write_report(out, "radv_offline: " + ", ".join(s.hash for s in stages), results,
                          stage_findings(stages, results))
    print(f"{out}/summary.md: {counts['error']} errors, {counts['warn']} warnings; "
          f"{sum(r['status'] == 'ok' for r in results)}/{len(results)} compiles ok")
    return failed(results)


def device_profile(args, out):
    """Writes the device profile shader-sweep translates with: what Lichii's pipeline cache takes from the
    RADV device. The selected chips must agree on it, since one translation serves them all."""
    profiles = {}
    for chip in args.chips:
        run = subprocess.run([tool("radv_pipe"), "--profile"], capture_output=True, text=True,
                             env=driver_env(chip, {}), timeout=args.timeout)
        if run.returncode != 0:
            sys.exit(f"radv_pipe --profile on {chip}: {(run.stdout + run.stderr).strip()[-2000:]}")
        profiles[chip] = json.loads(run.stdout)
    first = args.chips[0]
    differ = [chip for chip, profile in profiles.items() if profile != profiles[first]]
    if differ:
        sys.exit(f"the device profiles of {', '.join(differ)} differ from {first}'s; run them separately")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "device-profile.json"
    path.write_text(json.dumps(profiles[first], indent=1))
    return path


def sweep_record(args, record, modules):
    run = subprocess.run([args.shader_sweep, "--pipeline", str(record), "--cache-version", str(args.cache_version),
                          "--device-profile", str(args.device_profile), "--output-dir", str(modules)],
                         capture_output=True, text=True, timeout=args.timeout)
    try:
        info = json.loads(run.stdout)
    except json.JSONDecodeError:
        return None, run.stderr.strip()[-2000:] or "shader-sweep printed no JSON"
    if info.get("status") != "ok":
        return None, json.dumps(info.get("error", info))
    return [(modules / s["spirv"], s.get("unique_hash")) for s in info["stages"] if s.get("spirv")], None


def cmd_record(args):
    out = Path(args.output)
    args.device_profile = device_profile(args, out)
    modules, error = sweep_record(args, Path(args.record), out / "modules")
    if error:
        sys.exit(f"shader-sweep: {error}")
    stages = [Stage(path, shader_hash=h) for path, h in modules]
    results = compile_pipeline(stages, args, out)
    counts = write_report(out, f"radv_offline: record {Path(args.record).name}", results,
                          stage_findings(stages, results))
    print(f"{out}/summary.md: {counts['error']} errors, {counts['warn']} warnings")
    return failed(results)


def cmd_cache(args):
    out = Path(args.output)
    args.device_profile = device_profile(args, out)
    records_dir = out / "records"
    run = subprocess.run([args.shader_sweep, "--list", args.cache, "--output-dir", str(records_dir)],
                         capture_output=True, text=True)
    try:
        records = json.loads(run.stdout).get("records", [])
    except json.JSONDecodeError:
        sys.exit(f"shader-sweep --list: {run.stderr.strip()[-2000:]}")
    wanted = {h.lower() for h in args.program or []}

    def sweep(record):
        name = Path(record["file"]).stem
        modules, error = sweep_record(args, records_dir / Path(record["file"]).name, out / "modules" / name)
        if modules is None or not (args.check_all or wanted & {h for _, h in modules}):
            shutil.rmtree(out / "modules" / name, ignore_errors=True)
        return name, modules, error

    def check(item):
        name, path, shader_hash = item
        try:
            return [{**f, "record": name} for f in stage_findings([Stage(path, shader_hash=shader_hash)], [])]
        except Exception as error:  # one unreadable module must not end a sweep of thousands
            return [{"hash": shader_hash, "stage": "?", "record": name,
                     **finding("checker-failed", "error", f"{type(error).__name__}: {error}")}]

    # The listing has no program hashes, so every record is translated to find the selected ones.
    with ThreadPoolExecutor(args.jobs) as pool:
        swept = list(pool.map(sweep, records))
    selected, broken, seen, checks = [], [], set(), []
    for name, modules, error in swept:
        if error:
            broken.append({"record": name, "error": error[:500]})
            continue
        if wanted & {h for _, h in modules}:
            selected.append((name, modules))
        if args.check_all or wanted & {h for _, h in modules}:
            for path, shader_hash in modules:
                if shader_hash not in seen:
                    seen.add(shader_hash)
                    checks.append((name, path, shader_hash))
    findings = []
    with ThreadPoolExecutor(args.jobs) as pool:
        for result in pool.map(check, checks):
            findings += result
    done, rows = [], []
    for name, modules in selected:
        stages = [Stage(path, shader_hash=h) for path, h in modules]
        target = out / "pipelines" / name
        results = compile_pipeline(stages, args, target)
        write_report(target, f"radv_offline: record {name}", results, stage_findings(stages, results))
        done.append({"record": name, "hashes": [s.hash for s in stages],
                     "ok": sum(r["status"] == "ok" for r in results), "runs": len(results)})
        rows += [{**r, "chip": f"{name} {r['chip']}"} for r in results]
    (out / "records.json").write_text(json.dumps({"compiled": done, "not_compiled": broken}, indent=1))
    write_report(out, f"radv_offline: cache {Path(args.cache).name}", rows, findings)
    by_check = Counter((f["severity"], f["check"]) for f in findings)
    print(f"{len(records)} records, {len(seen)} programs checked, {len(broken)} records the offline compiler "
          f"can't emit, {len(done)} pipelines compiled, {sum(r['status'] != 'ok' for r in rows)} compiles "
          f"failed. Findings: " + ", ".join(f"{c} {s} {k}" for (s, k), c in sorted(by_check.items())))
    return failed(rows)


def cmd_foz(args):
    out = Path(args.output)
    asm = out / "spirv"
    asm.mkdir(parents=True, exist_ok=True)
    subprocess.run([tool("fossilize-disasm"), args.database, "--target", "asm", "--module-only", "--output", str(asm)],
                   check=True, capture_output=True)
    findings = []
    for path in sorted(asm.iterdir()):
        m = Module(path.read_text(errors="replace"), path.name)
        if m.entries:
            findings += [{"hash": path.stem, "stage": STAGES.get(m.entries[0][0], ("?",))[0], **f}
                         for f in check_module(m)]
    rows = []
    for chip in args.chips:
        for variant, env in parse_variants(args.variant):
            target = out / (chip if variant == "default" else f"{chip}+{variant}")
            target.mkdir(exist_ok=True)
            run = subprocess.run([tool("fossilize-disasm"), args.database, "--target", "isa", "--output", str(target)],
                                 capture_output=True, text=True, env=driver_env(chip, env), timeout=args.timeout)
            rows.append({"chip": chip, "variant": variant, "ok": run.returncode == 0, "error": run.stderr[-2000:]})
    counts = write_report(out, f"radv_offline: {Path(args.database).name}", [], findings)
    print(f"{out}: ISA for " + ", ".join(r["chip"] + ("" if r["ok"] else " (failed)") for r in rows) +
          f"; {counts['error']} errors, {counts['warn']} warnings in the SPIR-V")
    return 0 if all(r["ok"] for r in rows) else 1


def cmd_check(args):
    findings = stage_findings([Stage(item) for item in args.modules], [])
    counts = write_report(Path(args.output), "radv_offline: SPIR-V checks", [], findings)
    print(f"{args.output}/summary.md: {counts['error']} errors, {counts['warn']} warnings, {counts['info']} notes")
    return 1 if counts["error"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def compile_options(p):
        p.add_argument("--chip", dest="chips", action="append", choices=sorted(CHIPS) + sorted(SHIM_NAMES),
                       help="AMD chip, repeatable (default: all six)")
        p.add_argument("--variant", action="append", metavar="NAME:VAR=VALUE[,...]",
                       help="also compile with these driver variables, e.g. wave32:RADV_PERFTEST=pswave32")
        p.add_argument("--color", action="append", metavar="FORMAT",
                       help="colour attachment format per location, a name or VkFormat number "
                            "(default R16G16B16A16_SFLOAT, or R32G32B32A32 for integer outputs)")
        p.add_argument("--depth", metavar="FORMAT", help="depth attachment format")
        p.add_argument("--subgroup", choices=("lichii", "32", "default"), default="lichii",
                       help="lichii (default): ask for 32 where the shader uses subgroup operations")
        p.add_argument("--robust", choices=("lichii", "none"), default="lichii",
                       help="lichii (default): robustBufferAccess on, robustness2 access off")
        p.add_argument("--timeout", type=int, default=300)
        p.add_argument("-o", "--output", required=True)

    p = sub.add_parser("spirv", help="compile one pipeline from SPIR-V modules")
    p.add_argument("modules", nargs="+", metavar="MODULE.spv[:entry]")
    p.add_argument("--spec", action="append", metavar="STAGE:ID=VALUE",
                   help="specialization constant for the stage at that position (0-based)")
    compile_options(p)
    p.set_defaults(run=cmd_spirv)

    p = sub.add_parser("record", help="compile one pipeline cache record")
    p.add_argument("record")
    p.add_argument("--shader-sweep", required=True, help="the shader-sweep binary")
    p.add_argument("--cache-version", type=int, default=54)
    compile_options(p)
    p.set_defaults(run=cmd_record)

    p = sub.add_parser("cache", help="check the records of a cache and compile the selected ones")
    p.add_argument("cache")
    p.add_argument("--shader-sweep", required=True, help="the shader-sweep binary")
    p.add_argument("--cache-version", type=int, default=54)
    p.add_argument("--program", action="append", metavar="HASH",
                   help="compile the records that use this program (its unique hash)")
    p.add_argument("--check-all", action="store_true", help="run the SPIR-V checks on every record")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    compile_options(p)
    p.set_defaults(run=cmd_cache)

    p = sub.add_parser("foz", help="compile a Fossilize database")
    p.add_argument("database")
    compile_options(p)
    p.set_defaults(run=cmd_foz)

    p = sub.add_parser("check", help="only spirv-val and the driver-difference checks")
    p.add_argument("modules", nargs="+")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(run=cmd_check)

    args = parser.parse_args()
    if hasattr(args, "chips"):
        args.chips = [SHIM_NAMES.get(c, c) for c in (args.chips or list(CHIPS))]
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
