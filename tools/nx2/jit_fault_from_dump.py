#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Recover the real x64 fault from an abort() minidump (eden_crash.dmp).

crash_handler.cpp writes eden_crash.dmp from its SIGABRT handler with a fake 0xC0000409
exception, so the exception stream says nothing about the fault that led to abort(). When the
abort came from an SEH handler (dynarmic's FastmemCallback UNREACHABLE, say), the original
EXCEPTION_RECORD and CONTEXT that KiUserExceptionDispatcher pushed are still in the dumped stack
of that thread. This scans the stack for them, pairs a record with the CONTEXT whose Rip is its
ExceptionAddress, and prints the code, RIP, access type and address, registers and module.

    python3 -I tools/nx2/jit_fault_from_dump.py eden_crash.dmp              # exception thread
    python3 -I tools/nx2/jit_fault_from_dump.py eden_crash.dmp --threads    # list threads
    python3 -I tools/nx2/jit_fault_from_dump.py eden_crash.dmp --thread 1234 --jit-base 0x...

The JIT block base comes from --jit-base, else from the DISPATCHER_CONTEXT RtlDispatchException
left on the stack (its ImageBase is the base dynarmic passed to RtlAddFunctionTable, which is
code.getCode(), the base FastmemCallback subtracts). Standard library only, so it also reads
dumps with streams the minidump package rejects (wine writes stream 0xFFF0).
"""
import argparse
import os
import struct
import sys

CONTEXT_SIZE = 0x4D0
RECORD_SIZE = 0x98
CONTEXT_AMD64 = 0x00100000
CONTEXT_ALLOWED = 0xFC10007F
GPRS = ("rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi",
        "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15")

CODES = {
    0x80000001: "STATUS_GUARD_PAGE_VIOLATION",
    0x80000002: "STATUS_DATATYPE_MISALIGNMENT",
    0x80000003: "STATUS_BREAKPOINT",
    0x80000004: "STATUS_SINGLE_STEP",
    0xC0000005: "STATUS_ACCESS_VIOLATION",
    0xC0000006: "STATUS_IN_PAGE_ERROR",
    0xC000001D: "STATUS_ILLEGAL_INSTRUCTION",
    0xC0000025: "STATUS_NONCONTINUABLE_EXCEPTION",
    0xC000008C: "STATUS_ARRAY_BOUNDS_EXCEEDED",
    0xC000008D: "STATUS_FLOAT_DENORMAL_OPERAND",
    0xC000008E: "STATUS_FLOAT_DIVIDE_BY_ZERO",
    0xC000008F: "STATUS_FLOAT_INEXACT_RESULT",
    0xC0000090: "STATUS_FLOAT_INVALID_OPERATION",
    0xC0000091: "STATUS_FLOAT_OVERFLOW",
    0xC0000092: "STATUS_FLOAT_STACK_CHECK",
    0xC0000093: "STATUS_FLOAT_UNDERFLOW",
    0xC0000094: "STATUS_INTEGER_DIVIDE_BY_ZERO",
    0xC0000095: "STATUS_INTEGER_OVERFLOW",
    0xC0000096: "STATUS_PRIVILEGED_INSTRUCTION",
    0xC00000FD: "STATUS_STACK_OVERFLOW",
    0xC0000374: "STATUS_HEAP_CORRUPTION",
    0xC0000409: "STATUS_STACK_BUFFER_OVERRUN (fail-fast)",
    0xC000041D: "STATUS_FATAL_USER_CALLBACK_EXCEPTION",
    0xE06D7363: "C++ exception",
}
ACCESS = {0: "read", 1: "write", 8: "execute (DEP)"}
STATE = {0x1000: "commit", 0x2000: "reserve", 0x10000: "free"}


def user_address(v):
    return 0x10000 <= v < 0x800000000000


class Dump:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = f.read()
        sig, _, count, rva = struct.unpack_from("<4sIII", self.data, 0)
        if sig != b"MDMP":
            sys.exit(f"{path}: not a minidump")
        self.streams = {}
        for i in range(count):
            kind, size, at = struct.unpack_from("<III", self.data, rva + 12 * i)
            self.streams.setdefault(kind, (at, size))
        self.threads = self._threads()
        self.names = self._thread_names()
        self.start_addresses = self._thread_starts()
        self.modules = self._modules()
        self.segments = self._memory()
        self.regions = self._memory_info()
        self.exception = self._exception()

    def u32(self, at):
        return struct.unpack_from("<I", self.data, at)[0]

    def string(self, rva):
        size = self.u32(rva)
        return self.data[rva + 4:rva + 4 + size].decode("utf-16-le", "replace")

    def _threads(self):
        if 3 not in self.streams:
            return []
        at, _ = self.streams[3]
        threads = []
        for i in range(self.u32(at)):
            tid, _, _, _, teb, start, size, rva, ctx_size, ctx_rva = struct.unpack_from(
                "<IIIIQQIIII", self.data, at + 4 + 48 * i)
            threads.append({"id": tid, "teb": teb, "stack": start, "stack_size": size,
                            "stack_rva": rva, "context_rva": ctx_rva if ctx_size else None})
        return threads

    def _thread_names(self):
        if 24 not in self.streams:
            return {}
        at, size = self.streams[24]
        count = self.u32(at)
        entry = 16 if count and size >= 4 + 16 * count else 12
        names = {}
        for i in range(count):
            tid, rva = struct.unpack_from("<IQ", self.data, at + 4 + entry * i)
            if rva:
                names[tid] = self.string(rva)
        return names

    def _thread_starts(self):
        if 17 not in self.streams:
            return {}
        at, _ = self.streams[17]
        header, entry, count = struct.unpack_from("<III", self.data, at)
        starts = {}
        for i in range(count):
            base = at + header + entry * i
            starts[self.u32(base)] = struct.unpack_from("<Q", self.data, base + 48)[0]
        return starts

    def _modules(self):
        if 4 not in self.streams:
            return []
        at, _ = self.streams[4]
        modules = []
        for i in range(self.u32(at)):
            base, size, _, _, name_rva = struct.unpack_from("<QIIII", self.data, at + 4 + 108 * i)
            modules.append((base, size, self.string(name_rva).replace("\\", "/").split("/")[-1]))
        return modules

    def _memory(self):
        segments = []
        if 5 in self.streams:
            at, _ = self.streams[5]
            for i in range(self.u32(at)):
                start, size, rva = struct.unpack_from("<QII", self.data, at + 4 + 16 * i)
                segments.append((start, size, rva))
        if 9 in self.streams:
            at, _ = self.streams[9]
            count, rva = struct.unpack_from("<QQ", self.data, at)
            for i in range(count):
                start, size = struct.unpack_from("<QQ", self.data, at + 16 + 16 * i)
                segments.append((start, size, rva))
                rva += size
        return segments

    def _memory_info(self):
        if 16 not in self.streams:
            return []
        at, _ = self.streams[16]
        header, entry, count = struct.unpack_from("<IIQ", self.data, at)
        regions = []
        for i in range(count):
            base, _, _, size, state, protect, _ = struct.unpack_from(
                "<QQIxxxxQIII", self.data, at + header + entry * i)
            regions.append((base, size, state, protect))
        return regions

    def _exception(self):
        if 6 not in self.streams:
            return None
        at, _ = self.streams[6]
        tid = self.u32(at)
        return {"thread": tid, "record": parse_record(self.data, at + 8)}

    def module_of(self, address):
        for base, size, name in self.modules:
            if base <= address < base + size:
                return name, address - base
        return None

    def read(self, address, size):
        """The bytes at [address, address + size), joined across adjacent ranges; b"" on a gap."""
        out = bytearray()
        end = address + size
        while address < end:
            segment = next(((s, l, r) for s, l, r in self.segments if s <= address < s + l), None)
            if segment is None:
                return b""
            start, length, rva = segment
            take = min(end, start + length) - address
            out += self.data[rva + address - start:rva + address - start + take]
            address += take
        return bytes(out)

    def region_of(self, address):
        for base, size, state, protect in self.regions:
            if base <= address < base + size:
                return base, size, state, protect
        return None

    def plausible_code(self, address):
        if not user_address(address):
            return False
        if self.regions:
            region = self.region_of(address)
            return region is not None and region[2] == 0x1000
        return True


def parse_record(data, at):
    code, flags, nested, address, count = struct.unpack_from("<IIQQI", data, at)
    info = struct.unpack_from("<15Q", data, at + 32)
    return {"code": code, "flags": flags, "nested": nested, "address": address,
            "params": list(info[:min(count, 15)]), "count": count}


def parse_context(data, at):
    flags, mxcsr, cs = struct.unpack_from("<IIH", data, at + 0x30)
    ss, eflags = struct.unpack_from("<HI", data, at + 0x42)
    regs = dict(zip(GPRS, struct.unpack_from("<16Q", data, at + 0x78)))
    regs["rip"] = struct.unpack_from("<Q", data, at + 0xF8)[0]
    return {"flags": flags, "mxcsr": mxcsr, "cs": cs, "ss": ss, "eflags": eflags, "regs": regs}


def is_context(dump, data, at, record_addresses):
    flags = struct.unpack_from("<I", data, at + 0x30)[0]
    if not flags & CONTEXT_AMD64 or flags & ~CONTEXT_ALLOWED or flags & 3 != 3:
        return None
    ctx = parse_context(data, at)
    if ctx["cs"] != 0x33 or ctx["ss"] != 0x2B or not ctx["eflags"] & 2 or ctx["eflags"] >> 22:
        return None
    if ctx["mxcsr"] >> 16 or not user_address(ctx["regs"]["rsp"]):
        return None
    # An execute fault's RIP can lie outside committed memory; a record at that RIP vouches for it.
    rip = ctx["regs"]["rip"]
    return ctx if rip in record_addresses or dump.plausible_code(rip) else None


def is_record(data, at):
    code = struct.unpack_from("<I", data, at)[0]
    if code not in CODES:
        return None
    rec = parse_record(data, at)
    if rec["flags"] >> 8 or rec["count"] > 15 or not user_address(rec["address"]):
        return None
    if rec["nested"] and not user_address(rec["nested"]):
        return None
    if code == 0xC0000005 and (rec["count"] < 2 or rec["params"][0] not in ACCESS):
        return None
    return rec


def scan(dump, stack, base):
    records = [(base + off, rec) for off in range(0, len(stack) - RECORD_SIZE + 1, 8)
               if (rec := is_record(stack, off))]
    record_addresses = {rec["address"] for _, rec in records}
    contexts = [(base + off, ctx) for off in range(0, len(stack) - CONTEXT_SIZE + 1, 8)
                if (ctx := is_context(dump, stack, off, record_addresses))]
    return contexts, records


def pair(contexts, records):
    pairs = []
    for rec_at, rec in records:
        matches = [(c_at, c) for c_at, c in contexts if c["regs"]["rip"] == rec["address"]]
        matches.sort(key=lambda m: (rec_at - m[0] != 0x4F0, abs(rec_at - m[0])))
        if matches:
            pairs.append((rec_at, rec, matches[0][0], matches[0][1]))
    pairs.sort(key=lambda p: (p[1]["code"] == 0xC0000409, p[0] - p[2] != 0x4F0, p[0]))
    return pairs


def find_dispatcher_context(stack, base, rip, context_addresses):
    """DISPATCHER_CONTEXT: ControlPc, ImageBase, FunctionEntry, EstablisherFrame, TargetIp,
    ContextRecord, LanguageHandler, HandlerData."""
    best = None
    for off in range(0, len(stack) - 64 + 1, 8):
        control_pc, image_base, entry, _, _, ctx, handler = struct.unpack_from("<7Q", stack, off)
        if control_pc != rip or not 0 < rip - image_base < 1 << 32:
            continue
        if not 0 <= entry - image_base < 1 << 32 or not 0 <= handler - image_base < 1 << 32:
            continue
        found = (base + off, image_base, entry, handler, ctx in context_addresses)
        if best is None or found[4] and not best[4]:
            best = found
    return best


def where(dump, address, jit_base):
    module = dump.module_of(address)
    if module:
        return f"{module[0]}+{module[1]:#x}"
    text = "not in a module"
    if jit_base is not None and 0 <= address - jit_base < 1 << 32:
        text += f", JIT block +{address - jit_base:#x}"
    region = dump.region_of(address)
    if region:
        text += f" (region {region[0]:#x}+{region[1]:#x} {STATE.get(region[2], hex(region[2]))}" \
                f" protect {region[3]:#x})"
    return text


def thread_label(dump, tid):
    return dump.names.get(tid, f"thread {tid}")


def list_threads(dump):
    print(f"{len(dump.threads)} threads" + ("" if dump.names else " (no thread name stream)"))
    for t in dump.threads:
        mark = "*" if dump.exception and dump.exception["thread"] == t["id"] else " "
        line = f"{mark} {t['id']:6} {dump.names.get(t['id'], ''):24} stack " \
               f"{t['stack']:#x}+{t['stack_size']:#x}"
        if t["context_rva"] is not None:
            line += f"  rip {where(dump, parse_context(dump.data, t['context_rva'])['regs']['rip'], None)}"
        start = dump.start_addresses.get(t["id"])
        if start:
            line += f"  start {where(dump, start, None)}"
        print(line)


def print_fault(dump, rec_at, rec, ctx_at, ctx, jit_base, jit_source, stack_bytes, stack_base,
                depth):
    regs = ctx["regs"]
    code = rec["code"]
    layout = ", KiUserExceptionDispatcher layout" if rec_at - ctx_at == 0x4F0 else ""
    print(f"EXCEPTION_RECORD @ {rec_at:#x}, CONTEXT @ {ctx_at:#x} "
          f"(record - context = {rec_at - ctx_at:#x}{layout})")
    print(f"  code      {code:#010x} {CODES[code]}")
    flag_names = [n for b, n in ((1, "noncontinuable"), (2, "unwinding"), (0x10, "nested"))
                  if rec["flags"] & b]
    print(f"  flags     {rec['flags']:#x} {' '.join(flag_names)}".rstrip())
    print(f"  rip       {regs['rip']:#018x} {where(dump, regs['rip'], jit_base)}")
    if code in (0xC0000005, 0xC0000006) and len(rec["params"]) >= 2:
        print(f"  access    {ACCESS.get(rec['params'][0], rec['params'][0])} at "
              f"{rec['params'][1]:#018x}")
        if code == 0xC0000006 and len(rec["params"]) >= 3:
            print(f"  status    {rec['params'][2]:#010x}")
    elif rec["params"]:
        print("  params    " + " ".join(f"{p:#x}" for p in rec["params"]))
    if rec["nested"]:
        print(f"  nested    record at {rec['nested']:#x}")
    if jit_base is not None:
        print(f"  jit base  {jit_base:#018x} ({jit_source}); rip offset {regs['rip'] - jit_base:#x}")
    elif not dump.module_of(regs["rip"]):
        print("  jit base  not found; pass --jit-base")
    names = GPRS[:4] + GPRS[6:8] + GPRS[8:] + ("rsp", "rbp")
    for i in range(0, len(names), 4):
        print("  " + "  ".join(f"{n:>3} {regs[n]:016x}" for n in names[i:i + 4]))
    print(f"  rflags {ctx['eflags']:#x}  mxcsr {ctx['mxcsr']:#x}  context flags {ctx['flags']:#x}")

    rsp = regs["rsp"]
    if not depth or not stack_base <= rsp < stack_base + len(stack_bytes):
        return
    print(f"  code addresses on the stack from rsp (first {depth}):")
    shown = 0
    for off in range(rsp - stack_base, len(stack_bytes) - 7, 8):
        value = struct.unpack_from("<Q", stack_bytes, off)[0]
        module = dump.module_of(value)
        if module or jit_base is not None and 0 <= value - jit_base < 1 << 32:
            print(f"    [rsp+{off - (rsp - stack_base):#06x}] {value:#018x} {where(dump, value, jit_base)}")
            shown += 1
            if shown == depth:
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dump")
    parser.add_argument("--threads", action="store_true", help="list threads and exit")
    parser.add_argument("--thread", type=lambda s: int(s, 0),
                        help="thread id to scan (default: the exception stream's thread)")
    parser.add_argument("--jit-base", type=lambda s: int(s, 16), help="JIT code block base (hex)")
    parser.add_argument("--all", action="store_true", help="also print every candidate found")
    parser.add_argument("--depth", type=int, default=12,
                        help="code addresses to print from the faulting rsp (default 12)")
    args = parser.parse_args()

    dump = Dump(args.dump)
    print(f"{os.path.basename(args.dump)}: {len(dump.threads)} threads, {len(dump.modules)} modules, "
          f"{len(dump.segments)} memory ranges, memory info {'yes' if dump.regions else 'no'}")
    if dump.exception:
        rec = dump.exception["record"]
        fake = " (crash_handler.cpp's fake fail-fast record)" if rec["code"] == 0xC0000409 else ""
        print(f"exception stream: thread {dump.exception['thread']} "
              f"({thread_label(dump, dump.exception['thread'])}) code {rec['code']:#010x} at "
              f"{where(dump, rec['address'], None)}{fake}")
    if args.threads:
        list_threads(dump)
        return

    tid = args.thread if args.thread is not None else dump.exception and dump.exception["thread"]
    thread = next((t for t in dump.threads if t["id"] == tid), None)
    if thread is None:
        list_threads(dump)
        sys.exit(f"no thread {tid}; pass --thread")
    # Wine's full-memory dumps leave the stack RVA at 0 and keep the stack in the memory list.
    stack = (dump.data[thread["stack_rva"]:thread["stack_rva"] + thread["stack_size"]]
             if thread["stack_rva"] else dump.read(thread["stack"], thread["stack_size"]))
    base = thread["stack"]
    print(f"scanning thread {tid} ({thread_label(dump, tid)}) stack {base:#x}+{len(stack):#x}")

    contexts, records = scan(dump, stack, base)
    pairs = pair(contexts, records)
    print(f"{len(contexts)} CONTEXT candidates, {len(records)} EXCEPTION_RECORD candidates, "
          f"{len(pairs)} paired")
    if args.all:
        for at, ctx in contexts:
            print(f"  CONTEXT @ {at:#x} flags {ctx['flags']:#x} rip {ctx['regs']['rip']:#x} "
                  f"{where(dump, ctx['regs']['rip'], args.jit_base)}")
        for at, rec in records:
            print(f"  RECORD  @ {at:#x} code {rec['code']:#010x} address {rec['address']:#x}")
    if not pairs:
        sys.exit("no EXCEPTION_RECORD pairs with a CONTEXT on this stack; try --all or --thread")

    context_addresses = {at for at, _ in contexts}
    for i, (rec_at, rec, ctx_at, ctx) in enumerate(pairs if args.all else pairs[:1]):
        print()
        if i == 0 and len(pairs) > 1:
            print(f"fault (most likely of {len(pairs)} pairs; --all prints the rest):")
        jit_base, source = args.jit_base, "--jit-base"
        if jit_base is None and not dump.module_of(ctx["regs"]["rip"]):
            found = find_dispatcher_context(stack, base, ctx["regs"]["rip"], context_addresses)
            if found:
                dc_at, jit_base, entry, handler, _ = found
                source = (f"DISPATCHER_CONTEXT @ {dc_at:#x}: RUNTIME_FUNCTION at +{entry - jit_base:#x},"
                          f" language handler at +{handler - jit_base:#x}")
        print_fault(dump, rec_at, rec, ctx_at, ctx, jit_base, source, stack, base, args.depth)


if __name__ == "__main__":
    main()
