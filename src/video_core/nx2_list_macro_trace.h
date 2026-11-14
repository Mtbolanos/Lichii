// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>
#include <span>
#include <utility>
#include <vector>

#include "common/common_types.h"

// EDEN_NX2_LIST_MACRO_TRACE=1: the command list entries the split's front read from GPU-written
// pages (DmaPusher::ReadGpuWrittenCommands) and the native macros they fed. Every 10 s the GPU
// thread logs the entries by what they held (commands, macro data, other method data) and how they
// were read (from a copy at once, after the channel waited for the host GPU's copy, or after a
// drain and flush), the methods whose data they continued, and the macros, by the hash of their
// code (TuringMacroInterpreter::CodeHash), whose calls took words from them. Each such macro's
// code and its first calls (parameters, registers read, methods written) are written once to
// EDEN_NX2_DEBUG_DIR (else list-macros in the log directory) as list-macro-<engine>-<hash>.txt.
//
// The trace also follows the data RAM words such calls store. A front call that may use the data
// RAM then runs on the interpreter, which reports its data RAM reads and writes. A call that reads
// a stored word is a reader: the trace lines add, by reader, the words it read and who stored them,
// the methods that carry each word as is (a replay with the words tagged), and where an inline
// upload (0x6c/0x6d) or a 3D constant buffer load (0x8e4) puts it. The back reports a QMD it loads
// or runs (PCAS, inline QMD) and a 3D constant buffer it binds that holds such an address. The
// reader's code and first calls go to its dump file.
//
// EDEN_NX2_LIST_MACRO_VERIFY=1 (implies the trace): every such call the front ran is run again
// with its GPU-written words replaced by tags (CheckForwarding), and the trace lines add, by
// macro, how many calls only forward those words to the methods they write, which methods take
// which word, and the calls that do more with them.
namespace VideoCommon::ListMacroTrace {

inline bool g_verify = std::getenv("EDEN_NX2_LIST_MACRO_VERIFY") != nullptr;
inline bool g_enabled = g_verify || std::getenv("EDEN_NX2_LIST_MACRO_TRACE") != nullptr;

/// Called by VideoCore::CreateGPU, after the frontend loaded nx2_switches.txt.
inline void ReadSwitch() {
    g_verify = std::getenv("EDEN_NX2_LIST_MACRO_VERIFY") != nullptr;
    g_enabled = g_verify || std::getenv("EDEN_NX2_LIST_MACRO_TRACE") != nullptr;
}

enum class Path : u8 {
    Copied,  ///< From the copy or guest memory at once
    Waited,  ///< After the channel waited for the host GPU's copy
    Flushed, ///< After a drain and flush (FlushGpuWrittenCommands)
    Count,
};

enum class Engine : u8 { Maxwell3D, Compute, Other, Count };

u64 Now();

/// GPU thread: the pusher parses an entry of `words` words on GPU-written pages (the rest of one
/// resumed after a semaphore acquire, counted again); `engine` (the subchannel's object) was
/// still due `method_count` words of `method` at its start.
void BeginEntry(Path path, u64 waited_ns, Engine engine, const void* engine_object, u32 method,
                u32 method_count, u32 words);
/// The entry ended inside the data of a macro call the front runs for `macro_engine`, still due
/// `method_count` words, unless `macro_engine` is null: that call took words from it too.
void EndEntry(const void* macro_engine, u32 method_count);

struct Call {
    bool capture{};  ///< Note the call's register reads and method writes
    bool data_ram{}; ///< Run it on the interpreter, which notes its data RAM reads and writes
    std::vector<std::pair<u32, u32>> reads;
    std::vector<std::pair<u32, u32>> writes;
};

/// GPU thread, as a front macro call starts with `data_ram` and the engine's `registers` as they
/// are then: the call to note if a GPU-written entry fed it or it may read a data RAM word one of
/// those stored (valid until EndCall), else nullptr. With `data_ram` set, the engine runs it on
/// the interpreter.
Call* BeginCall(Engine engine, const void* engine_object, u32 method, std::span<const u32> code,
                u64 code_generation, u32 start, std::span<const u32> parameters,
                std::span<const u32> data_ram, std::span<const u32> registers);

inline void NoteRead(Call* call, u32 address, u32 value) {
    if (call != nullptr && call->capture) {
        call->reads.emplace_back(address, value);
    }
}

inline void NoteWrite(Call* call, u32 address, u32 value) {
    if (call != nullptr && call->capture) {
        call->writes.emplace_back(address, value);
    }
}

using DataRamAccesses = std::span<const std::pair<u32, u32>>;

/// The call ran; on the split's back in the front's place if `on_back` (its effects unseen). With
/// Call::data_ram, the interpreter's data RAM reads and writes (address, value) in order.
void EndCall(Call& call, bool on_back, DataRamAccesses data_ram_reads = {},
             DataRamAccesses data_ram_writes = {});

/// Any thread: the native QMD scheduler read the QMD at `address` to load or run it.
void NoteQmd(u64 address, std::span<const u32, 64> qmd);

/// Any thread: Maxwell3D bound [address, address + size) to constant buffer `slot` of `stage`.
void NoteConstBufferBind(u32 stage, u32 slot, u64 address, u32 size);

enum class Forwarding : u8 {
    Forwards,     ///< Each write is fixed by the other words or is one GPU-written word as is
    Steers,       ///< The words change which methods it writes or which registers it reads
    Transforms,   ///< A write is computed from a word
    DataRam,      ///< The words change what it stores in the data RAM
    Unreproduced, ///< Run again with its own words, it did not write what it wrote
    Count,
};

struct ForwardedWord {
    u32 word;   ///< Index in the parameters
    u32 write;  ///< Index in the writes
    u32 method; ///< The method written
    bool operator==(const ForwardedWord&) const = default;
};

struct ForwardCheck {
    Forwarding result{};
    std::vector<ForwardedWord> map; ///< With Forwards: where each forwarded word went
};

/// Runs a call that read `reads` and wrote `writes` again, with its parameters, then twice with
/// those at the indices `gpu_words` replaced by two sets of tags. A register is read back as the
/// run last wrote it, else as the call read it; the data RAM starts as `data_ram` was before it.
ForwardCheck CheckForwarding(std::span<const u32> code, u32 start, std::span<const u32> parameters,
                             std::span<const u32> gpu_words,
                             std::span<const std::pair<u32, u32>> reads,
                             std::span<const std::pair<u32, u32>> writes,
                             std::span<const u32> data_ram);

} // namespace VideoCommon::ListMacroTrace
