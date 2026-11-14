// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <tuple>

#include <fmt/format.h>

#include "common/fs/path_util.h"
#include "common/logging.h"
#include "video_core/gpu_split.h"
#include "video_core/macro_turing.h"
#include "video_core/nx2_list_macro_trace.h"

namespace VideoCommon::ListMacroTrace {

namespace {

using Effects = Tegra::TuringMacroInterpreter::Effects;

constexpr u32 MacroMethods = 0xe00;
constexpr u32 SamplesPerMacro = 16;
constexpr size_t MaxDumps = 64;
constexpr size_t MaxLandings = 64;

// Inline upload (both engines) and Maxwell3D's constant buffer load registers.
constexpr u32 UploadLineLength = 0x60;
constexpr u32 UploadLineCount = 0x61;
constexpr u32 UploadAddressHigh = 0x62;
constexpr u32 UploadAddressLow = 0x63;
constexpr u32 UploadPitch = 0x64;
constexpr u32 UploadExec = 0x6c;
constexpr u32 UploadData = 0x6d;
constexpr u32 ConstBufferAddressHigh = 0x8e1;
constexpr u32 ConstBufferAddressLow = 0x8e2;
constexpr u32 ConstBufferOffset = 0x8e3;
constexpr u32 ConstBufferData = 0x8e4;
constexpr u32 ConstBufferDataEnd = 0x8f4;

enum class Content : u8 { Commands, MacroData, MethodData, Count };

constexpr std::array<const char*, static_cast<size_t>(Engine::Count)> engine_names{"3D", "compute",
                                                                                   "other"};

struct PathCounts {
    std::array<u64, static_cast<size_t>(Path::Count)> entries{};
    u64 waited_ns{};
    u64 words{};

    void Add(Path path, u64 ns, u32 count) {
        ++entries[static_cast<size_t>(path)];
        waited_ns += ns;
        words += count;
    }
    u64 Total() const {
        return entries[0] + entries[1] + entries[2];
    }
    u64 Waited() const {
        return entries[static_cast<size_t>(Path::Waited)];
    }
};

/// Words a macro call still had due when a GPU-written entry began and when it ended: the call's
/// parameters [total - before, total - after) came from the entry (all from its start if before
/// is Whole).
struct GpuRange {
    static constexpr u32 Whole = ~0U;
    u32 before;
    u32 after;
};

/// The GPU-written entry being parsed.
struct EntryState {
    bool active{};
    Path path{};
    u64 waited_ns{};
    const void* engine{}; ///< The engine whose call, begun before the entry, it continues
    u32 before{};         ///< That call's words due at the entry's start
    bool attributed{};    ///< A call took the entry's wait
};

/// A macro call that GPU-written entries fed and that has not run yet.
struct PendingCall {
    u32 split_starts{}; ///< GpuSplit::Starts then; one from an earlier start ran elsewhere
    std::vector<GpuRange> ranges;
    u64 entries{}; ///< Entries not yet counted for a macro, and their waits
    u64 waited{};
    u64 waited_ns{};
};

struct MacroWindow {
    u64 calls{};
    u64 entries{};
    u64 waited{};
    u64 waited_ns{};
    u64 continued{};
    u64 parameters{};
    u64 gpu_words{};
    u64 on_back{};
    std::array<u64, static_cast<size_t>(Forwarding::Count)> checks{};
    u64 other_maps{}; ///< Forwarding calls whose map differs from the macro's first one
    u64 reader_calls{}; ///< Calls that read a data RAM word a call fed by an entry stored
    struct ReaderNote {
        u64 calls{};
        std::optional<u64> address; ///< The last address it was uploaded or loaded to
    };
    std::map<std::string, ReaderNote> reader_notes; ///< By word: who stored it, what carried it
};

struct Macro {
    Engine engine{};
    u64 hash{};
    u32 start{};
    u32 method{};
    u32 samples{};
    u32 reader_samples{};
    u32 mismatches_logged{};
    std::string dump; ///< Its dump file, empty if not written
    std::optional<std::vector<ForwardedWord>> map; ///< The first forwarding call's
    MacroWindow window;
};

struct CodeInfo {
    u64 hash{};
    Effects effects;
};

/// A data RAM word a call fed by a GPU-written entry stored.
struct Stored {
    Engine engine{};
    u64 hash{};
    u32 method{};
    u32 value{};
};

struct State {
    std::array<PathCounts, static_cast<size_t>(Content::Count)> contents{};
    std::map<std::tuple<Content, Engine, u32>, PathCounts> methods;
    std::map<std::pair<Engine, u64>, Macro> macros;
    std::map<std::tuple<const u32*, u64, u32>, CodeInfo> code;
    EntryState current;
    std::map<const void*, PendingCall> pending; ///< By engine
    size_t dumps{};
    u64 window_start{};
    std::map<std::pair<const u32*, u32>, Stored> stored; ///< By data RAM and address
    Call call;
    Macro* call_macro{};
    bool call_fed{};
    bool call_sampled{};
    const CodeInfo* call_info{};
    u64 call_generation{};
    u32 call_method{};
    std::span<const u32> call_live_data_ram;
    std::array<u32, 8> call_registers{}; ///< 0x60-0x64 and 0x8e1-0x8e3 as the call began
    std::span<const u32> call_code;
    std::vector<u32> call_parameters;
    std::vector<u32> call_data_ram;
    std::vector<u32> call_gpu_words; ///< Indices of the parameters GPU-written entries held
};

State& Get() {
    static State state;
    return state;
}

/// An address a reader uploaded or loaded a stored data RAM word to, and what the back found
/// there. Shared with the back, which loads QMDs and binds constant buffers.
struct Landing {
    u64 address{};
    Engine engine{};
    u64 reader{};
    u32 word{}; ///< Its data RAM address
};

using LandingKey = std::tuple<Engine, u64, std::string>; ///< Reader and what held which word

struct Landings {
    std::mutex mutex;
    std::vector<Landing> recent; ///< Oldest first
    std::map<LandingKey, u64> seen;
    std::set<LandingKey> ever;
    std::vector<LandingKey> first; ///< Seen for the first time, for the readers' dump files
};

Landings& GetLandings() {
    static Landings landings;
    return landings;
}

void AddLanding(const Landing& landing) {
    Landings& landings = GetLandings();
    std::scoped_lock lock{landings.mutex};
    std::erase_if(landings.recent, [&](const Landing& l) { return l.address == landing.address; });
    if (landings.recent.size() == MaxLandings) {
        landings.recent.erase(landings.recent.begin());
    }
    landings.recent.push_back(landing);
}

/// Counts each recent landing in [address, address + size) as held by `what` at its offset.
void FindLandings(u64 address, u64 size, const auto& what) {
    Landings& landings = GetLandings();
    std::scoped_lock lock{landings.mutex};
    for (const Landing& landing : landings.recent) {
        if (landing.address < address || landing.address - address >= size) {
            continue;
        }
        LandingKey key{landing.engine, landing.reader,
                       fmt::format("data RAM {:#x} -> {}", landing.word,
                                   what(landing.address - address))};
        ++landings.seen[key];
        if (landings.ever.insert(key).second) {
            landings.first.push_back(std::move(key));
        }
    }
}

constexpr std::array<const char*, static_cast<size_t>(Forwarding::Count)> forwarding_names{
    "forward", "steer", "transform", "data RAM", "unreproduced"};

std::filesystem::path DumpDirectory() {
    const char* const directory = std::getenv("EDEN_NX2_DEBUG_DIR");
    if (directory != nullptr && *directory != '\0') {
        return std::filesystem::path{directory};
    }
    return Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir) / "list-macros";
}

std::string DumpCode(Engine engine, u64 hash, u32 method, u32 start, u64 generation,
                     std::span<const u32> code, const Effects& effects) {
    const auto directory = DumpDirectory();
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    const auto path =
        directory /
        fmt::format("list-macro-{}-{:016x}.txt", engine == Engine::Maxwell3D ? "3d" : "compute",
                    hash);
    std::ofstream file{path};
    if (!file) {
        LOG_ERROR(HW_GPU, "List macro trace: cannot write {}", path.string());
        return {};
    }
    file << fmt::format("{} macro {:016x} at {:#x} (method {:#x}, code generation {})\n",
                        engine_names[static_cast<size_t>(engine)], hash, start, method,
                        generation);
    file << fmt::format("Methods it can write: {:#x}{}; data RAM: {}\n",
                        fmt::join(effects.methods, ","),
                        effects.methods_known ? "" : " and computed ones",
                        effects.data_ram ? "used" : "not used");
    file << "Instructions it can run (code[pc*3], code[pc*3+1], code[pc*3+2], then the fields of "
            "TuringMacroInterpreter::Decode, lane 0/lane 1):\n";
    for (const u32 pc : effects.instructions) {
        const auto d = Tegra::TuringMacroInterpreter::Decode(
            {code[pc * 3 + 2], code[pc * 3 + 1], code[pc * 3]});
        file << fmt::format(
            "+{:#05x} pc {:#06x}: {:08x} {:08x} {:08x} end {} pred {}/{} op {}/{} dst {}/{} a "
            "{}/{} b {}/{} imm {:#x}/{:#x} method {}/{} emit {}/{}\n",
            pc - start, pc, code[pc * 3], code[pc * 3 + 1], code[pc * 3 + 2], d.end ? 1 : 0,
            d.pred_mode, d.pred_reg, d.op[0], d.op[1], d.dst[0], d.dst[1], d.src_a[0], d.src_a[1],
            d.src_b[0], d.src_b[1], d.imm[0], d.imm[1], d.out_method[0], d.out_method[1],
            d.out_emit[0], d.out_emit[1]);
    }
    file << "Calls:\n";
    LOG_INFO(HW_GPU, "List macro trace: wrote {} macro {:016x} to {}",
             engine_names[static_cast<size_t>(engine)], hash, path.string());
    return path.string();
}

struct Run {
    bool success{};
    std::vector<std::pair<u32, u32>> writes;
    std::vector<std::pair<u32, u32>> data_ram_writes;
};

/// Runs a call that read registers `reads` again, with the load FIFO `words` and the data RAM
/// starting as `data_ram`. A register is read back as the run last wrote it, else as the call read
/// it. A run of tags (`real` set) stops at the first write the real run did not make: a macro that
/// loops on a tag would otherwise write until the step limit (3 ms a run).
Run Replay(std::span<const u32> code, u32 start, std::span<const u32> words,
           std::span<const std::pair<u32, u32>> reads, std::span<const u32> data_ram,
           const Run* real) {
    static Tegra::TuringMacroInterpreter interpreter;
    // A tag the macro loops on without writing would otherwise run a million instructions.
    interpreter.step_limit = 1U << 16;
    Run result;
    interpreter.dram.fill(0);
    std::copy_n(data_ram.begin(), std::min(data_ram.size(), interpreter.dram.size()),
                interpreter.dram.begin());
    interpreter.active_dram = &interpreter.dram;
    std::vector<u32> fifo(words.begin(), words.end());
    size_t next_read = 0;
    bool reads_match = true;
    const auto executed = interpreter.Execute(
        code, start, fifo,
        [&](u32 address) {
            if (next_read >= reads.size() || reads[next_read].first != address) {
                reads_match = false;
                return 0U;
            }
            // The engines apply a macro's writes before its next read: a register the call
            // wrote reads back what this run wrote (which must be what the call read).
            const auto written = std::ranges::find(result.writes.rbegin(), result.writes.rend(),
                                                   address, &std::pair<u32, u32>::first);
            if (written == result.writes.rend()) {
                return reads[next_read++].second;
            }
            if (real == nullptr && written->second != reads[next_read].second) {
                reads_match = false;
            }
            ++next_read;
            return written->second;
        },
        [&](u32 method, u32 value) {
            if (method >= MacroMethods) {
                return false;
            }
            if (real != nullptr && (result.writes.size() >= real->writes.size() ||
                                    real->writes[result.writes.size()].first != method)) {
                return false;
            }
            result.writes.emplace_back(method, value);
            return true;
        });
    result.success = executed.success && reads_match && next_read == reads.size();
    result.data_ram_writes = interpreter.data_ram_writes;
    return result;
}

std::string Pairs(std::span<const std::pair<u32, u32>> pairs) {
    std::string text;
    for (const auto& [address, value] : pairs) {
        text += fmt::format("{}{:#x}={:#x}", text.empty() ? "" : " ", address, value);
    }
    return text.empty() ? "none" : text;
}

std::string MapText(std::span<const ForwardedWord> map) {
    std::string text;
    for (const ForwardedWord& word : map) {
        text += fmt::format("{}word {} -> {:#x} (write {})", text.empty() ? "" : ", ", word.word,
                            word.method, word.write);
    }
    return text.empty() ? "no word written" : text;
}

void Verify(State& state, Macro& macro, const Call& call) {
    const auto check =
        CheckForwarding(state.call_code, macro.start, state.call_parameters, state.call_gpu_words,
                        call.reads, call.writes, state.call_data_ram);
    MacroWindow& w = macro.window;
    bool mismatch = check.result != Forwarding::Forwards;
    if (!mismatch) {
        if (!macro.map) {
            macro.map = check.map;
        } else if (*macro.map != check.map) {
            ++w.other_maps;
            mismatch = true;
        }
    }
    ++w.checks[static_cast<size_t>(check.result)];
    if (mismatch && macro.mismatches_logged < 16) {
        ++macro.mismatches_logged;
        LOG_WARNING(HW_GPU,
                    "List macro verify: {} macro {:016x} at {:#x} does not only forward its "
                    "GPU-written words {} ({}{}): parameters {:08x}, writes {}",
                    engine_names[static_cast<size_t>(macro.engine)], macro.hash, macro.start,
                    fmt::join(state.call_gpu_words, ","),
                    forwarding_names[static_cast<size_t>(check.result)],
                    check.result == Forwarding::Forwards
                        ? fmt::format(", {} instead of {}", MapText(check.map), MapText(*macro.map))
                        : std::string{},
                    fmt::join(state.call_parameters, " "), Pairs(call.writes));
    }
}

/// A call that read data RAM words calls fed by GPU-written entries stored (`hits`, address and
/// value): the writes that carry each word as is (two replays with the words tagged) and where an
/// inline upload or a 3D constant buffer load puts them.
void NoteReader(State& state, Macro& macro, const Call& call, DataRamAccesses data_ram_reads,
                std::span<const std::pair<u32, u32>> hits) {
    // The data RAM as the call found it: each word it read as it first read it.
    std::vector<u32> before(state.call_live_data_ram.begin(), state.call_live_data_ram.end());
    std::vector<u32> restored;
    for (const auto& [address, value] : data_ram_reads) {
        if (address < before.size() && std::ranges::find(restored, address) == restored.end()) {
            restored.push_back(address);
            before[address] = value;
        }
    }
    const auto tag_a = [](u32 address) { return 0xa5000357U | (address << 12); };
    const auto tag_b = [](u32 address) { return 0x5a000468U | (address << 12); };
    std::vector<u32> tags_a = before;
    std::vector<u32> tags_b = before;
    for (const auto& [address, value] : hits) {
        tags_a[address] = tag_a(address);
        tags_b[address] = tag_b(address);
    }
    const auto& code = state.call_code;
    const auto& parameters = state.call_parameters;
    const Run real = Replay(code, macro.start, parameters, call.reads, before, nullptr);
    const Run a = Replay(code, macro.start, parameters, call.reads, tags_a, &real);
    const Run b = Replay(code, macro.start, parameters, call.reads, tags_b, &real);
    const bool steered = !a.success || !b.success || a.writes.size() != real.writes.size() ||
                         b.writes.size() != real.writes.size();
    const char* const failure = !real.success || real.writes != call.writes ? "not reproduced"
                                : steered                                  ? "steers the call"
                                                                           : nullptr;
    // Where each write puts its value: a data word of a linear inline upload the call began
    // (Upload::State: line by line, each at the next pitch), or a 3D constant buffer load word.
    std::vector<std::optional<u64>> lands(real.writes.size());
    const auto& r = state.call_registers;
    struct Upload {
        u64 address;
        u32 line_length;
        u32 line_count;
        u32 pitch;
        u32 offset;
    };
    std::array<u32, 5> upload_state{r[0], r[1], r[2], r[3], r[4]};
    std::optional<Upload> upload;
    bool upload_began = false;
    std::vector<bool> begun_before(real.writes.size());
    u64 const_buffer = (u64{r[5]} << 32) | r[6];
    u32 const_buffer_offset = r[7];
    for (size_t i = 0; i < real.writes.size(); ++i) {
        const auto [method, value] = real.writes[i];
        if (method >= UploadLineLength && method <= UploadPitch) {
            upload_state[method - UploadLineLength] = value;
        } else if (method == UploadExec) {
            upload_began = true;
            const auto& [length, count, high, low, pitch] = upload_state;
            upload = (value & 1) != 0 && length != 0
                         ? std::optional{Upload{(u64{high} << 32) | low, length, count, pitch, 0}}
                         : std::nullopt;
        } else if (method == UploadData && !upload_began) {
            begun_before[i] = true;
        } else if (method == UploadData && upload) {
            const u32 line = upload->offset / upload->line_length;
            if (line < upload->line_count) {
                lands[i] = upload->address + u64{line} * upload->pitch +
                           upload->offset % upload->line_length;
            }
            upload->offset += 4;
        } else if (macro.engine != Engine::Maxwell3D) {
            continue;
        } else if (method == ConstBufferAddressHigh) {
            const_buffer = (u64{value} << 32) | (const_buffer & 0xffffffffU);
        } else if (method == ConstBufferAddressLow) {
            const_buffer = (const_buffer & ~u64{0xffffffffU}) | value;
        } else if (method == ConstBufferOffset) {
            const_buffer_offset = value;
        } else if (method >= ConstBufferData && method < ConstBufferDataEnd) {
            lands[i] = const_buffer + const_buffer_offset;
            const_buffer_offset += 4;
        }
    }
    ++macro.window.reader_calls;
    std::string sample;
    for (const auto& [address, value] : hits) {
        const Stored& stored = state.stored.at({state.call_live_data_ram.data(), address});
        std::string carried;
        std::optional<u64> landing;
        for (size_t i = 0; i < real.writes.size() && failure == nullptr; ++i) {
            if (real.writes[i].second != value || a.writes[i].second != tag_a(address) ||
                b.writes[i].second != tag_b(address)) {
                continue;
            }
            carried += fmt::format("{}write {} to {:#x}{}", carried.empty() ? "" : ", ", i,
                                   real.writes[i].first,
                                   begun_before[i] ? " (an upload begun before the call)" : "");
            if (lands[i]) {
                landing = lands[i];
                AddLanding({*lands[i], macro.engine, macro.hash, address});
            }
        }
        const std::string note = fmt::format(
            "{:#x} (stored by {} {:016x}, method {:#x}) -> {}", address,
            engine_names[static_cast<size_t>(stored.engine)], stored.hash, stored.method,
            failure != nullptr ? failure : carried.empty() ? "no write as is" : carried.c_str());
        auto& reader_note = macro.window.reader_notes[note];
        ++reader_note.calls;
        if (landing) {
            reader_note.address = landing;
        }
        sample += fmt::format("\n  {}{}", note,
                              landing ? fmt::format(", at {:#x}", *landing) : std::string{});
    }
    if (macro.dump.empty() && state.dumps < MaxDumps) {
        ++state.dumps;
        macro.dump = DumpCode(macro.engine, macro.hash, state.call_method, macro.start,
                              state.call_generation, code, state.call_info->effects);
    }
    if (macro.dump.empty() || macro.reader_samples >= SamplesPerMacro) {
        return;
    }
    ++macro.reader_samples;
    std::ofstream file{macro.dump, std::ios::app};
    file << fmt::format("reader call {}: method {:#x}, {} parameters {:08x}\n  data RAM read: "
                        "{}\n  registers read: {}\n  methods written: {}{}\n",
                        macro.reader_samples, state.call_method, parameters.size(),
                        fmt::join(parameters, " "), Pairs(data_ram_reads), Pairs(call.reads),
                        Pairs(call.writes), sample);
}

/// A call that ran on the interpreter: its reads of stored words, and what it stored or overwrote.
void FollowDataRam(State& state, Macro& macro, const Call& call, DataRamAccesses data_ram_reads,
                   DataRamAccesses data_ram_writes) {
    const u32* const data_ram = state.call_live_data_ram.data();
    std::vector<std::pair<u32, u32>> hits;
    for (const auto& [address, value] : data_ram_reads) {
        const auto it = state.stored.find({data_ram, address});
        if (it != state.stored.end() && it->second.value == value &&
            std::ranges::find(hits, address, &std::pair<u32, u32>::first) == hits.end()) {
            hits.emplace_back(address, value);
        }
    }
    if (!hits.empty()) {
        NoteReader(state, macro, call, data_ram_reads, hits);
    }
    for (const auto& [address, value] : data_ram_writes) {
        if (state.call_fed) {
            state.stored[{data_ram, address}] = {macro.engine, macro.hash, state.call_method,
                                                 value};
        } else {
            state.stored.erase({data_ram, address});
        }
    }
}

/// The first time the back found a reader's word somewhere: a line in the reader's dump file.
void DumpFirstLandings(State& state) {
    std::vector<LandingKey> first;
    {
        Landings& landings = GetLandings();
        std::scoped_lock lock{landings.mutex};
        first.swap(landings.first);
    }
    for (const auto& [engine, hash, what] : first) {
        const auto it = state.macros.find({engine, hash});
        if (it != state.macros.end() && !it->second.dump.empty()) {
            std::ofstream{it->second.dump, std::ios::app} << "landing: " << what << "\n";
        }
    }
}

void MaybeLog(State& state) {
    const u64 now = Now();
    if (state.window_start == 0) {
        state.window_start = now;
        return;
    }
    if (now - state.window_start < 10'000'000'000ULL) {
        return;
    }
    const double seconds = static_cast<double>(now - state.window_start) / 1e9;
    state.window_start = now;
    const auto ms = [](u64 ns) { return static_cast<double>(ns) / 1e6; };
    PathCounts all;
    for (const PathCounts& counts : state.contents) {
        for (size_t i = 0; i < all.entries.size(); ++i) {
            all.entries[i] += counts.entries[i];
        }
        all.waited_ns += counts.waited_ns;
    }
    const auto& contents = state.contents;
    LOG_INFO(HW_GPU,
             "List macro trace in {:.1f} s: {} GPU-written entries, {} after the channel waited "
             "for the host GPU's copy ({:.2f} ms waited), {} from a copy at once, {} after a drain "
             "and flush; commands {} ({} waited), macro data {} ({} waited), other method data {} "
             "({} waited)",
             seconds, all.Total(), all.Waited(), ms(all.waited_ns),
             all.entries[static_cast<size_t>(Path::Copied)],
             all.entries[static_cast<size_t>(Path::Flushed)], contents[0].Total(),
             contents[0].Waited(), contents[1].Total(), contents[1].Waited(), contents[2].Total(),
             contents[2].Waited());
    std::vector<std::pair<const std::tuple<Content, Engine, u32>*, const PathCounts*>> methods;
    for (const auto& [key, counts] : state.methods) {
        methods.emplace_back(&key, &counts);
    }
    std::ranges::sort(methods, [](const auto& a, const auto& b) {
        return std::pair{a.second->waited_ns, a.second->Total()} >
               std::pair{b.second->waited_ns, b.second->Total()};
    });
    for (size_t i = 0; i < std::min<size_t>(methods.size(), 8); ++i) {
        const auto& [content, engine, method] = *methods[i].first;
        const PathCounts& counts = *methods[i].second;
        LOG_INFO(HW_GPU,
                 "List macro trace: {} of {} method {:#x}: {} entries ({} waited, {:.2f} ms), "
                 "{:.1f} words each",
                 content == Content::MacroData ? "macro data" : "data",
                 engine_names[static_cast<size_t>(engine)], method, counts.Total(),
                 counts.Waited(), ms(counts.waited_ns),
                 static_cast<double>(counts.words) / static_cast<double>(counts.Total()));
    }
    std::vector<Macro*> macros;
    for (auto& [key, macro] : state.macros) {
        if (macro.window.calls != 0) {
            macros.push_back(&macro);
        }
    }
    std::ranges::sort(macros, [](const Macro* a, const Macro* b) {
        return std::pair{a->window.waited_ns, a->window.calls} >
               std::pair{b->window.waited_ns, b->window.calls};
    });
    for (size_t i = 0; i < std::min<size_t>(macros.size(), 8); ++i) {
        const Macro& macro = *macros[i];
        const MacroWindow& w = macro.window;
        LOG_INFO(HW_GPU,
                 "List macro trace: {} macro {:016x} at {:#x} (method {:#x}): {} calls fed by "
                 "GPU-written entries, {} entries ({} waited, {:.2f} ms), {} continued a call begun "
                 "before the entry, {:.1f} parameters and {:.1f} GPU-written words a call, {} ran "
                 "on the back",
                 engine_names[static_cast<size_t>(macro.engine)], macro.hash, macro.start,
                 macro.method, w.calls, w.entries, w.waited, ms(w.waited_ns), w.continued,
                 static_cast<double>(w.parameters) / static_cast<double>(w.calls),
                 static_cast<double>(w.gpu_words) / static_cast<double>(w.calls), w.on_back);
        if (!g_verify) {
            continue;
        }
        const auto& c = w.checks;
        LOG_INFO(HW_GPU,
                 "List macro verify: {} macro {:016x}: {} calls only forward their GPU-written "
                 "words ({}), {} with another map, {} steer, {} transform, {} data RAM, {} "
                 "unreproduced",
                 engine_names[static_cast<size_t>(macro.engine)], macro.hash,
                 c[static_cast<size_t>(Forwarding::Forwards)] - w.other_maps,
                 macro.map ? MapText(*macro.map) : std::string{"no map yet"}, w.other_maps,
                 c[static_cast<size_t>(Forwarding::Steers)],
                 c[static_cast<size_t>(Forwarding::Transforms)],
                 c[static_cast<size_t>(Forwarding::DataRam)],
                 c[static_cast<size_t>(Forwarding::Unreproduced)]);
    }
    std::map<LandingKey, u64> seen;
    {
        Landings& landings = GetLandings();
        std::scoped_lock lock{landings.mutex};
        seen.swap(landings.seen);
    }
    for (const auto& [key, macro] : state.macros) {
        const MacroWindow& w = macro.window;
        std::string landed;
        for (const auto& [landing, count] : seen) {
            if (std::get<0>(landing) == macro.engine && std::get<1>(landing) == macro.hash) {
                landed += fmt::format("{}{} ({} times)", landed.empty() ? "" : ", ",
                                      std::get<2>(landing), count);
            }
        }
        if (w.reader_calls == 0 && landed.empty()) {
            continue;
        }
        std::string words;
        for (const auto& [note, reader_note] : w.reader_notes) {
            words += fmt::format("{}{}, {} calls{}", words.empty() ? "" : "; ", note,
                                 reader_note.calls,
                                 reader_note.address
                                     ? fmt::format(", last at {:#x}", *reader_note.address)
                                     : std::string{});
        }
        LOG_INFO(HW_GPU,
                 "List macro reader: {} macro {:016x} at {:#x} (method {:#x}): {} calls read data "
                 "RAM words that calls fed by GPU-written entries stored: {}; the back found: {}",
                 engine_names[static_cast<size_t>(macro.engine)], macro.hash, macro.start,
                 macro.method, w.reader_calls, words.empty() ? "none in this window" : words,
                 landed.empty() ? "no QMD it loaded or ran and no constant buffer bound" : landed);
    }
    state.contents = {};
    state.methods.clear();
    for (auto& [key, macro] : state.macros) {
        macro.window = {};
    }
}

} // Anonymous namespace

u64 Now() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

void BeginEntry(Path path, u64 waited_ns, Engine engine, const void* engine_object, u32 method,
                u32 method_count, u32 words) {
    State& state = Get();
    const Content content = method_count == 0 ? Content::Commands
                            : engine != Engine::Other && method >= MacroMethods
                                ? Content::MacroData
                                : Content::MethodData;
    state.contents[static_cast<size_t>(content)].Add(path, waited_ns, words);
    if (content != Content::Commands) {
        state.methods[{content, engine, method}].Add(path, waited_ns, words);
    }
    const bool macro_data = content == Content::MacroData;
    state.current = {
        .active = true,
        .path = path,
        .waited_ns = waited_ns,
        .engine = macro_data ? engine_object : nullptr,
        .before = macro_data ? method_count : 0,
        .attributed = false,
    };
}

void EndEntry(const void* macro_engine, u32 method_count) {
    State& state = Get();
    EntryState& entry = state.current;
    if (macro_engine != nullptr && entry.active) {
        PendingCall& call = state.pending[macro_engine];
        if (const u32 starts = GpuSplit::Get().Starts(); call.split_starts != starts) {
            call = {.split_starts = starts};
        }
        call.ranges.push_back({entry.engine == macro_engine ? entry.before : GpuRange::Whole,
                               method_count});
        if (!entry.attributed) {
            entry.attributed = true;
            ++call.entries;
            if (entry.path == Path::Waited) {
                ++call.waited;
                call.waited_ns += entry.waited_ns;
            }
        }
    }
    entry.active = false;
    DumpFirstLandings(state);
    MaybeLog(state);
}

Call* BeginCall(Engine engine, const void* engine_object, u32 method, std::span<const u32> code,
                u64 code_generation, u32 start, std::span<const u32> parameters,
                std::span<const u32> data_ram, std::span<const u32> registers) {
    State& state = Get();
    EntryState& entry = state.current;
    const auto pending_it = state.pending.find(engine_object);
    std::optional<PendingCall> pending;
    if (pending_it != state.pending.end()) {
        if (pending_it->second.split_starts == GpuSplit::Get().Starts()) {
            pending = std::move(pending_it->second);
        }
        state.pending.erase(pending_it);
    }
    const bool fed = entry.active || pending;
    if (!fed) {
        const auto stored = state.stored.lower_bound({data_ram.data(), 0});
        if (stored == state.stored.end() || stored->first.first != data_ram.data()) {
            return nullptr;
        }
    }
    if (static_cast<size_t>(start) * 3 + 2 >= code.size()) {
        return nullptr;
    }
    if (state.code.size() > 4096) {
        state.code.clear();
    }
    auto [it, inserted] = state.code.try_emplace({code.data(), code_generation, start});
    CodeInfo& info = it->second;
    if (inserted) {
        info.effects = Tegra::TuringMacroInterpreter::AnalyzeEffects(code, start);
        info.hash = Tegra::TuringMacroInterpreter::CodeHash(code, start, info.effects);
    }
    if (!fed && !info.effects.data_ram) {
        return nullptr;
    }
    Macro& macro = state.macros[{engine, info.hash}];
    macro.engine = engine;
    macro.hash = info.hash;
    macro.start = start;
    macro.method = method;
    state.call_macro = &macro;
    state.call_fed = fed;
    state.call_info = &info;
    state.call_generation = code_generation;
    state.call_method = method;
    state.call_live_data_ram = data_ram;
    const auto reg = [&registers](u32 index) {
        return index < registers.size() ? registers[index] : 0U;
    };
    state.call_registers = {reg(UploadLineLength),       reg(UploadLineCount),
                            reg(UploadAddressHigh),      reg(UploadAddressLow),
                            reg(UploadPitch),            reg(ConstBufferAddressHigh),
                            reg(ConstBufferAddressLow), reg(ConstBufferOffset)};
    Call& call = state.call;
    call.data_ram = info.effects.data_ram;
    call.reads.clear();
    call.writes.clear();
    if (!fed) {
        state.call_sampled = false;
        state.call_gpu_words.clear();
        call.capture = true;
        state.call_code = code;
        state.call_parameters.assign(parameters.begin(), parameters.end());
        return &call;
    }
    // The parameters GPU-written entries held: those due when one began and not yet when it
    // ended; all of a call that began and ran inside the current one.
    std::vector<GpuRange> ranges = pending ? pending->ranges : std::vector<GpuRange>{};
    if (entry.active) {
        ranges.push_back({entry.engine == engine_object ? entry.before : GpuRange::Whole, 0});
        if (entry.engine == engine_object) {
            entry.engine = nullptr;
        }
    }
    const u32 total = static_cast<u32>(parameters.size());
    std::vector<u32> gpu_words;
    bool continued = false;
    for (const GpuRange& range : ranges) {
        continued |= range.before != GpuRange::Whole;
        const u32 before = std::min(range.before, total);
        for (u32 index = total - before; index < total - std::min(range.after, before); ++index) {
            gpu_words.push_back(index);
        }
    }
    std::ranges::sort(gpu_words);
    gpu_words.erase(std::unique(gpu_words.begin(), gpu_words.end()), gpu_words.end());
    if (macro.dump.empty() && state.dumps < MaxDumps) {
        ++state.dumps;
        macro.dump =
            DumpCode(engine, info.hash, method, start, code_generation, code, info.effects);
    }
    MacroWindow& w = macro.window;
    ++w.calls;
    w.parameters += total;
    w.gpu_words += gpu_words.size();
    w.continued += continued ? 1 : 0;
    if (pending) {
        w.entries += pending->entries;
        w.waited += pending->waited;
        w.waited_ns += pending->waited_ns;
    }
    if (entry.active && !entry.attributed) {
        entry.attributed = true;
        ++w.entries;
        if (entry.path == Path::Waited) {
            ++w.waited;
            w.waited_ns += entry.waited_ns;
        }
    }
    state.call_sampled = !macro.dump.empty() && macro.samples < SamplesPerMacro;
    call.capture = state.call_sampled || call.data_ram || (g_verify && !gpu_words.empty());
    state.call_gpu_words = std::move(gpu_words);
    if (call.capture) {
        state.call_code = code;
        state.call_parameters.assign(parameters.begin(), parameters.end());
        if (g_verify) {
            state.call_data_ram.assign(data_ram.begin(), data_ram.end());
        }
    }
    return &call;
}

void EndCall(Call& call, bool on_back, DataRamAccesses data_ram_reads,
             DataRamAccesses data_ram_writes) {
    State& state = Get();
    Macro* const macro = std::exchange(state.call_macro, nullptr);
    if (macro == nullptr) {
        return;
    }
    if (call.data_ram && !on_back) {
        FollowDataRam(state, *macro, call, data_ram_reads, data_ram_writes);
    }
    if (!state.call_fed) {
        return;
    }
    if (on_back) {
        ++macro->window.on_back;
    } else if (g_verify && call.capture && !state.call_gpu_words.empty()) {
        Verify(state, *macro, call);
    }
    if (!state.call_sampled) {
        return;
    }
    ++macro->samples;
    std::ofstream file{macro->dump, std::ios::app};
    file << fmt::format(
        "call {}: method {:#x}, {} parameters {:08x}, GPU-written: {}\n  registers read: {}\n  "
        "methods written: {}\n",
        macro->samples, macro->method, state.call_parameters.size(),
        fmt::join(state.call_parameters, " "), fmt::join(state.call_gpu_words, ","),
        Pairs(call.reads), on_back ? std::string{"(ran on the back)"} : Pairs(call.writes));
}

void NoteQmd(u64 address, std::span<const u32, 64> qmd) {
    FindLandings(address, sizeof(u32) * qmd.size(), [](u64 offset) {
        const u64 word = offset / sizeof(u32);
        return word >= 12 && word <= 14
                   ? fmt::format("QMD word {} (grid {})", word, "xyz"[word - 12])
                   : fmt::format("QMD word {}", word);
    });
    for (u32 slot = 0; slot < 8; ++slot) {
        if (((qmd[20] >> slot) & 1) == 0) {
            continue;
        }
        const u32 high = qmd[33 + slot * 2];
        FindLandings((u64{high & 0xff} << 32) | qmd[32 + slot * 2], ((high >> 19) & 0x1fff) << 4,
                     [slot](u64 offset) {
                         return fmt::format("compute constant buffer {} +{:#x}", slot, offset);
                     });
    }
}

void NoteConstBufferBind(u32 stage, u32 slot, u64 address, u32 size) {
    FindLandings(address, size, [stage, slot](u64 offset) {
        return fmt::format("3D constant buffer stage {} slot {} +{:#x}", stage, slot, offset);
    });
}

ForwardCheck CheckForwarding(std::span<const u32> code, u32 start, std::span<const u32> parameters,
                             std::span<const u32> gpu_words,
                             std::span<const std::pair<u32, u32>> reads,
                             std::span<const std::pair<u32, u32>> writes,
                             std::span<const u32> data_ram) {
    const auto run = [&](std::span<const u32> words, const Run* real) {
        return Replay(code, start, words, reads, data_ram, real);
    };
    const Run real = run(parameters, nullptr);
    if (!real.success || !std::ranges::equal(real.writes, writes)) {
        return {Forwarding::Unreproduced, {}};
    }
    std::vector<u32> tags_a(parameters.begin(), parameters.end());
    std::vector<u32> tags_b = tags_a;
    for (const u32 k : gpu_words) {
        if (k >= parameters.size()) {
            return {Forwarding::Unreproduced, {}};
        }
        tags_a[k] = 0x5a001357U | (k << 16);
        tags_b[k] = 0x25002468U | (k << 16);
    }
    const Run a = run(tags_a, &real);
    const Run b = run(tags_b, &real);
    const auto same_methods = [&real](const Run& tagged) {
        return tagged.success && tagged.writes.size() == real.writes.size() &&
               std::ranges::equal(tagged.writes, real.writes,
                                  [](const auto& x, const auto& y) { return x.first == y.first; });
    };
    if (!same_methods(a) || !same_methods(b)) {
        return {Forwarding::Steers, {}};
    }
    ForwardCheck check{Forwarding::Forwards, {}};
    for (size_t i = 0; i < real.writes.size(); ++i) {
        const u32 value = real.writes[i].second;
        if (a.writes[i].second == value && b.writes[i].second == value) {
            continue;
        }
        const auto forwarded = std::ranges::find_if(gpu_words, [&](u32 k) {
            return a.writes[i].second == tags_a[k] && b.writes[i].second == tags_b[k] &&
                   value == parameters[k];
        });
        if (forwarded == gpu_words.end()) {
            return {Forwarding::Transforms, {}};
        }
        check.map.push_back({*forwarded, static_cast<u32>(i), real.writes[i].first});
    }
    if (a.data_ram_writes != real.data_ram_writes || b.data_ram_writes != real.data_ram_writes) {
        return {Forwarding::DataRam, {}};
    }
    return check;
}

} // namespace VideoCommon::ListMacroTrace
