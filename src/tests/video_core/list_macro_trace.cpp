// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>

#include "video_core/macro_turing.h"
#include "video_core/nx2_list_macro_trace.h"

namespace {

void SetDebugDir(const std::filesystem::path& directory) {
    std::filesystem::remove_all(directory);
#ifdef _WIN32
    _putenv_s("EDEN_NX2_DEBUG_DIR", directory.string().c_str());
#else
    setenv("EDEN_NX2_DEBUG_DIR", directory.string().c_str(), 1);
#endif
}

std::string ReadDump(const std::filesystem::path& directory) {
    for (const auto& file : std::filesystem::directory_iterator{directory}) {
        std::ifstream stream{file.path()};
        std::stringstream text;
        text << stream.rdbuf();
        return text.str();
    }
    return {};
}

} // Anonymous namespace

TEST_CASE("ListMacroTrace: a macro call is noted only when a GPU-written entry fed it",
          "[video_core]") {
    namespace Trace = VideoCommon::ListMacroTrace;
    const auto directory = std::filesystem::temp_directory_path() / "lichii-list-macro-trace";
    SetDebugDir(directory);
    const std::vector<u32> code(9, 0);
    const std::vector<u32> parameters{7, 8, 9};
    const std::vector<u32> data_ram(0xc00);
    int engine{};
    int other_engine{};
    const auto begin_call = [&](const void* object) {
        return Trace::BeginCall(Trace::Engine::Maxwell3D, object, 0xe62, code, 1, 0, parameters,
                                data_ram, {});
    };
    CHECK(begin_call(&engine) == nullptr);

    // The entry after the macro's first word holds its other two, read once the host GPU copied it.
    Trace::BeginEntry(Trace::Path::Waited, 1000, Trace::Engine::Maxwell3D, &engine, 0xe62, 2, 2);
    Trace::Call* const call = begin_call(&engine);
    REQUIRE(call != nullptr);
    CHECK(call->capture);
    Trace::NoteRead(call, 0x35e, 5);
    Trace::NoteWrite(call, 0x5f8, 8);
    Trace::EndCall(*call, false);
    Trace::EndEntry(nullptr, 0);
    const std::string dump = ReadDump(directory);
    CHECK(dump.find("3D macro") == 0);
    CHECK(dump.find("call 1: method 0xe62, 3 parameters 00000007 00000008 00000009, GPU-written: "
                    "1,2\n") != std::string::npos);
    CHECK(dump.find("registers read: 0x35e=0x5") != std::string::npos);
    CHECK(dump.find("methods written: 0x5f8=0x8") != std::string::npos);
    CHECK(begin_call(&engine) == nullptr);

    // A call begun inside an entry that ended with two of its words due: it takes the entry when
    // it runs later, on its engine only.
    Trace::BeginEntry(Trace::Path::Copied, 0, Trace::Engine::Maxwell3D, &engine, 0, 0, 4);
    Trace::EndEntry(&engine, 2);
    CHECK(begin_call(&other_engine) == nullptr);
    Trace::Call* const carried = begin_call(&engine);
    REQUIRE(carried != nullptr);
    Trace::EndCall(*carried, true);
    CHECK(begin_call(&engine) == nullptr);

    // Only the middle word from a GPU-written entry: due 2 at its start, 1 at its end.
    Trace::BeginEntry(Trace::Path::Copied, 0, Trace::Engine::Maxwell3D, &engine, 0xe62, 2, 1);
    Trace::EndEntry(&engine, 1);
    Trace::Call* const middle = begin_call(&engine);
    REQUIRE(middle != nullptr);
    Trace::EndCall(*middle, false);
    const std::string samples = ReadDump(directory);
    CHECK(samples.find("call 2: method 0xe62, 3 parameters 00000007 00000008 00000009, "
                       "GPU-written: 0\n") != std::string::npos);
    CHECK(samples.find("call 3: method 0xe62, 3 parameters 00000007 00000008 00000009, "
                       "GPU-written: 1\n") != std::string::npos);
    std::filesystem::remove_all(directory);
}

TEST_CASE("ListMacroTrace: what reads the data RAM words a GPU-written entry's call stored",
          "[video_core]") {
    namespace Trace = VideoCommon::ListMacroTrace;
    const auto directory = std::filesystem::temp_directory_path() / "lichii-list-macro-reader";
    SetDebugDir(directory);
    // Compute macro 055743923e135e09 (method 0xe74): data RAM 0x20-0x22 = its three parameters.
    const std::vector<u32> store{0x00000087, 0xb9c7c008, 0x399c7f01,
                                 0x00000003, 0x18c00008, 0xb99c7f00};
    // An inline upload of data RAM 0x20-0x22 to the address its two parameters give.
    const std::vector<u32> reader{0x00740003, 0x18c00018, 0xb18c0000, 0x00740003, 0x18c00018,
                                  0xf18c0000, 0x00d40107, 0x18c0001b, 0x318c0000, 0x003801b7,
                                  0x18c00008, 0x319c7800, 0x003801b7, 0x18c00008, 0x719c7800,
                                  0x003801b7, 0x18c00008, 0xb19c7801, 0x00000003, 0x18c00000,
                                  0x318c0000};
    Tegra::TuringMacroInterpreter interpreter;
    std::array<u32, 0x1000> registers{};
    int engine{};
    const auto run = [&](const std::vector<u32>& code, u32 method, std::vector<u32> parameters) {
        Trace::Call* const call =
            Trace::BeginCall(Trace::Engine::Compute, &engine, method, code, 1, 0, parameters,
                             interpreter.dram, registers);
        if (call == nullptr) {
            return false;
        }
        CHECK(call->data_ram);
        const auto result = interpreter.Execute(
            code, 0, parameters,
            [&](u32 address) {
                Trace::NoteRead(call, address, registers[address]);
                return registers[address];
            },
            [&](u32 address, u32 value) {
                Trace::NoteWrite(call, address, value);
                registers[address] = value;
                return true;
            });
        REQUIRE(result.success);
        Trace::EndCall(*call, false, interpreter.data_ram_reads, interpreter.data_ram_writes);
        return true;
    };
    CHECK_FALSE(run(reader, 0xe7a, {0x12, 0x34560000}));

    const auto stored_run = [&](std::vector<u32> parameters) {
        Trace::BeginEntry(Trace::Path::Waited, 1000, Trace::Engine::Compute, &engine, 0xe74, 3, 3);
        CHECK(run(store, 0xe74, std::move(parameters)));
        Trace::EndEntry(nullptr, 0);
    };
    // One 12-byte line.
    registers[0x60] = 12;
    registers[0x61] = 1;
    stored_run({8, 1, 1});
    CHECK(run(reader, 0xe7a, {0x12, 0x34560000}));
    // The back loads a QMD whose grid is the three words.
    Trace::NoteQmd(0x12'3456'0000 - 12 * 4, std::array<u32, 64>{});
    Trace::BeginEntry(Trace::Path::Copied, 0, Trace::Engine::Compute, &engine, 0, 0, 1);
    Trace::EndEntry(nullptr, 0);
    std::string dumps;
    for (const auto& file : std::filesystem::directory_iterator{directory}) {
        std::ifstream stream{file.path()};
        std::stringstream text;
        text << stream.rdbuf();
        dumps += text.str();
    }
    CHECK(dumps.find("compute macro 055743923e135e09 at 0x0 (method 0xe74") != std::string::npos);
    CHECK(dumps.find("reader call 1: method 0xe7a, 2 parameters 00000012 34560000\n  data RAM "
                     "read: 0x20=0x8 0x21=0x1 0x22=0x1\n") != std::string::npos);
    for (const auto& [word, write, address] :
         {std::tuple{0x20, 3, "0x1234560000"}, std::tuple{0x21, 4, "0x1234560004"},
          std::tuple{0x22, 5, "0x1234560008"}}) {
        CHECK(dumps.find(fmt::format("{:#x} (stored by compute 055743923e135e09, method 0xe74) "
                                     "-> write {} to 0x6d, at {}",
                                     word, write, address)) != std::string::npos);
    }
    CHECK(dumps.find("landing: data RAM 0x20 -> QMD word 12 (grid x)\n") != std::string::npos);
    CHECK(dumps.find("landing: data RAM 0x21 -> QMD word 13 (grid y)\n") != std::string::npos);
    CHECK(dumps.find("landing: data RAM 0x22 -> QMD word 14 (grid z)\n") != std::string::npos);

    // Three 4-byte lines 0x100 apart.
    registers[0x60] = 4;
    registers[0x61] = 3;
    registers[0x64] = 0x100;
    stored_run({8, 2, 3});
    CHECK(run(reader, 0xe7a, {0x12, 0x34560000}));
    std::string pitched;
    for (const auto& file : std::filesystem::directory_iterator{directory}) {
        std::ifstream stream{file.path()};
        std::stringstream text;
        text << stream.rdbuf();
        pitched += text.str();
    }
    CHECK(pitched.find("0x22 (stored by compute 055743923e135e09, method 0xe74) -> write 5 to "
                       "0x6d, at 0x1234560200") != std::string::npos);

    // A call no entry fed stores over the words: nothing to follow any more.
    CHECK(run(store, 0xe74, {0, 0, 0}));
    CHECK_FALSE(run(reader, 0xe7a, {0x12, 0x34560000}));
    std::filesystem::remove_all(directory);
}
