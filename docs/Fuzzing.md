# Fuzzing the game-file parsers

`-DLICHII_FUZZERS=ON` builds three libFuzzer harnesses in `src/fuzz/` and compiles the whole build with AddressSanitizer and UndefinedBehaviorSanitizer. It needs clang with its compiler-rt runtimes (`libclang-rt-<version>-dev` on Ubuntu). Use a separate build directory, since every object is instrumented.

| Harness | Input |
|---|---|
| `fuzz_partition_romfs` | Raw PFS0/HFS0 partition, then the same bytes as a raw RomFS image |
| `fuzz_nca` | One NCA, on its own, as an update NCA with a missing base, and inside an NSP |
| `fuzz_package` | A whole NSP, then the same bytes as an XCI |

NCA headers are normally encrypted. The NCA reader also accepts a plaintext header (magic `NCA3` at 0x200) and sections with encryption type None, so fuzz NCA inputs in that form. `fuzz_nca` and `fuzz_package` point the key manager at a scratch keys directory with made-up key area keys, so plaintext NCAs pass the key check. They never read or write your own keys.

```sh
cmake -S . -B build-fuzz -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DLICHII_FUZZERS=ON -DENABLE_WERROR=OFF -DENABLE_QT=OFF -DLICHII_CMD=OFF -DLICHII_ROOM=OFF
cmake --build build-fuzz --target fuzz_partition_romfs fuzz_nca fuzz_package
mkdir -p corpus/nca && ASAN_OPTIONS=detect_leaks=0 \
    build-fuzz/bin/fuzz_nca -rss_limit_mb=3072 -malloc_limit_mb=3072 -timeout=25 corpus/nca seeds/
```

Seed the corpus with small files you made yourself, never with game dumps. A crash writes `crash-<sha1>` to the working directory. To replay it, pass the file to the harness: `build-fuzz/bin/fuzz_nca crash-<sha1>`. Turn each fixed crash into a case in `src/tests/core/file_sys_parsers.cpp` that builds the input from scratch.

With `-DLICHII_TESTS=ON`, the same build directory also builds `tests` under the sanitizers.
