#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# Builds fsr4-vk (github.com/Astyyyyy/fsr4-vk, GPL-3.0-only), the community Vulkan port of FSR 4
# the Linux packages ship. It is built from AMD's signed FSR 4.1.1 upscaler DLL of FidelityFX SDK
# 2.3.0, which the SDK's license.md lists under MIT. The DLL, the SDK licence and the fsr4-vk commit
# are pinned.
#
# Usage: tools/fsr4-vk/build.sh <work dir> <output dir>
# The work dir holds the fsr4-vk checkout and its generated assets, and may be cached between runs.
# The output dir receives amd_fidelityfx_upscaler_vk.so and licenses/.
# Needs git, curl, cmake, a C++20 compiler, python3 with zstandard, and llvm-objdump.

set -euo pipefail

FSR4VK_REPO=https://github.com/Astyyyyy/fsr4-vk.git
FSR4VK_COMMIT=b0125ea9798633750f83ebabbc00279bcb49d8dd
# FidelityFX SDK v2.3.0.
SDK_URL=https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/60f4ea81909200d8542eca14dccb2628b763a9a3
DLL_ENTRY=Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll
DLL_SHA256=d0dcccc74a43c44ba435b7a369b456e0970d8a4464e4bd683119b374f2c9fb46
LICENSE_ENTRY=Kits/FidelityFX/docs/license.md
LICENSE_SHA256=6757f2fec461238da6f84f83536ec195e8c036b1773a06f6f34301b9ed8c0963

if [ $# -ne 2 ]; then
    echo "usage: $0 <work dir> <output dir>" >&2
    exit 2
fi
mkdir -p "$1" "$2"
work=$(cd "$1" && pwd)
out=$(cd "$2" && pwd)

if ! command -v llvm-objdump >/dev/null; then
    for dir in /usr/lib/llvm-*/bin; do
        [ -x "$dir/llvm-objdump" ] && PATH="$dir:$PATH"
    done
fi
command -v llvm-objdump >/dev/null || { echo "llvm-objdump not found" >&2; exit 1; }

fetch() {
    local entry=$1 sha256=$2 file=$3
    if [ ! -f "$file" ] || ! echo "$sha256  $file" | sha256sum --check --status; then
        curl -fsSL --retry 3 -o "$file.part" "$SDK_URL/$entry"
        echo "$sha256  $file.part" | sha256sum --check --quiet
        mv "$file.part" "$file"
    fi
}
fetch "$DLL_ENTRY" "$DLL_SHA256" "$work/amd_fidelityfx_upscaler_dx12.dll"
fetch "$LICENSE_ENTRY" "$LICENSE_SHA256" "$work/amd-fidelityfx-sdk-license.md"

src="$work/fsr4-vk"
if [ "$(git -C "$src" rev-parse HEAD 2>/dev/null)" != "$FSR4VK_COMMIT" ]; then
    rm -rf "$src"
    git init -q "$src"
    git -C "$src" fetch -q --depth 1 "$FSR4VK_REPO" "$FSR4VK_COMMIT"
    git -C "$src" -c advice.detachedHead=false checkout -q FETCH_HEAD
fi
git -C "$src" submodule update -q --init --recursive --depth 1

python3 "$src/tools/setup.py" --dll "$work/amd_fidelityfx_upscaler_dx12.dll"
FSR4_BUILD_JOBS=${FSR4_BUILD_JOBS:-$(nproc)} bash "$src/tools/build/linux.sh" -G Ninja

library="$src/build/linux/amd_fidelityfx_upscaler_vk.so"
for symbol in ffxCreateContext ffxDestroyContext ffxQuery ffxDispatch; do
    if ! nm -D --defined-only "$library" | grep -qw "$symbol"; then
        echo "$library lacks $symbol" >&2
        exit 1
    fi
done
install -m 644 "$library" "$out/"
rm -rf "$out/licenses"
mkdir -p "$out/licenses/fsr4-vk"
cp "$src/LICENSE" "$src/THIRD_PARTY_NOTICES.md" "$out/licenses/fsr4-vk/"
cp -r "$src/LICENSES" "$out/licenses/fsr4-vk/"
cp "$work/amd-fidelityfx-sdk-license.md" "$out/licenses/fsr4-vk/AMD-FidelityFX-SDK-license.md"
