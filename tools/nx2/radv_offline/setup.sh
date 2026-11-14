#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Builds Mesa RADV with the amdgpu drm-shim, Fossilize and the radv_pipe host program for
# offline AMD shader compilation on an x86_64 or aarch64 Ubuntu 24.04 machine without an AMD GPU.
# Idempotent: rerunning skips finished steps. See README.md.
set -euo pipefail

MESA_VERSION=${MESA_VERSION:-26.3.0}
FOSSILIZE_COMMIT=${FOSSILIZE_COMMIT:-556da9ff7d7616704dda515756219ab1dab693ec}
PREFIX=${RADV_OFFLINE_HOME:-$HOME/.cache/lichii-radv-offline}
JOBS=${JOBS:-$(nproc)}
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$PREFIX/src"

if [ "${SKIP_APT:-0}" != 1 ]; then
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
        build-essential cmake ninja-build pkg-config git curl ca-certificates xz-utils \
        python3 python3-venv python3-mako python3-yaml python3-ply bison flex \
        libexpat1-dev libpciaccess-dev zlib1g-dev libzstd-dev glslang-tools \
        spirv-tools libvulkan1 llvm-20-dev >/dev/null
fi

if [ ! -x "$PREFIX/venv/bin/meson" ]; then
    python3 -m venv "$PREFIX/venv"
    "$PREFIX/venv/bin/pip" install -q 'meson==1.9.1' mako pyyaml packaging
fi

MESA_SRC="$PREFIX/src/mesa-$MESA_VERSION"
if [ ! -f "$MESA_SRC/meson.build" ]; then
    curl -fsSL "https://archive.mesa3d.org/mesa-$MESA_VERSION.tar.xz" -o "$PREFIX/src/mesa.tar.xz"
    tar -xf "$PREFIX/src/mesa.tar.xz" -C "$PREFIX/src"
    rm "$PREFIX/src/mesa.tar.xz"
fi

# Mesa 26.3 has no RADV_FORCE_FAMILY null device; the amdgpu drm-shim (LD_PRELOAD) replaces it.
# RADV still compiles with ACO; LLVM is only there to disassemble the ISA.
MESA_STAMP="$MESA_VERSION drm-shim llvm-20 $(sha256sum "$HERE/add_shim_devices.py" | cut -c1-12)"
ICD="$PREFIX/mesa/share/vulkan/icd.d/radeon_icd.$(uname -m).json"
if [ "$(cat "$PREFIX/mesa/.stamp" 2>/dev/null)" != "$MESA_STAMP" ]; then
    python3 -I "$HERE/add_shim_devices.py" "$MESA_SRC/src/amd/common/amdgpu_devices.c"
    printf "[binaries]\nllvm-config = '/usr/lib/llvm-20/bin/llvm-config'\n" > "$PREFIX/llvm.ini"
    [ -f "$PREFIX/build/mesa/build.ninja" ] && RECONFIGURE=--reconfigure || RECONFIGURE=
    PATH="$PREFIX/venv/bin:$PATH" meson setup $RECONFIGURE "$PREFIX/build/mesa" "$MESA_SRC" \
        --prefix "$PREFIX/mesa" --libdir lib --native-file "$PREFIX/llvm.ini" \
        -Dbuildtype=debugoptimized -Db_ndebug=false \
        -Dvulkan-drivers=amd -Dgallium-drivers= -Dplatforms= -Dllvm=enabled -Dshared-llvm=enabled \
        -Dopengl=false -Dgles1=disabled -Dgles2=disabled -Degl=disabled -Dglx=disabled \
        -Dgbm=disabled -Dvideo-codecs= -Dvulkan-layers= -Dtools=drm-shim -Dbuild-tests=false \
        -Dshader-cache=disabled -Dzstd=enabled -Dallow-fallback-for=libdrm --force-fallback-for=libdrm
    PATH="$PREFIX/venv/bin:$PATH" ninja -C "$PREFIX/build/mesa" -j "$JOBS" install
    echo "$MESA_STAMP" > "$PREFIX/mesa/.stamp"
fi

FOSSILIZE_SRC="$PREFIX/src/Fossilize"
if [ ! -x "$PREFIX/fossilize/bin/fossilize-replay" ] || \
   [ "$(git -C "$FOSSILIZE_SRC" rev-parse HEAD 2>/dev/null)" != "$FOSSILIZE_COMMIT" ]; then
    [ -d "$FOSSILIZE_SRC/.git" ] || git clone -q https://github.com/ValveSoftware/Fossilize.git "$FOSSILIZE_SRC"
    git -C "$FOSSILIZE_SRC" fetch -q origin "$FOSSILIZE_COMMIT"
    git -C "$FOSSILIZE_SRC" checkout -q "$FOSSILIZE_COMMIT"
    git -C "$FOSSILIZE_SRC" submodule update -q --init --recursive
    cmake -S "$FOSSILIZE_SRC" -B "$PREFIX/build/fossilize" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$PREFIX/fossilize" -DFOSSILIZE_TESTS=OFF >/dev/null
    ninja -C "$PREFIX/build/fossilize" -j "$JOBS" install >/dev/null
fi

mkdir -p "$PREFIX/bin"
c++ -std=c++20 -O1 -Wall -I "$MESA_SRC/include" "$HERE/radv_pipe.cpp" -ldl -o "$PREFIX/bin/radv_pipe.new"
mv "$PREFIX/bin/radv_pipe.new" "$PREFIX/bin/radv_pipe"

cat > "$PREFIX/env.sh" <<EOF
export RADV_OFFLINE_HOME="$PREFIX"
export VK_DRIVER_FILES="$ICD"
export VK_ICD_FILENAMES="$ICD"
export RADV_OFFLINE_LIBS="$PREFIX/mesa/lib"
export RADV_OFFLINE_SHIM="$PREFIX/mesa/lib/libamdgpu_noop_drm_shim.so"
export PATH="$PREFIX/bin:$PREFIX/fossilize/bin:\$PATH"
EOF
echo "ready: source $PREFIX/env.sh (Mesa $MESA_VERSION, Fossilize ${FOSSILIZE_COMMIT:0:10})"
