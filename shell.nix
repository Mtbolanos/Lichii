#!/usr/bin/nix-shell
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

let
  # nixos-26.05; bump the rev and sha256 together (nix-prefetch-url --unpack <url>)
  nixpkgs = fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/ac588ef1c914865dad517b337ea0eb9694a4021c.tar.gz";
    sha256 = "1sxd2sqlpy1bx6b5xiwy1pbyygazl9d1a5yy0rvqfsihqpcibq2y";
  };
  pkgs = import nixpkgs { config = {}; overlays = []; };
in
pkgs.mkShellNoCC {
  packages = with pkgs; [
    # essential programs
    git cmake clang gnumake ninja patch jq pkg-config
    # libraries
    openssl boost fmt_11 nlohmann_json lz4 zlib zstd
    enet libopus vulkan-headers vulkan-utility-libraries
    spirv-tools spirv-headers vulkan-loader mesa unzip
    glslang python3 httplib cpp-jwt ffmpeg-headless
    libusb1 cubeb
    # eden
    qt6.qtbase qt6.qtmultimedia qt6.qtwayland qt6.qttools qt6.qtcharts
    qt6.qtwebengine qt6.qt5compat
    # eden-cli
    sdl3
    # optional components
    discord-rpc gamemode
  ];
  shellHook = ''
    # Host /usr Vulkan drivers can't load inside the Nix shell (their libdrm/libz deps
    # aren't visible to the Nix dynamic linker), so default the loader to every Mesa ICD.
    if [ -z "$VK_DRIVER_FILES" ]; then
      export VK_DRIVER_FILES="$(echo ${pkgs.mesa}/share/vulkan/icd.d/*.json | tr ' ' ':')"
    fi
    export CMAKE_PREFIX_PATH="$(echo $nativeBuildInputs $buildInputs | tr ' ' ':')"
  '';
}
