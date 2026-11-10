# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# nxformat (https://github.com/fab918/nxformat) reads and writes .nx game
# archives. The core uses its header-only C++ reader, vendored in
# externals/nxformat. The Qt frontend's NX converter calls its C library to
# convert games to .nx and to edit them. The pinned release of that library is
# downloaded and checked here; LICHII_NXFORMAT_ROOT may instead point at an
# unpacked release, such as a local build packed by nxformat's
# tools/package_release.py. When neither is available, the frontend is built
# without the converter.
#
# Defines nxformat::nxformat and LICHII_NXFORMAT_RUNTIME, the shared library to
# ship next to the executable, when the converter is built.
set(LICHII_NXFORMAT_ROOT "" CACHE PATH "Unpacked nxformat release (empty: download the pinned one)")
set(LICHII_NXFORMAT_RUNTIME "" CACHE INTERNAL "")

function(lichii_find_nxformat)
    set(version 1.0.1)
    set(tag V1.0.1)
    set(root "${LICHII_NXFORMAT_ROOT}")
    if (NOT root)
        if (WIN32 AND ARCHITECTURE_x86_64)
            set(platform windows-x64)
            set(sha256 b49ca68fa313dcb977e542514314040334cabab92912394a84038b9eb1497e76)
        elseif (APPLE)
            set(platform macos-universal)
            set(sha256 3cedc71db420cb45ee4e385f54fee371cc02b04a70484bae44acea15d126ee0d)
        elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND ARCHITECTURE_x86_64)
            set(platform linux-x64)
            set(sha256 b1c37992f8d01c8425d5946b8b125694409142f2b89731370006e5b54b876a98)
        elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND ARCHITECTURE_arm64)
            set(platform linux-arm64)
            set(sha256 f511a9ac6b45cf430e8ed5d27e080c901c4c5be1eb53b4ad862177cc53845e11)
        else()
            message(WARNING "NX converter disabled: nxformat has no prebuilt library for this platform. "
                            "Set LICHII_NXFORMAT_ROOT to an unpacked nxformat release to build it. "
                            "Loading .nx games is not affected.")
            return()
        endif()
        if (WIN32)
            set(ext zip)
        else()
            set(ext tar.gz)
        endif()
        set(name "nxformat-${version}-${platform}")
        set(root "${CMAKE_BINARY_DIR}/externals/${name}")
        if (NOT EXISTS "${root}/lib/cmake/nxformat/nxformatConfig.cmake")
            set(url "https://github.com/fab918/nxformat/releases/download/${tag}/${name}.${ext}")
            set(archive "${CMAKE_BINARY_DIR}/externals/${name}.${ext}")
            # The hash is checked here rather than by file(DOWNLOAD), which reports a failed
            # download as a configure error.
            file(DOWNLOAD "${url}" "${archive}" TLS_VERIFY ON STATUS status)
            list(GET status 0 code)
            if (code EQUAL 0)
                file(SHA256 "${archive}" actual)
                if (NOT actual STREQUAL sha256)
                    set(code 1)
                    set(status "${code};SHA-256 ${actual} does not match the pinned ${sha256}")
                endif()
            endif()
            if (NOT code EQUAL 0)
                list(GET status 1 reason)
                file(REMOVE "${archive}")
                message(WARNING "NX converter disabled: could not download ${url} (${reason}). "
                                "Reconfigure with network access, or set LICHII_NXFORMAT_ROOT to an "
                                "unpacked nxformat release, to build it. Loading .nx games is not affected.")
                return()
            endif()
            file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${root}")
        endif()
    endif()

    find_package(nxformat ${version} REQUIRED CONFIG PATHS "${root}/lib/cmake/nxformat" NO_DEFAULT_PATH)
    set_target_properties(nxformat::nxformat PROPERTIES IMPORTED_GLOBAL TRUE)
    set(LICHII_NXFORMAT_RUNTIME "${NXFORMAT_RUNTIME}" CACHE INTERNAL "")
    message(STATUS "nxformat ${nxformat_VERSION}: ${root}")
endfunction()

if (ENABLE_QT)
    lichii_find_nxformat()
endif()
