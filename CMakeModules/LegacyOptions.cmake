# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# The build options were renamed from YUZU_<name> to LICHII_<name> (docs/nx2/rename.md).
# A configure that still passes or caches the old name carries its value to the new one,
# so existing build directories and scripts keep their settings. A configure cannot tell an
# explicit -DLICHII_<name> from its cache entry, so the old name keeps deciding while
# LICHII_<name> still holds the value last carried from it, and warns when that replaces a
# different value. A LICHII_<name> set to anything else wins and the old name is ignored.
set(_lichii_renamed_options
    BUILD_PRESET CMD CRASH_DUMPS DISABLE_LLVM DOWNLOAD_ANDROID_VVL DOWNLOAD_TIME_ZONE_DATA
    INSTALL_UDEV_RULES LEGACY QT_LANGUAGES QT_MIRROR ROOM ROOM_STANDALONE STATIC_BUILD STATIC_ROOM
    TESTS TZDB_PATH USE_BUNDLED_FFMPEG USE_BUNDLED_MOLTENVK USE_BUNDLED_OPENSSL USE_BUNDLED_QT
    USE_BUNDLED_SDL3 USE_BUNDLED_SIRIT USE_EXTERNAL_FFMPEG USE_QT_MULTIMEDIA USE_QT_WEB_ENGINE)

foreach (_name IN LISTS _lichii_renamed_options)
    if (NOT DEFINED CACHE{YUZU_${_name}})
        continue()
    endif()
    set(_old "${YUZU_${_name}}")
    # The new name was carried from the old one before (or never set), so the old name still decides.
    if (NOT DEFINED LICHII_${_name} OR "${LICHII_${_name}}" STREQUAL "${_LICHII_CARRIED_${_name}}")
        if (DEFINED LICHII_${_name} AND NOT "${LICHII_${_name}}" STREQUAL "${_old}")
            # A configure cannot tell a -DLICHII_<name> repeating the carried value from the cached carry.
            message(WARNING "YUZU_${_name}=${_old} replaces LICHII_${_name}=${LICHII_${_name}}; pass only LICHII_${_name}")
        endif()
        message(DEPRECATION "YUZU_${_name} is now LICHII_${_name}; using YUZU_${_name}=${_old}")
        set(LICHII_${_name} "${_old}" CACHE STRING "" FORCE)
        set(_LICHII_CARRIED_${_name} "${_old}" CACHE INTERNAL "")
    else()
        message(WARNING "YUZU_${_name}=${_old} is ignored: LICHII_${_name}=${LICHII_${_name}} is set")
    endif()
    unset(YUZU_${_name} CACHE)
endforeach()
