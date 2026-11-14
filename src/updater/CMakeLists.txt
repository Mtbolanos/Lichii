# SPDX-FileCopyrightText: Copyright 2026 Lichii Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

add_executable(lichii-updater main.cpp updater.rc ../frontend_common/update_installer.cpp)
target_include_directories(lichii-updater PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_features(lichii-updater PRIVATE cxx_std_20)
target_link_libraries(lichii-updater PRIVATE user32 shell32)
set_target_properties(lichii-updater PROPERTIES WIN32_EXECUTABLE ON)

# The helper runs from a temporary folder, away from the DLLs next to lichii.exe.
if (MSVC)
    set_property(TARGET lichii-updater PROPERTY MSVC_RUNTIME_LIBRARY
                 "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    target_link_options(lichii-updater PRIVATE -static)
endif()

# The PGO profile is trained on lichii.exe and does nothing for the updater, and Defender flagged
# Eden's PGO-built updater as a trojan (eden-emu/eden#4706, fix from eden-emu/eden#4707).
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(lichii-updater PRIVATE -fno-profile-instr-use)
endif()
