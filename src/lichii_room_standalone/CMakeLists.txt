# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

add_executable(lichii_room_standalone lichii_room_standalone.cpp)

set_target_properties(lichii_room_standalone PROPERTIES OUTPUT_NAME "eden-room")

target_link_libraries(lichii_room_standalone PRIVATE lichii-room)

if(UNIX AND NOT APPLE)
    install(TARGETS lichii_room_standalone RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
endif()

if (LICHII_STATIC_ROOM)
    target_link_options(lichii_room_standalone PRIVATE "-static")
endif()

create_target_directory_groups(lichii_room_standalone)
