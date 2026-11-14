cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(compatibility_report_tests LANGUAGES CXX)
    enable_testing()
endif()

set(CMAKE_AUTOMOC ON)
find_package(Qt6 REQUIRED COMPONENTS Widgets Network Test)
add_executable(compatibility_report_tests
    compatibility_report.cpp
    ../../lichii/compatibility_report.cpp
    ../../lichii/compatibility_report.h
)
target_compile_features(compatibility_report_tests PRIVATE cxx_std_20)
target_compile_definitions(compatibility_report_tests PRIVATE
    QT_USE_QSTRINGBUILDER
    QT_NO_CAST_FROM_ASCII
    QT_NO_CAST_TO_ASCII
    QT_NO_NARROWING_CONVERSIONS_IN_CONNECT
    QT_NO_PROCESS_COMBINED_ARGUMENT_START
    QT_NO_URL_CAST_FROM_STRING
)
target_include_directories(compatibility_report_tests PRIVATE ../..)
target_link_libraries(compatibility_report_tests PRIVATE Qt6::Widgets Qt6::Network Qt6::Test)
add_test(NAME compatibility_report COMMAND compatibility_report_tests)
set_tests_properties(compatibility_report PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
