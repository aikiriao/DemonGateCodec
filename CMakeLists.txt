cmake_minimum_required(VERSION 3.15)

# デコーダライブラリ
project(DemonGateDecoderLibrary C)
set(DECODER_LIB_NAME dgdec)
add_library(${DECODER_LIB_NAME}
    STATIC
    $<TARGET_OBJECTS:dg_decoder>
    $<TARGET_OBJECTS:bit_stream>
    )

# 依存するプロジェクト
add_subdirectory(libs)

# テスト
if(NOT without-test)
    enable_testing()
    project(DemonGateCodecTest CXX)
    if(MSVC)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    else()
        set(CMAKE_CXX_FLAGS "-std=gnu++11") # gtestがGNU独自拡張を使用しているため
        set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -DDEBUG")
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    endif()
    include(cmake/gtest.cmake)
    add_subdirectory(test)
endif()
