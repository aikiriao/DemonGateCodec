cmake_minimum_required(VERSION 3.15)

set(PROJECT_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# プロジェクト名
project(DemonGateDecoderCUI C)

# アプリケーション名
set(APP_NAME dg_decoder_cui)

# ライブラリのテストはしない
set(without-test 1)

# 実行形式ファイル
add_executable(${APP_NAME} dg_decoder_main.c)

# 依存するサブディレクトリを追加
add_subdirectory(${PROJECT_ROOT_PATH} ${CMAKE_CURRENT_BINARY_DIR}/libdgdec)

# インクルードパス
target_include_directories(${APP_NAME}
    PRIVATE
    ${PROJECT_ROOT_PATH}/libs/dg_decoder/include
    )

# リンクするライブラリ
target_link_libraries(${APP_NAME} wav dgdec)
if (UNIX AND NOT APPLE)
    target_link_libraries(${APP_NAME} m)
endif()

# コンパイルオプション
if(MSVC)
    target_compile_options(${APP_NAME} PRIVATE /W4)
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /D DEBUG")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /D NDEBUG")
else()
    target_compile_options(${APP_NAME} PRIVATE -Wall -Wextra -Wpedantic -Wformat=2 -Wstrict-aliasing=2 -Wconversion -Wmissing-prototypes -Wstrict-prototypes -Wold-style-definition)
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g3 -DDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()
set_target_properties(${APP_NAME}
    PROPERTIES
    C_STANDARD 90 C_EXTENSIONS OFF
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
    )
