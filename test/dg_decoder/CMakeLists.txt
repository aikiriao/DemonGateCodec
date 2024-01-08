cmake_minimum_required(VERSION 3.15)

set(PROJECT_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# テスト名
set(TEST_NAME dg_decoder_test)

# 実行形式ファイル
add_executable(${TEST_NAME}
    dg_decoder_test.cpp
    dg_huffman_test.cpp
    dg_maindata_buffer_test.cpp
    main.cpp
    )

# インクルードディレクトリ
include_directories(${PROJECT_ROOT_PATH}/libs/dg_decoder/include)

# リンクするライブラリ
target_link_libraries(${TEST_NAME} gtest gtest_main bit_stream wav)
if (NOT MSVC)
target_link_libraries(${TEST_NAME} pthread)
endif()

# コンパイルオプション
# MSVCでデバッグ実行パスをtmp以下に
set_target_properties(${TEST_NAME}
    PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
    VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:${TEST_NAME}>/tmp"
    )

# 実行パスをtmp以下に
add_test(
    NAME dg_decoder
    WORKING_DIRECTORY $<TARGET_FILE_DIR:${TEST_NAME}>/tmp
    COMMAND $<TARGET_FILE:${TEST_NAME}>
    )

# run with: ctest -L lib
set_property(
    TEST dg_decoder
    PROPERTY LABELS lib dg_decoder
    )

# ビルド後にテストリソースを持ってくる
file(GLOB TEST_WAVE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/testdata/*.mpg
    ${CMAKE_CURRENT_SOURCE_DIR}/testdata/*.mp3
    ${CMAKE_CURRENT_SOURCE_DIR}/testdata/*.wav
    )
add_custom_command(
    TARGET ${TEST_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TEST_NAME}>/tmp
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_WAVE_FILES} $<TARGET_FILE_DIR:${TEST_NAME}>/tmp
    )
