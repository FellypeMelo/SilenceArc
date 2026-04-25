cmake_minimum_required(VERSION 3.20)
project(SilenceArc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    add_definitions(-DUNICODE -D_UNICODE)
endif()

# JSON Library
include(FetchContent)
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

add_subdirectory(third_party/imgui)

include_directories(include)
include_directories(tests) 
include_directories(tests/utils)
include_directories(third_party/miniaudio)

# GoogleTest (Optional)
set(USE_GTEST ON)
if(USE_GTEST)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/heads/main.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    
    # Correctly add GTest includes after FetchContent
    include_directories(${googletest_SOURCE_DIR}/googletest/include)
    include_directories(${googletest_SOURCE_DIR}/googlemock/include)
endif()

# ONNX Runtime DirectML Configuration
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/build/native/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/runtimes/win-x64/native")

include_directories(${ONNXRUNTIME_INCLUDE_DIR})
link_directories(${ONNXRUNTIME_LIB_DIR})

# DeepFilterNet Library (External Rust DLL)
add_library(df_shared SHARED IMPORTED)
set_target_properties(df_shared PROPERTIES
    IMPORTED_IMPLIB "${CMAKE_CURRENT_SOURCE_DIR}/DeepFilterNet/target/release/df.dll.lib"
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/DeepFilterNet/target/release/df.dll"
)

# SilenceArc Library (Infrastructure)
add_library(silence_arc_infra STATIC
    src/infrastructure/ui_manager.cpp
    src/infrastructure/deep_filter_adapter.cpp
    src/infrastructure/async_audio_pipeline.cpp
    src/infrastructure/miniaudio_device_manager.cpp
    src/infrastructure/miniaudio_pipeline.cpp
    src/infrastructure/onnx_adapter.cpp
    src/infrastructure/directml_telemetry_provider.cpp
    src/infrastructure/feature_extractor.cpp
    src/domain/audio_stream_buffer.cpp
    src/infrastructure/cpu_dsp_engine.cpp
    src/infrastructure/directml_audio_engine.cpp
    
    )
    target_link_libraries(silence_arc_infra PUBLIC
    imgui
    df_shared 
    nlohmann_json::nlohmann_json
    onnxruntime
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    target_compile_options(silence_arc_infra PUBLIC /EHsc  --target=x86_64-pc-windows-msvc -Wno-ignored-attributes)
    target_link_options(silence_arc_infra PUBLIC )
    endif()
# UI Manager Test
add_executable(test_ui_manager tests/test_ui_manager.cpp)
target_link_libraries(test_ui_manager PRIVATE GTest::gtest_main silence_arc_infra)

if(USE_GTEST)
    # Noise Suppression Test (GTest)
    add_executable(test_noise_suppression tests/test_noise_suppression.cpp)
    target_link_libraries(test_noise_suppression PRIVATE GTest::gtest_main silence_arc_infra)

    # Audio Pipeline Test (GTest)
    add_executable(test_audio_pipeline tests/test_audio_pipeline.cpp)
    target_link_libraries(test_audio_pipeline PRIVATE GTest::gtest_main silence_arc_infra)

    # E2E Loopback Test (GTest)
    add_executable(test_e2e_loopback tests/test_e2e_loopback.cpp)
    target_link_libraries(test_e2e_loopback PRIVATE GTest::gtest_main silence_arc_infra)

    # Audio Stream Buffer Test (GTest)
    add_executable(test_audio_stream_buffer tests/test_audio_stream_buffer.cpp)
    target_link_libraries(test_audio_stream_buffer PRIVATE GTest::gtest_main silence_arc_infra)

    # Telemetry Provider Test
    add_executable(test_telemetry_provider tests/test_telemetry_provider.cpp)
    target_link_libraries(test_telemetry_provider PRIVATE GTest::gtest_main silence_arc_infra)

    # Neural Path Test (GTest)
    add_executable(test_neural_path tests/test_neural_path.cpp)
    target_link_libraries(test_neural_path PRIVATE GTest::gtest_main silence_arc_infra)
endif()

# Main Application
add_executable(silence_arc src/main.cpp)
target_link_libraries(silence_arc PRIVATE silence_arc_infra)

enable_testing()
add_test(NAME UIManagerTest COMMAND test_ui_manager)

if(USE_GTEST)
    add_test(NAME NoiseSuppressionTest COMMAND test_noise_suppression)
    add_test(NAME AudioPipelineTest COMMAND test_audio_pipeline)
    add_test(NAME E2ELoopbackTest COMMAND test_e2e_loopback)
    add_test(NAME AudioStreamBufferTest COMMAND test_audio_stream_buffer)
    add_test(NAME TelemetryProviderTest COMMAND test_telemetry_provider)
    add_test(NAME NeuralPathTest COMMAND test_neural_path)
endif()
