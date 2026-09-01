cmake_minimum_required(VERSION 3.24)

project(triposplat_cpp
  VERSION 0.1.0
  LANGUAGES CXX
  DESCRIPTION "C++ Stage-A inference library for TripoSplat (single image -> 3D Gaussians)")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(TRIPOSPLAT_BUILD_SHARED "Build triposplat as a shared library (otherwise static)" OFF)
option(TRIPOSPLAT_BUILD_EXAMPLES "Build the example executables" ON)
option(TRIPOSPLAT_BUILD_TESTS "Build the C++ test executables" OFF)

# Require C++17 (libtorch + std::filesystem).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# MSVC: enable a reasonable warning level and UTF-8 source execution charset.
if(MSVC)
  add_compile_options(/W3 /permissive- /utf-8 /EHsc)
  # Avoid the min/max macro pollution from windows.h.
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
# libtorch: pass -DCMAKE_PREFIX_PATH=<libtorch-root> (the dir containing
# share/cmake/Torch). CUDA is pulled in transitively by Torch_CMAKE.
find_package(Torch REQUIRED)

# stb_image (single-header image loader) and nlohmann/json are fetched on first
# configure so the repo stays self-contained.
include(FetchContent)

FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG        f7f20f39fe4f206c6f19e26ebae1aa45ef89ddfd
  GIT_SHALLOW    TRUE
)
# stb has no CMake target; we just need the headers on the include path.
FetchContent_MakeAvailable(stb)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
set(TRIPOSPLAT_SOURCES
  src/Preprocess.cpp
  src/Encoders.cpp
  src/FlowSampler.cpp
  src/Decoder.cpp
  src/GaussianIO.cpp
  src/Pipeline.cpp
)

if(TRIPOSPLAT_BUILD_SHARED)
  add_library(triposplat SHARED ${TRIPOSPLAT_SOURCES})
  target_compile_definitions(triposplat PRIVATE TRIPOSPLAT_BUILD_DLL TRIPOSPLAT_EXPORTS)
else()
  add_library(triposplat STATIC ${TRIPOSPLAT_SOURCES})
endif()

target_include_directories(triposplat
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${stb_SOURCE_DIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(triposplat
  PUBLIC
    ${TORCH_LIBRARIES}
  PRIVATE
    nlohmann_json::nlohmann_json
)

# stb_image implementations are compiled in exactly one TU.
target_sources(triposplat PRIVATE src/StbImage.cpp)
target_compile_definitions(triposplat PRIVATE STB_IMAGE_IMPLEMENTATION)

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
if(TRIPOSPLAT_BUILD_EXAMPLES)
  add_executable(run_example examples/run_example.cpp)
  target_link_libraries(run_example PRIVATE triposplat)
  if(MSVC)
    set_target_properties(run_example PROPERTIES
      VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endif()
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(TRIPOSPLAT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS triposplat
  EXPORT triposplatTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

message(STATUS "triposplat_cpp configured:")
message(STATUS "  shared  : ${TRIPOSPLAT_BUILD_SHARED}")
message(STATUS "  examples: ${TRIPOSPLAT_BUILD_EXAMPLES}")
message(STATUS "  tests   : ${TRIPOSPLAT_BUILD_TESTS}")
message(STATUS "  torch   : ${TORCH_INSTALL_PREFIX}")
