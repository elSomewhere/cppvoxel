cmake_minimum_required(VERSION 3.10)
project(WebGPUTriangle)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output files to the web directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/web)

# Include Emscripten headers
include_directories(SYSTEM "/Users/estebanlanter/CLionProjects/emsdk/upstream/emscripten/cache/sysroot/include")

# Add the main executable
add_executable(voxel_engine main.cpp)

# Emscripten-specific compiler and linker flags
target_link_options(voxel_engine PRIVATE
    "SHELL:-sUSE_WEBGPU=1"
    "SHELL:-sWASM=1"
    "SHELL:-sALLOW_MEMORY_GROWTH=1"
    "SHELL:-sASSERTIONS=1"
    "SHELL:-sDISABLE_EXCEPTION_CATCHING=0"
    "SHELL:-sERROR_ON_UNDEFINED_SYMBOLS=1"
    "SHELL:-sEXPORT_NAME=createModule"
    "SHELL:-sMODULARIZE=1"
    "SHELL:--no-entry"
)
