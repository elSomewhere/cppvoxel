#!/bin/bash
# Build script for WebGPU Triangle project

# Exit on error
set -e

# Ensure Emscripten SDK is available
if ! command -v emcmake &> /dev/null; then
    echo "Error: Emscripten SDK not found in PATH"
    echo "Please source the emsdk_env.sh file from your Emscripten installation"
    exit 1
fi

echo "===== Building WebGPU Triangle Project ====="

# Create build directory if it doesn't exist
mkdir -p build

# Navigate to build directory
cd build

# Configure project with CMake via Emscripten
echo "Configuring project with Emscripten..."
emcmake cmake ..

# Build the project
echo "Building project..."
cmake --build . -j

echo "===== Build Complete ====="
echo "The build output is located in the build/web directory"
echo "To run the demo, navigate to the build/web directory and run:"
echo "   python3 server.py"
echo "Then open http://localhost:8080 in your browser" 