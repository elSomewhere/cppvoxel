#!/bin/bash
# WebGPU Application Build and Run Script
# Usage: ./build-and-run.sh [PORT]
# If PORT is not specified, defaults to 8080

# Default port
PORT=${1:-8080}

# Project name
PROJECT_NAME="WebGPU Triangle"

# Set colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored status messages
print_status() {
    echo -e "${BLUE}===> $1${NC}"
}

print_success() {
    echo -e "${GREEN}===> $1${NC}"
}

print_error() {
    echo -e "${RED}===> ERROR: $1${NC}"
}

# Check if emscripten is available
if ! command -v emcc &> /dev/null; then
    print_error "Emscripten compiler (emcc) not found. Please make sure Emscripten is installed and activated."
    exit 1
fi

print_status "Building $PROJECT_NAME..."

# Create build directory if it doesn't exist
BUILD_DIR="build/web"
mkdir -p $BUILD_DIR

# Clean up unnecessary files
print_status "Cleaning up unnecessary files..."
rm -rf build/web/index.html.bak 2>/dev/null
rm -rf emscripten-build 2>/dev/null
# Keep only essential files
find . -type f -name "*.o" -delete 2>/dev/null
find . -type f -name "*.tmp" -delete 2>/dev/null

# Kill any running server processes to prevent port conflicts
print_status "Checking for existing server processes..."
pkill -f "python.*server.py" || true

# Compile the application with WebGPU support
print_status "Compiling project with emcc..."
emcc -std=c++17 -O1 main.cpp \
     -o $BUILD_DIR/voxel_engine.js \
     -s USE_WEBGPU=1 \
     -s WASM=1 \
     -s ALLOW_MEMORY_GROWTH=1 \
     -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap'] \
     -s "EXPORTED_FUNCTIONS=['_main','_malloc','_free']" \
     -s ASYNCIFY \
     -s "ASYNCIFY_IMPORTS=['OnDeviceCreated','OnAdapterReady']" \
     --shell-file index.html

if [ $? -ne 0 ]; then
    print_error "Compilation failed!"
    exit 1
fi

print_success "Compilation successful!"

# Create a simple server script to serve the WebGPU application
print_status "Creating server script with port $PORT..."
cat > $BUILD_DIR/server.py << EOL
#!/usr/bin/env python3
import http.server
import socketserver
import os
import sys
from pathlib import Path

PORT = $PORT
DIRECTORY = Path(__file__).parent.absolute()

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)
        
    def end_headers(self):
        # Add CORS headers for WebGPU to work
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        super().end_headers()

def run_server():
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"Starting server at http://localhost:{PORT}")
            print(f"Server running from directory: {DIRECTORY}")
            print("Press Ctrl+C to stop")
            httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"Error: Port {PORT} is already in use.")
            print("Please stop any running servers and try again.")
            sys.exit(1)
        else:
            raise

if __name__ == "__main__":
    run_server()
EOL

# Make the server script executable
chmod +x $BUILD_DIR/server.py

print_success "Build complete! The output is in the $BUILD_DIR directory."
print_status "Starting server on port $PORT..."

# Start the server
cd $BUILD_DIR
python3 server.py 