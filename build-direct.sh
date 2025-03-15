#!/bin/bash
# Direct build script for Emscripten project - fixed to avoid recursion issues

# Exit on error
set -e

# Ensure Emscripten SDK is available
if ! command -v emcc &> /dev/null; then
    echo "Error: Emscripten SDK not found in PATH"
    echo "Please source the emsdk_env.sh file from your Emscripten installation"
    exit 1
fi

echo "===== Building Minimal Emscripten Project ====="

# Create build directory if it doesn't exist
mkdir -p build/web

# Check for any existing server processes and kill them
echo "Checking for existing server processes..."
pkill -f "python.*server.py" || true

# Compile the project directly with emcc
echo "Compiling project with emcc..."
emcc main.cpp \
     -o build/web/voxel_engine.js \
     -O1 \
     -s USE_WEBGPU=1 \
     -s WASM=1 \
     -s ALLOW_MEMORY_GROWTH=1 \
     -s ASSERTIONS=1 \
     -s STACK_OVERFLOW_CHECK=2 \
     -s DEMANGLE_SUPPORT=1 \
     -s EXPORTED_RUNTIME_METHODS=ccall,cwrap \
     -s EXPORTED_FUNCTIONS=_main,_malloc,_free \
     --no-entry

# Copy the server script to build directory
echo "Creating server script..."
cat > build/web/server.py << 'EOL'
#!/usr/bin/env python3
"""
Simple HTTP server for WebGPU applications
"""

import http.server
import socketserver
import os
import sys

PORT = 8080

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add headers needed for WebGPU to work correctly
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        http.server.SimpleHTTPRequestHandler.end_headers(self)

if __name__ == "__main__":
    # Get the directory of this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print(f"Starting server at http://localhost:{PORT}")
    print(f"Server running from directory: {os.getcwd()}")
    print("Press Ctrl+C to stop")
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"Error: Port {PORT} is already in use.")
            print("Please stop any running servers and try again.")
            sys.exit(1)
        else:
            raise
    except KeyboardInterrupt:
        print("\nServer stopped")
EOL

# Make server script executable
chmod +x build/web/server.py

echo "===== Build Complete ====="
echo "The build output is located in the build/web directory"
echo "To run the demo, navigate to the build/web directory and run:"
echo "   python3 server.py"
echo "Then open http://localhost:8080 in your browser"

# Automatically start the server
echo "Starting server..."
cd build/web && python3 server.py & 
echo "Server starting in background. Open http://localhost:8080 in your browser" 