#!/bin/bash
# Build script for WebGPU triangle example using Emscripten

# Check if emscripten is available
if ! command -v emcc &> /dev/null; then
    echo "Emscripten compiler (emcc) not found. Please make sure Emscripten is installed and activated."
    exit 1
fi

echo "===== Building Minimal Emscripten Project ====="

# Create build directory if it doesn't exist
mkdir -p build/web

# Kill any running server processes to prevent port conflicts
echo "Checking for existing server processes..."
pkill -f "python.*server.py" || true

# Compile the application with WebGPU support
echo "Compiling project with emcc..."
emcc -std=c++17 -O1 main.cpp \
     -o build/web/voxel_engine.js \
     -s USE_WEBGPU=1 \
     -s WASM=1 \
     -s ALLOW_MEMORY_GROWTH=1 \
     -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap'] \
     -s "EXPORTED_FUNCTIONS=['_main','_malloc','_free']" \
     -s ASYNCIFY \
     -s "ASYNCIFY_IMPORTS=['OnDeviceCreated','OnAdapterReady']" \
     --shell-file index.html

# Create a simple server script to serve the WebGPU application
cat > build/web/server.py << 'EOL'
#!/usr/bin/env python3
import http.server
import socketserver
import os
import sys
from pathlib import Path

PORT = 8080
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
chmod +x build/web/server.py

echo "===== Build Complete ====="
echo "The build output is located in the build/web directory"
echo "To run the demo, navigate to the build/web directory and run:"
echo "   python3 server.py"
echo "Then open http://localhost:8080 in your browser"

# Start the server automatically
echo "Starting server..."
cd build/web && python3 server.py &
echo "Server starting in background. Open http://localhost:8080 in your browser" 