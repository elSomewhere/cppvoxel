#!/usr/bin/env python3
import http.server
import socketserver
import os
import sys
from pathlib import Path

PORT = 8082
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
