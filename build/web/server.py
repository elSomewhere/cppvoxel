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
