# SPDX-License-Identifier: Apache-2.0
"""Static file server with COOP/COEP headers so served pages are cross-origin
isolated. Cross-origin isolation enables SharedArrayBuffer, which lets
onnxruntime-web use its multi-threaded WASM build (larger heap). It is NOT
strictly required to load the 350m fp32 model (1.3 GB fits the single-threaded
WASM heap once weights are passed via `externalData`), but it is the correct
way to serve larger browser models and avoids surprises.

Usage:
    python web/test/coi_server.py [PORT] [DIR]

Defaults: PORT=8732, DIR=current working directory. Serve the directory that
CONTAINS the packaged repo (transformers.js resolves a model name as a subdir),
or serve the repo itself when loading raw ONNX with onnxruntime-web.
"""
import http.server
import os
import socketserver
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8732
DIRECTORY = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()

    def log_message(self, *a):
        sys.stderr.write("%s - %s\n" % (self.address_string(), a[0] % a[1:]))


with socketserver.ThreadingTCPServer(("127.0.0.1", PORT), Handler) as httpd:
    print(f"serving COI on http://127.0.0.1:{PORT} (dir: {DIRECTORY})", flush=True)
    httpd.serve_forever()
