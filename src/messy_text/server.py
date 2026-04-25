"""Lightweight HTTP API server for the messy-text classifier.

Serves both the web UI (static HTML) and the classification API.

Usage:
    uv run python -m messy_text.server
    # Then open http://localhost:8080
"""

from __future__ import annotations

import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from messy_text.classifier import classify


STATIC_DIR = Path(__file__).parent / "web"
PORT = int(os.environ.get("MESSY_TEXT_PORT", "8080"))


class ClassifierHandler(SimpleHTTPRequestHandler):
    """HTTP handler that serves the web UI and handles API requests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_POST(self):
        """Handle POST /api/classify."""
        if self.path == "/api/classify":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")

            try:
                data = json.loads(body)
                text = data.get("text", "")
            except json.JSONDecodeError:
                text = body

            result = classify(text)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(result.to_json().encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        """Suppress noisy static file logs, keep API logs."""
        msg = str(args[0]) if args else ""
        if "/api/" in msg:
            super().log_message(format, *args)


def main():
    """Start the classifier server."""
    server = HTTPServer(("0.0.0.0", PORT), ClassifierHandler)
    print(f"[messy-text] classifier running at http://localhost:{PORT}")
    print("   Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
