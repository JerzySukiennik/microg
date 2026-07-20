"""
Tiny receiver for browser-rendered PNGs.

Why a server: macOS QuickLook (qlmanage) rasterises SVG onto an opaque white
background, so every "transparent" PNG it produced was a white box — the logo
in the README was literally white-on-white. A browser canvas starts fully
transparent, which is what we need, but a page cannot write to disk. So the
page renders and POSTs the bytes here.

Runs for a fixed number of seconds, writes whatever arrives, exits.
"""

import http.server
import os
import sys
import threading

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
PORT = 8765


class Handler(http.server.BaseHTTPRequestHandler):
    def _cors(self):
        # The page is loaded from file://, so every request here is
        # cross-origin. Allow it explicitly or the fetch never arrives.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        """Serve the rasteriser page itself.

        Loading it from file:// and POSTing here counts as a public-to-private
        network request, which Chrome blocks outright — the fetch fails before
        it leaves the page. Served from this origin, the POST is same-origin
        and simply works.
        """
        page = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "rasterize.html")
        with open(page, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        name = os.path.basename(self.path.lstrip("/"))
        length = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(length)
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, name), "wb") as f:
            f.write(data)
        print(f"wrote {name}  {len(data)/1024:.1f} KB", flush=True)
        self.send_response(200)
        self._cors()
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    threading.Timer(seconds, srv.shutdown).start()
    print(f"listening on {PORT}, writing to {OUT}, for {seconds}s", flush=True)
    srv.serve_forever()
    print("done", flush=True)
