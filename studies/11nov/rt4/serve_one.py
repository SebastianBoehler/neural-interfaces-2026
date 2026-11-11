"""Serve one local file once to the Codabench page (the Chrome tool cannot attach files above 10 MB).
Binds to 127.0.0.1 only, answers a single GET for /file with a CORS header for codabench.org, then exits.
Usage: python serve_one.py FILE PORT"""
import http.server, sys
PATH, PORT = sys.argv[1], int(sys.argv[2])


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/file":
            self.send_error(404); return
        data = open(PATH, "rb").read()
        self.send_response(200); self.send_header("Content-Type", "application/zip"); self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "https://www.codabench.org"); self.end_headers(); self.wfile.write(data)
        self.server.done = True

    def log_message(self, *a):
        pass


srv = http.server.HTTPServer(("127.0.0.1", PORT), H); srv.done = False; srv.timeout = 600
while not srv.done:
    srv.handle_request()
print("served once; exiting")
