"""Serve the loan approval dashboard on the local computer."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from loan_approval import METRICS_PATH, MODEL_PATH, ROOT, score_application


class DashboardHandler(BaseHTTPRequestHandler):
    def send_body(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, value):
        self.send_body(status, "application/json", json.dumps(value).encode("utf-8"))

    def do_GET(self):
        if self.path == "/":
            page = (ROOT / "dashboard.html").read_bytes()
            self.send_body(200, "text/html; charset=utf-8", page)
        elif self.path == "/api/overview":
            metrics = None
            if self.server.metrics_path.exists():
                metrics = json.loads(self.server.metrics_path.read_text(encoding="utf-8"))
            sample = json.loads((ROOT / "sample_application.json").read_text(encoding="utf-8"))
            self.send_json(200, {
                "metrics": metrics,
                "ready": self.server.model_path.exists(),
                "sample": sample,
            })
        else:
            self.send_json(404, {"error": "Page not found."})

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_json(404, {"error": "Endpoint not found."})
            return
        if not self.server.model_path.exists():
            self.send_json(409, {"error": "Run python loan_approval.py train first."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= 4096:
                raise ValueError("Submit an application under 4 KB.")
            application = json.loads(self.rfile.read(length))
            result = score_application(application, self.server.model_path)
        except (ValueError, TypeError, OverflowError) as error:
            self.send_json(400, {"error": str(error)})
            return
        except OSError:
            self.send_json(503, {"error": "The trained model could not be opened."})
            return
        self.send_json(200, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler) as server:
        server.model_path = args.model
        server.metrics_path = args.metrics
        print(f"Open http://127.0.0.1:{server.server_port} in your browser.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
