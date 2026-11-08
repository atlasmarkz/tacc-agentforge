"""Scriptable fake of the vLLM chat endpoint, shared by the client tests."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OK_BODY = {"choices": [{"message": {"content": '{"type":"ask_user"}', "reasoning_content": "thought"},
                        "finish_reason": "stop"}],
           "usage": {"prompt_tokens": 11, "completion_tokens": 7}}


class FakeVllm:
    """Scriptable chat endpoint. `script` is consumed one entry per request; the last entry repeats.
    Entry: (status, body_dict_or_bytes, delay_s)."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                outer.requests.append(json.loads(self.rfile.read(length)))
                idx = min(len(outer.requests) - 1, len(outer.script) - 1)
                status, body, delay = outer.script[idx]
                time.sleep(delay)
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    pass

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
