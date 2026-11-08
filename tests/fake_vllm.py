"""Scriptable fake of the vLLM chat endpoint, shared by the client tests."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OK_BODY = {"choices": [{"message": {"content": '{"type":"ask_user"}', "reasoning_content": "thought"},
                        "finish_reason": "stop"}],
           "usage": {"prompt_tokens": 11, "completion_tokens": 7}}


def sse_events(content=(), reasoning=(), usage=(11, 7), finish="stop", done=True):
    """SSE events as a vLLM stream would send them: reasoning deltas, content deltas, finish, usage, [DONE]."""
    def ev(obj):
        return b"data: " + json.dumps(obj).encode() + b"\n\n"
    events = [ev({"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]})]
    events += [ev({"choices": [{"delta": {"reasoning_content": t}, "finish_reason": None}]}) for t in reasoning]
    events += [ev({"choices": [{"delta": {"content": t}, "finish_reason": None}]}) for t in content]
    events.append(ev({"choices": [{"delta": {}, "finish_reason": finish}]}))
    if usage:
        events.append(ev({"choices": [], "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}}))
    if done:
        events.append(b"data: [DONE]\n\n")
    return events


class Stream:
    """Scripted streaming response: one HTTP chunk per event, `interval_s` apart.
    chunked=False sends a close-delimited body; split=True cuts every event in two TCP writes;
    drop_after=n closes the connection after n events without the terminating chunk."""

    def __init__(self, events, interval_s=0.0, chunked=True, split=False, drop_after=None):
        self.events, self.interval_s, self.chunked = list(events), interval_s, chunked
        self.split, self.drop_after = split, drop_after


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
                if isinstance(body, Stream):
                    return self.send_stream(body)
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    pass

            def send_stream(self, stream):
                self.close_connection = True
                head = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n"
                head += b"Transfer-Encoding: chunked\r\n\r\n" if stream.chunked else b"\r\n"
                try:
                    self.wfile.write(head)
                    self.wfile.flush()
                    for n, event in enumerate(stream.events):
                        if stream.drop_after is not None and n >= stream.drop_after:
                            return
                        time.sleep(stream.interval_s)
                        framed = b"%x\r\n%s\r\n" % (len(event), event) if stream.chunked else event
                        cut = len(framed) // 2 if stream.split else len(framed)
                        self.wfile.write(framed[:cut])
                        self.wfile.flush()
                        if cut < len(framed):
                            time.sleep(0.02)
                            self.wfile.write(framed[cut:])
                            self.wfile.flush()
                    if stream.chunked:
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
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
