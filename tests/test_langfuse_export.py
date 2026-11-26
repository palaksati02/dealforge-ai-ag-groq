"""Requirement 9: the Langfuse integration really emits spans. A local mock server stands in for
Langfuse's OTLP ingestion endpoint, so this proves the SDK calls and payload content with no network."""
import gzip
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from app.core.config import settings
from app.core.observability import ObservabilityTracer


def _mock_langfuse():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("content-length", 0)))
            if self.headers.get("content-encoding") == "gzip":
                body = gzip.decompress(body)
            received.append((self.path, self.headers.get("authorization", ""), body))
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, received


def test_spans_and_token_usage_reach_langfuse_ingestion(monkeypatch):
    server, received = _mock_langfuse()
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setattr(settings, "LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setattr(settings, "LANGFUSE_HOST", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", True)
    try:
        tr = ObservabilityTracer()
        assert tr.langfuse_client is not None
        tr.start_trace("lf-test-1", "wf-1", "req-1", "Purchase 5 laptops")
        tr.start_span("lf-test-1", "Requirement Analysis Agent", "AGENT").finish(
            outputs={"ok": True}, prompt_tokens=1234, completion_tokens=321)
        tr.start_span("lf-test-1", "MCP search_vendors", "TOOL").finish(outputs={"n": 3})
        summary = tr.end_trace("lf-test-1")
        tr.langfuse_client.flush()

        deadline = time.time() + 8
        while time.time() < deadline and not received:
            time.sleep(0.2)
        assert received, "nothing reached the Langfuse ingestion endpoint"
        path, auth, _ = received[0]
        assert path == "/api/public/otel/v1/traces" and auth.startswith("Basic ")
        blob = b"".join(b for _, _, b in received)
        for needle in (b"DealForge_Procurement_Workflow", b"Requirement Analysis Agent", b"MCP search_vendors", b"1234"):
            assert needle in blob, needle
        assert summary["langfuse"] == "submitted"
    finally:
        try:
            tr.langfuse_client.shutdown()   # flush + stop background exporter BEFORE the mock server goes away
        except Exception:
            pass
        server.shutdown()
