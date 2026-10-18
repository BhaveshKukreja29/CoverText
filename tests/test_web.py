"""The local page returns the same JSON as the command line on stub inputs."""

import json
import threading
import urllib.request

from covertext.demo.cli import DEFAULT_CONTEXT, decode_message, encode_message
from covertext.demo.web import make_server
from covertext.detector.stub import StubDetector
from covertext.encoder.stub import StubEncoder


def test_web_matches_cli_on_stub_round_trip():
    server = make_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        encoded = _post(
            host,
            port,
            "/api/encode",
            {"message": "hi", "context": DEFAULT_CONTEXT, "encoder": "stub"},
        )
        assert encoded == encode_message(StubEncoder(), "hi", DEFAULT_CONTEXT)
        decoded = _post(
            host,
            port,
            "/api/decode",
            {
                "text": encoded["stego_text"],
                "context": DEFAULT_CONTEXT,
                "num_bits": encoded["num_bits"],
                "encoder": "stub",
            },
        )
        assert decoded == decode_message(
            StubEncoder(), encoded["stego_text"], DEFAULT_CONTEXT, encoded["num_bits"]
        )
        assert decoded["message"] == "hi"
        detected = _post(host, port, "/api/detect", {"text": encoded["stego_text"]})
        assert detected["scores"]["STUB"] == StubDetector(seed=42).score(encoded["stego_text"])
        page = urllib.request.urlopen(f"http://{host}:{port}/").read().decode("utf-8")
        assert "Encode" in page and DEFAULT_CONTEXT in page
    finally:
        server.shutdown()


def test_web_rejects_a_missing_message():
    server = make_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        status, payload = _post_status(host, port, "/api/encode", {"encoder": "stub"})
        assert status == 400
        assert "message" in payload["error"]
    finally:
        server.shutdown()


def _post(host, port, path, payload):
    status, body = _post_status(host, port, path, payload)
    assert status == 200
    return body


def _post_status(host, port, path, payload):
    request = urllib.request.Request(
        f"http://{host}:{port}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))
