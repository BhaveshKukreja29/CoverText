"""Local page for encode, decode, and detect.

The three actions call the same functions as the command line, with the same
default context and the same stub-detector seed, so a stub round trip matches
``covertext encode``, ``decode``, and ``detect`` on the same inputs.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from covertext.common.registry import build_encoder
from covertext.demo.cli import (
    DEFAULT_CONTEXT,
    decode_message,
    detect_text,
    encode_message,
    load_shared_model,
)

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CoverText — Linguistic Steganography & Steganalysis</title>
<style>
  :root {
    --bg: #090d16;
    --card: #111827;
    --border: #1f293d;
    --text: #f3f4f6;
    --muted: #9ca3af;
    --primary: #6366f1;
    --primary-hover: #4f46e5;
    --accent: #06b6d4;
    --mono: "SF Mono", Monaco, Consolas, monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.5;
    padding: 2rem 1rem;
    display: flex;
    justify-content: center;
  }
  .container { max-width: 44rem; width: 100%; }
  header { margin-bottom: 1.5rem; text-align: center; }
  .badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: rgba(99, 102, 241, 0.15);
    color: var(--primary);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 9999px;
    margin-bottom: 0.5rem;
  }
  h1 { font-size: 1.85rem; font-weight: 700; letter-spacing: -0.02em; }
  p.subtitle { color: var(--muted); font-size: 0.9rem; margin-top: 0.25rem; }
  .tabs {
    display: flex;
    gap: 0.4rem;
    background: var(--card);
    padding: 0.3rem;
    border-radius: 0.6rem;
    border: 1px solid var(--border);
    margin-bottom: 1.25rem;
  }
  .tab-btn {
    flex: 1;
    padding: 0.55rem 0.75rem;
    border: none;
    background: transparent;
    color: var(--muted);
    font-size: 0.88rem;
    font-weight: 600;
    cursor: pointer;
    border-radius: 0.45rem;
    transition: all 0.15s ease;
  }
  .tab-btn:hover { color: var(--text); }
  .tab-btn.active { background: var(--primary); color: #fff; }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 0.75rem;
    padding: 1.25rem;
    margin-bottom: 1rem;
  }
  .field { margin-bottom: 1rem; }
  label { display: block; font-size: 0.8rem; font-weight: 600; color: var(--muted); margin-bottom: 0.35rem; }
  input, textarea, select {
    width: 100%;
    padding: 0.6rem 0.8rem;
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid var(--border);
    border-radius: 0.45rem;
    color: var(--text);
    font-family: inherit;
    font-size: 0.9rem;
  }
  input:focus, textarea:focus, select:focus { outline: none; border-color: var(--primary); }
  .mono { font-family: var(--mono); font-size: 0.85rem; }
  button.submit {
    background: var(--primary);
    color: #fff;
    border: none;
    padding: 0.65rem 1.25rem;
    border-radius: 0.45rem;
    font-weight: 600;
    font-size: 0.9rem;
    cursor: pointer;
    width: 100%;
    transition: background 0.15s;
  }
  button.submit:hover { background: var(--primary-hover); }
  .btn-sm {
    padding: 0.35rem 0.7rem;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--border);
    border-radius: 0.35rem;
    color: var(--text);
    font-size: 0.78rem;
    cursor: pointer;
  }
  .btn-sm:hover { background: rgba(255, 255, 255, 0.12); }
  .result-box { margin-top: 1.25rem; padding-top: 1rem; border-top: 1px solid var(--border); display: none; }
  .result-title { font-size: 0.8rem; font-weight: 600; color: var(--accent); margin-bottom: 0.4rem; }
  .result-val {
    background: rgba(0, 0, 0, 0.4);
    border: 1px solid var(--border);
    border-radius: 0.45rem;
    padding: 0.75rem;
    font-family: var(--mono);
    font-size: 0.85rem;
    white-space: pre-wrap;
    word-break: break-all;
    margin-bottom: 0.6rem;
  }
  .actions { display: flex; gap: 0.4rem; flex-wrap: wrap; }
  .error-box {
    margin-top: 0.75rem;
    background: rgba(244, 63, 94, 0.1);
    border: 1px solid rgba(244, 63, 94, 0.3);
    color: #fda4af;
    padding: 0.65rem;
    border-radius: 0.45rem;
    font-size: 0.85rem;
    display: none;
  }
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="badge">Small Model Steganography &bull; Qwen2.5-0.5B</div>
    <h1>CoverText</h1>
    <p class="subtitle">Embed and recover secret payloads in fluent neural language generations</p>
  </header>

  <div class="tabs">
    <button class="tab-btn active" id="tab-btn-enc" onclick="switchTab('enc')">Encode</button>
    <button class="tab-btn" id="tab-btn-dec" onclick="switchTab('dec')">Decode</button>
    <button class="tab-btn" id="tab-btn-det" onclick="switchTab('det')">Detect</button>
  </div>

  <div class="card tab-pane" id="pane-enc">
    <div class="field">
      <label for="enc-message">Secret Message</label>
      <input id="enc-message" value="hi" placeholder="Enter plaintext message">
    </div>
    <div class="field">
      <label for="enc-context">Cover Context / Prompt</label>
      <input id="enc-context" value="__CONTEXT__">
    </div>
    <div class="field">
      <label for="enc-encoder">Steganographic Algorithm</label>
      <select id="enc-encoder">
        <option value="stub">Stub (Plain Text Marker)</option>
        <option value="ac">AC (Arithmetic Coding)</option>
        <option value="mec">MEC (Minimum-Entropy Coupling)</option>
      </select>
    </div>
    <button class="submit" id="encode" onclick="runEncode()">Encode Message</button>
    <div class="error-box" id="enc-err"></div>
    <div class="result-box" id="enc-res">
      <div class="result-title">Generated Stegotext Continuation</div>
      <div class="result-val" id="enc-stego"></div>
      <div class="result-title">Payload Bits (<span id="enc-bits-count"></span> bits)</div>
      <div class="result-val mono" id="enc-bits"></div>
      <div class="actions">
        <button class="btn-sm" onclick="sendToDecode()">Transfer to Decoder &rarr;</button>
        <button class="btn-sm" onclick="sendToDetect()">Transfer to Detector &rarr;</button>
      </div>
    </div>
  </div>

  <div class="card tab-pane" id="pane-dec" style="display:none;">
    <div class="field">
      <label for="dec-text">Stegotext (Continuation)</label>
      <textarea id="dec-text" rows="2" class="mono" placeholder="Paste generated stego text here"></textarea>
    </div>
    <div class="field">
      <label for="dec-context">Context / Prompt (Must match encode prompt)</label>
      <input id="dec-context" value="__CONTEXT__">
    </div>
    <div class="field">
      <label for="dec-bits">Payload Bits (num_bits)</label>
      <input id="dec-bits" type="number" min="1" value="16" placeholder="e.g. 16, 32, 64">
    </div>
    <div class="field">
      <label for="dec-encoder">Steganographic Algorithm</label>
      <select id="dec-encoder">
        <option value="stub">Stub (Plain Text Marker)</option>
        <option value="ac">AC (Arithmetic Coding)</option>
        <option value="mec">MEC (Minimum-Entropy Coupling)</option>
      </select>
    </div>
    <button class="submit" id="decode" onclick="runDecode()">Decode Stegotext</button>
    <div class="error-box" id="dec-err"></div>
    <div class="result-box" id="dec-res">
      <div class="result-title">Recovered Plaintext Message</div>
      <div class="result-val" id="dec-msg"></div>
      <div class="result-title">Recovered Bitstream</div>
      <div class="result-val mono" id="dec-out-bits"></div>
    </div>
  </div>

  <div class="card tab-pane" id="pane-det" style="display:none;">
    <div class="field">
      <label for="det-text">Candidate Text to Analyze</label>
      <textarea id="det-text" rows="3" placeholder="Enter text to score for stego presence"></textarea>
    </div>
    <div class="field">
      <label for="det-detector">Detector Mode</label>
      <select id="det-detector">
        <option value="stub">Stub Detector</option>
        <option value="ppl">PPL Detector (Perplexity & Entropy Deficit)</option>
      </select>
    </div>
    <button class="submit" id="detect" onclick="runDetect()">Score Text</button>
    <div class="error-box" id="det-err"></div>
    <div class="result-box" id="det-res">
      <div class="result-title">Steganalysis Detection Scores</div>
      <div class="result-val mono" id="det-out"></div>
    </div>
  </div>
</div>

<script>
let lastResult = null;

async function postJson(url, data) {
  const resp = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(data),
  });
  const body = await resp.json();
  if (!resp.ok) throw new Error(body.error || resp.statusText);
  return body;
}

function switchTab(tab) {
  ["enc", "dec", "det"].forEach(t => {
    document.getElementById("pane-" + t).style.display = (t === tab) ? "block" : "none";
    document.getElementById("tab-btn-" + t).className = "tab-btn" + (t === tab ? " active" : "");
  });
}

function setErr(prefix, msg) {
  const el = document.getElementById(prefix + "-err");
  el.style.display = msg ? "block" : "none";
  el.textContent = msg || "";
}

async function runEncode() {
  setErr("enc", null);
  document.getElementById("enc-res").style.display = "none";
  try {
    const payload = {
      message: document.getElementById("enc-message").value,
      context: document.getElementById("enc-context").value,
      encoder: document.getElementById("enc-encoder").value,
    };
    lastResult = await postJson("/api/encode", payload);
    document.getElementById("enc-stego").textContent = lastResult.stego_text;
    document.getElementById("enc-bits-count").textContent = lastResult.num_bits;
    document.getElementById("enc-bits").textContent = lastResult.payload_bits;
    document.getElementById("enc-res").style.display = "block";
  } catch (err) { setErr("enc", err.message); }
}

async function runDecode() {
  setErr("dec", null);
  document.getElementById("dec-res").style.display = "none";
  try {
    const numBits = parseInt(document.getElementById("dec-bits").value, 10);
    const payload = {
      text: document.getElementById("dec-text").value,
      context: document.getElementById("dec-context").value,
      num_bits: isNaN(numBits) ? 0 : numBits,
      encoder: document.getElementById("dec-encoder").value,
    };
    const res = await postJson("/api/decode", payload);
    document.getElementById("dec-msg").textContent = res.message !== null ? res.message : "(Non-text binary payload)";
    document.getElementById("dec-out-bits").textContent = res.payload_bits;
    document.getElementById("dec-res").style.display = "block";
  } catch (err) { setErr("dec", err.message); }
}

async function runDetect() {
  setErr("det", null);
  document.getElementById("det-res").style.display = "none";
  try {
    const payload = {
      text: document.getElementById("det-text").value,
      detector: document.getElementById("det-detector").value,
    };
    const res = await postJson("/api/detect", payload);
    document.getElementById("det-out").textContent = JSON.stringify(res.scores, null, 2);
    document.getElementById("det-res").style.display = "block";
  } catch (err) { setErr("det", err.message); }
}

function sendToDecode() {
  if (!lastResult) return;
  document.getElementById("dec-text").value = lastResult.stego_text;
  document.getElementById("dec-context").value = lastResult.context;
  document.getElementById("dec-bits").value = lastResult.num_bits;
  document.getElementById("dec-encoder").value = document.getElementById("enc-encoder").value;
  switchTab("dec");
}

function sendToDetect() {
  if (!lastResult) return;
  document.getElementById("det-text").value = lastResult.stego_text;
  switchTab("det");
}
</script>
</body>
</html>
""".replace("__CONTEXT__", DEFAULT_CONTEXT)


class App:
    """HTTP handlers. The stub detector seed matches the command-line default."""

    def __init__(self, detector_seed: int = 42):
        self.detector_seed = detector_seed

    def encode(self, payload: dict) -> dict:
        message = payload.get("message")
        context = payload.get("context", DEFAULT_CONTEXT)
        encoder_name = payload.get("encoder", "stub")
        if not message:
            raise ValueError("message is required")
        return encode_message(_encoder(encoder_name), message, context)

    def decode(self, payload: dict) -> dict:
        text = payload.get("text") or ""
        context = payload.get("context", DEFAULT_CONTEXT)
        num_bits = payload.get("num_bits")
        encoder_name = payload.get("encoder", "stub")
        if not text:
            raise ValueError("text is required")
        if not isinstance(num_bits, int) or num_bits < 1:
            raise ValueError("num_bits must be a positive integer")
        return decode_message(_encoder(encoder_name), text, context, num_bits)

    def detect(self, payload: dict) -> dict:
        text = payload.get("text") or ""
        detector_name = (payload.get("detector") or "stub").strip().lower()
        from covertext.detector.stub import StubDetector

        detectors = [StubDetector(seed=self.detector_seed)]
        if detector_name == "ppl":
            model, tokenizer = load_shared_model()
            from covertext.detector.ppl_detector import PPLDetector

            ppl = PPLDetector(model, tokenizer)
            ppl.calibrate([DEFAULT_CONTEXT, "A river cuts through the valley below the ridge."])
            detectors.append(ppl)
        return detect_text(text, detectors)


def _encoder(name: str):
    """MEC's default seed is 0, which is what the command line uses when none is set."""
    model = tokenizer = None
    if name.strip().lower() != "stub":
        model, tokenizer = load_shared_model()
    kwargs = {"seed": 0} if name.strip().lower() == "mec" else {}
    return build_encoder(name, model, tokenizer, **kwargs)


def make_server(host: str = "127.0.0.1", port: int = 0, detector_seed: int = 42) -> ThreadingHTTPServer:
    app = App(detector_seed=detector_seed)
    handler = _handler(app)
    return ThreadingHTTPServer((host, port), handler)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = make_server(host, port)
    print(f"CoverText demo at http://{host}:{port}")
    server.serve_forever()


def _handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path.split("?", 1)[0] != "/":
                self._send(404, {"error": "not found"})
                return
            body = _PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8") if length else "{}"
                payload = json.loads(raw)
                if path == "/api/encode":
                    self._send(200, app.encode(payload))
                elif path == "/api/decode":
                    self._send(200, app.decode(payload))
                elif path == "/api/detect":
                    self._send(200, app.detect(payload))
                else:
                    self._send(404, {"error": "not found"})
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                self._send(400, {"error": str(exc)})

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            return

    return Handler


def main() -> None:
    serve()


if __name__ == "__main__":
    main()
