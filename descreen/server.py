"""Local HTTP server for the descreen tool (stdlib only).

It serves a single-page UI and a small JSON API. The browser never re-implements
detection or processing; it only sends parameters and displays returned images.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from .analysis import analyze_image, channel_array, crop_overview_png, luma
from .pipeline import result_json, result_pngs, run_pipeline
from .synth import (
    make_fabric_fixture,
    make_known_screen_fixture,
    make_palette_fixture,
    make_photo_fixture,
    make_text_fixture,
    to_png_bytes,
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
_STORES: dict[str, dict] = {}
_STORE_LOCK = threading.Lock()


def _fixture(name: str) -> tuple[np.ndarray, dict]:
    makers = {
        "text": make_text_fixture,
        "fabric": make_fabric_fixture,
        "photo": make_photo_fixture,
        "palette": make_palette_fixture,
        "known": make_known_screen_fixture,
    }
    if name not in makers:
        raise ValueError(f"unknown fixture {name!r}")
    return makers[name]()


def _decode_image(data: bytes) -> tuple[np.ndarray, str]:
    im = Image.open(io.BytesIO(data))
    im.load()
    if im.mode in ("L", "I;16", "I"):
        arr = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        return arr, "grayscale"
    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return arr, "rgb"


def _png(arr: np.ndarray) -> bytes:
    return to_png_bytes(arr)


class Handler(BaseHTTPRequestHandler):
    server_version = "descreen/1.0"

    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str,
              extra: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj, default=_default).encode("utf-8"),
                   "application/json")

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def do_GET(self):
        u = urlparse(self.path)
        path = u.path
        if path in ("/", "/index.html"):
            return self._serve_file("index.html", "text/html; charset=utf-8")
        if path == "/app.js":
            return self._serve_file("app.js", "application/javascript")
        if path == "/app.css":
            return self._serve_file("app.css", "text/css")
        if path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")
        if path == "/api/fixtures":
            return self._json(200, {"fixtures": [
                {"id": "text", "label": "Text + hairlines (screened 7.2 px / 105°)"},
                {"id": "text_clean", "label": "Text + hairlines (no screen)"},
                {"id": "fabric", "label": "Woven fabric (real 9/12.5 px repeats)"},
                {"id": "fabric_clean", "label": "Woven fabric (no screen)"},
                {"id": "photo", "label": "Photo (screened 8 px / 97°)"},
                {"id": "photo_clean", "label": "Photo (no screen)"},
                {"id": "palette", "label": "Color page, 3 rotated screens"},
                {"id": "known", "label": "Flat tone, known 6 px / 33° screen"},
            ]})
        if path == "/api/state":
            q = parse_qs(u.query)
            sid = (q.get("session") or [""])[0]
            with _STORE_LOCK:
                store = _STORES.get(sid)
            if not store:
                return self._json(404, {"error": "unknown session"})
            return self._json(200, store["meta"])
        if path == "/api/image":
            q = parse_qs(u.query)
            sid = (q.get("session") or [""])[0]
            with _STORE_LOCK:
                store = _STORES.get(sid)
            if not store:
                return self._json(404, {"error": "unknown session"})
            rgb = store["rgb"]
            return self._send(200, _source_png(rgb), "image/png",
                              extra={"X-Full-Width": str(rgb.shape[1]),
                                     "X-Full-Height": str(rgb.shape[0])})
        return self._send(404, b"not found", "text/plain")

    def _serve_file(self, name: str, ctype: str) -> None:
        path = os.path.join(STATIC_DIR, name)
        try:
            with open(path, "rb") as f:
                data = f.read()
        except FileNotFoundError:
            return self._send(404, b"missing static file", "text/plain")
        return self._send(200, data, ctype)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            if u.path == "/api/load_fixture":
                return self._load_fixture(self._read_json())
            if u.path == "/api/upload":
                return self._upload()
            if u.path == "/api/analyze":
                return self._analyze(self._read_json())
            if u.path == "/api/run":
                return self._run(self._read_json())
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        except Exception as exc:  # surfaced to the UI for manual adjustment
            import traceback
            traceback.print_exc()
            return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
        return self._send(404, b"not found", "text/plain")

    # ---- API ----

    def _new_session(self, rgb: np.ndarray, meta: dict) -> dict:
        import uuid
        sid = uuid.uuid4().hex[:12]
        with _STORE_LOCK:
            _STORES[sid] = {"rgb": rgb, "meta": meta, "result": None}
        return {"session": sid, **meta}

    def _load_fixture(self, body: dict) -> None:
        name = body.get("fixture", "text")
        if name.endswith("_clean"):
            base = name[:-6]
            makers = {
                "text": lambda: make_text_fixture(screen=None),
                "fabric": lambda: make_fabric_fixture(screen=None),
                "photo": lambda: make_photo_fixture(screen=None),
            }
            if base not in makers:
                raise ValueError(f"no clean variant for {name!r}")
            rgb, meta = makers[base]()
        else:
            rgb, meta = _fixture(name)
        info = {
            "kind": meta.get("kind", name),
            "shape": list(rgb.shape[:2]),
            "channels": 1 if rgb.ndim == 2 else 3,
            "truth": {k: v for k, v in meta.items() if k != "kind"},
        }
        return self._json(200, self._new_session(rgb, info))

    def _upload(self) -> None:
        data = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        arr, kind = _decode_image(data)
        info = {"kind": "upload", "shape": list(arr.shape[:2]),
                "channels": 1 if arr.ndim == 2 else 3,
                "truth": {"note": "uploaded image; no ground truth"}}
        return self._json(200, self._new_session(arr, info))

    def _get(self, body: dict) -> tuple[str, dict]:
        sid = body.get("session", "")
        with _STORE_LOCK:
            store = _STORES.get(sid)
        if not store:
            raise ValueError("unknown session; load an image first")
        return sid, store

    def _analyze(self, body: dict) -> None:
        sid, store = self._get(body)
        rgb = store["rgb"]
        roi = body.get("roi")
        roi_t = tuple(int(v) for v in roi) if roi else None
        x0, y0, rw, rh = roi_t if roi_t else (0, 0, rgb.shape[1], rgb.shape[0])
        crop = rgb[y0:y0 + rh, x0:x0 + rw]
        res = analyze_image(crop, None,
                            deep_channels=bool(body.get("deep_channels")),
                            flat_tile_assist=bool(body.get("flat_tile_assist", True)))
        luma_arr = channel_array(crop, res.get("selected_channel", "luma"))
        payload = {
            "session": sid,
            "analysis": _strip_arrays(res),
            "roi_absolute": [int(x0), int(y0), int(rw), int(rh)],
            "images": {
                "overview": _b64(crop_overview_png(crop, (0, 0, crop.shape[1],
                                                          crop.shape[0]))),
                "spectrum": _b64(_spectrum_for(luma_arr, res["peaks"])),
            },
        }
        store["last_analysis"] = res
        store["last_crop"] = (x0, y0, rw, rh)
        return self._json(200, payload)

    def _run(self, body: dict) -> None:
        sid, store = self._get(body)
        rgb = store["rgb"]
        roi = body.get("roi")
        roi_t = tuple(int(v) for v in roi) if roi else None
        params = body.get("params", {})
        manual = body.get("manual_peaks") or None
        reject = body.get("reject_peaks") or None
        scale = float(params.get("scale", 1.0))
        method = params.get("method", "frequency_notch")
        if method not in ("frequency_notch", "spatial_fir"):
            raise ValueError("method must be frequency_notch or spatial_fir")
        radius = params.get("radius_cyc_per_px")
        radius = None if radius in (None, "", 0, "0") else float(radius)
        harmonics = tuple(int(k) for k in params.get("harmonic_orders", [1, 2]))
        harmonics = tuple(sorted({k for k in harmonics if k in (1, 2, 3)}))

        result = run_pipeline(
            rgb,
            roi=roi_t,
            scale=scale,
            method=method,
            deep_channels=bool(params.get("deep_channels")),
            flat_tile_assist=bool(params.get("flat_tile_assist", True)),
            manual_peaks=manual,
            reject_peaks=reject,
            harmonic_orders=harmonics or (1,),
            include_lattice_lines=bool(params.get("include_lattice_lines", True)),
            radius_cyc_per_px=radius,
            apodization=float(params.get("apodization", 0.35)),
            strength=float(params.get("strength", 1.0)),
            pad=_opt_int(params.get("pad")),
            feather=_opt_int(params.get("feather")),
            taps=_opt_int(params.get("taps")),
            force_peaks=bool(params.get("force_peaks")),
        )
        store["result"] = result
        pngs = result_pngs(result)
        j = result_json(result)
        store["last_json"] = j
        payload = {
            "session": sid,
            "result": _strip_arrays(result),
            "params_json": j,
            "images": {k: _b64(v) for k, v in pngs.items()},
        }
        return self._json(200, payload)


def _opt_int(v):
    if v in (None, "", 0, "0"):
        return None
    return int(v)


def _default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(type(obj))


def _strip_arrays(obj):
    """Remove non-serializable/oversized diagnostic arrays from API payloads."""
    if isinstance(obj, dict):
        return {k: _strip_arrays(v) for k, v in obj.items()
                if k not in ("logmag", "bg", "_images")}
    if isinstance(obj, (list, tuple)):
        return [_strip_arrays(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return None
    return obj


def _spectrum_for(arr: np.ndarray, peaks: list[dict]) -> bytes:
    from .analysis import spectrum_png
    return spectrum_png(arr, peaks, output_side=640)


def _b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode("ascii")


def _source_png(rgb: np.ndarray, *, max_side: int = 1100) -> bytes:
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim == 2:
        im = Image.fromarray(np.clip(arr * 255 + 0.5, 0, 255).astype(np.uint8), "L")
    else:
        im = Image.fromarray(np.clip(arr * 255 + 0.5, 0, 255).astype(np.uint8), "RGB")
    if max(im.size) > max_side:
        f = max_side / max(im.size)
        im = im.resize((max(1, int(im.width * f)), max(1, int(im.height * f))),
                       Image.Resampling.NEAREST)
    return to_png_bytes(np.asarray(im, dtype=np.float32) / 255.0)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"descreen UI on http://{host}:{port}/", flush=True)
    httpd.serve_forever()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Serve the descreen web UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
