"""TripoSplat Camera Viewer — open the viewer in the default browser.

Starts a tiny local HTTP server on an auto-selected free port (no port
conflicts), serves static/, and opens cam_viewer.html in the system's default
browser. No Qt / WebEngine dependencies — uses the browser's own WebGL + ES
module support, which is more reliable than an embedded engine.

Renders the dataset camera frustums (intrinsics + extrinsics), the sparse
point cloud, and an optional TripoSplat-generated Gaussian splat — all in one
browser tab, with photo / splat / both image-plane comparison.

Usage
    python run_cam_viewer.py
    python run_cam_viewer.py --splat gradio_outputs/abc123/splat.ply
    python run_cam_viewer.py --base static/000001/mvs --splat static/000001/splat.ply
    python run_cam_viewer.py --port 8000          # pin a specific port
"""
import argparse
import http.server
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path


def find_free_port(preferred=None):
    """Return a free TCP port, preferring `preferred` if it is available."""
    if preferred is not None:
        with socketserver.TCPServer(("127.0.0.1", preferred), None) as s:
            s.__class__ = socketserver.TCPServer  # for the context manager
        try:
            return preferred
        except Exception:
            pass
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description="TripoSplat Camera Viewer (browser)")
    parser.add_argument(
        "--base", type=str, default="000000/mvs",
        help="Path to the mvs folder, relative to static/ (or an absolute path). "
             "Default: 000000/mvs  →  static/000000/mvs",
    )
    parser.add_argument(
        "--splat", type=str, default=None,
        help="Path to a TripoSplat-generated splat file (PLY/Splat), relative "
             "to static/ or absolute. Default: 000000/splat.ply if it exists.",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Port to serve on. Default: auto-select a free port.",
    )
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open the browser.")
    args = parser.parse_args()

    static_dir = (Path(__file__).resolve().parent / "static").resolve()
    html_rel = "cam_viewer/cam_viewer.html"
    html_path = static_dir / html_rel
    if not html_path.is_file():
        raise SystemExit(f"cam_viewer.html not found at {html_path}")

    # Resolve splat default: static/000000/splat.ply if it exists.
    splat_arg = args.splat
    if splat_arg is None:
        candidate = static_dir / "000000" / "splat.ply"
        if candidate.is_file():
            splat_arg = "000000/splat.ply"

    # Normalize base path to be relative to static/ for the URL.
    base_arg = args.base
    # If absolute, make it relative to static/ if it's inside; otherwise serve
    # is limited to static/, so warn.
    base_abs = Path(base_arg)
    if base_abs.is_absolute():
        try:
            base_arg = str(base_abs.resolve().relative_to(static_dir)).replace("\\", "/")
        except ValueError:
            raise SystemExit(f"--base must be inside static/ ({static_dir}), got {base_arg}")
    # Ensure trailing slash so URL resolution treats it as a folder.
    if not base_arg.endswith("/"):
        base_arg += "/"

    # Build the query string. Under HTTP, query strings work reliably.
    from urllib.parse import urlencode
    qs = {"base": "/" + base_arg}
    if splat_arg is not None:
        # Splat path also relative to static/ root as a served URL.
        splat_clean = splat_arg.replace("\\", "/").lstrip("/")
        qs["splat"] = "/" + splat_clean

    port = find_free_port(args.port)

    # Serve static/ with a simple handler that allows directory listing off.
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(static_dir), **kw)
        def log_message(self, fmt, *a):
            pass  # quiet

    httpd = socketserver.TCPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/{html_rel}?{urlencode(qs)}"

    print(f"Serving {static_dir} on http://127.0.0.1:{port}")
    print(f"Viewer URL: {url}")
    print(f"  base:  {qs['base']}")
    print(f"  splat: {qs.get('splat', '(none)')}")
    print("Press Ctrl+C to stop.")

    if not args.no_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
