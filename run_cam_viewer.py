"""TripoSplat Camera Viewer — open the viewer in the default browser.

Starts a tiny local HTTP server on an auto-selected free port (no port
conflicts), serves the PROJECT ROOT, and opens cam_viewer.html in the system's
default browser. Serving from the project root (not just static/) means splat
files anywhere in the repo — scratch_output/, gradio_outputs/, demo_output/,
etc. — are reachable via relative paths from the HTML.

No Qt / WebEngine dependencies — uses the browser's own WebGL + ES module
support, which is more reliable than an embedded engine.

Renders the dataset camera frustums (intrinsics + extrinsics), the sparse
point cloud, and an optional TripoSplat-generated Gaussian splat — all in one
browser tab, with photo / splat / both image-plane comparison.

Usage
    python run_cam_viewer.py
    python run_cam_viewer.py --splat scratch_output/refined.ply
    python run_cam_viewer.py --splat ../scratch_output/refined.ply   # relative to HTML
    python run_cam_viewer.py --base static/000001/mvs --splat static/000001/splat.ply
    python run_cam_viewer.py --port 8848          # pin a specific port

Path conventions for --base / --splat:
    - Relative paths are resolved relative to the HTML file
      (static/cam_viewer/), so "../scratch_output/refined.ply" reaches the
      project root's scratch_output/ folder, and "../000000/splat.ply" reaches
      static/000000/splat.ply.
    - Absolute paths are used as-is (must be inside the project root to be
      served).
"""
import argparse
import http.server
import socketserver
import threading
import webbrowser
from pathlib import Path
from urllib.parse import urlencode, quote


def find_free_port(preferred=None):
    """Return a free TCP port, preferring `preferred` if it is available."""
    import socket
    if preferred is not None:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", preferred))
                return preferred
        except OSError:
            pass  # preferred port busy → auto-select
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description="TripoSplat Camera Viewer (browser)")
    parser.add_argument(
        "--base", type=str, default=None,
        help="Path to the mvs folder, relative to the HTML file or absolute. "
             "Default: ../000000/mvs/ (i.e. static/000000/mvs/).",
    )
    parser.add_argument(
        "--splat", type=str, default=None,
        help="Path to a TripoSplat-generated splat file (PLY/Splat), relative "
             "to the HTML file or absolute. Default: ../000000/splat.ply if it "
             "exists, else none.",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Port to serve on. Default: auto-select a free port.",
    )
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open the browser.")
    args = parser.parse_args()

    # Serve from the PROJECT ROOT so splat files anywhere in the repo are
    # reachable. The HTML lives at static/cam_viewer/cam_viewer.html.
    project_root = Path(__file__).resolve().parent
    html_rel = "static/cam_viewer/cam_viewer.html"
    html_path = project_root / html_rel
    if not html_path.is_file():
        raise SystemExit(f"cam_viewer.html not found at {html_path}")

    # The HTML file's directory, for resolving relative --base/--splat defaults.
    html_dir = html_path.parent  # static/cam_viewer/

    # Resolve base default: static/000000/mvs/ relative to project root.
    base_arg = args.base if args.base is not None else "static/000000/mvs/"
    # Normalize to a path relative to the project root, then make it an
    # absolute URL path (starts with /) so it resolves correctly regardless of
    # the HTML file's depth in the directory tree.
    def to_url_path(p):
        """Convert a --base/--splat argument to an absolute URL path.

        Accepts paths relative to the project root or absolute filesystem
        paths (which must be inside the project root). Returns a POSIX-style
        path with leading '/'.
        """
        pp = Path(p)
        if not pp.is_absolute():
            pp = (project_root / pp)
        pp = pp.resolve()
        try:
            rel = pp.relative_to(project_root)
        except ValueError:
            raise SystemExit(f"path must be inside the project root ({project_root}), got {p}")
        return "/" + rel.as_posix()

    base_url_path = to_url_path(base_arg)
    if not base_url_path.endswith("/"):
        base_url_path += "/"

    # Resolve splat default: static/000000/splat.ply if it exists.
    splat_arg = args.splat
    if splat_arg is None:
        candidate = project_root / "static" / "000000" / "splat.ply"
        if candidate.is_file():
            splat_arg = "static/000000/splat.ply"

    # Build the query string with absolute URL paths (no ../ counting needed).
    qs = {"base": base_url_path}
    if splat_arg is not None:
        qs["splat"] = to_url_path(splat_arg)

    port = find_free_port(args.port)

    # Serve the project root.
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(project_root), **kw)
        def log_message(self, fmt, *a):
            pass  # quiet

    httpd = socketserver.TCPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/{html_rel}?{urlencode(qs, quote_via=quote)}"

    print(f"Serving {project_root} on http://127.0.0.1:{port}")
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
