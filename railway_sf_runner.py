
import html
import os
import traceback
import zipfile
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import btc_breakout_success_failure_study as study

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "btc_sf_results"
ZIP_PATH = ROOT / "btc-breakout-success-failure-results.zip"
ERROR_PATH = ROOT / "run_error.txt"

def run_study():
    os.chdir(ROOT)
    try:
        print("=== Starting BTC breakout success/failure study ===", flush=True)
        study.main()
        if RESULT_DIR.exists():
            with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
                for p in RESULT_DIR.rglob("*"):
                    if p.is_file():
                        z.write(p, p.relative_to(ROOT))
            print(f"=== Results ZIP ready: {ZIP_PATH.name} ===", flush=True)
        else:
            raise RuntimeError("Result directory not created.")
    except Exception:
        err = traceback.format_exc()
        ERROR_PATH.write_text(err, encoding="utf-8")
        print("=== STUDY FAILED ===", flush=True)
        print(err, flush=True)

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()

            if ZIP_PATH.exists():
                files = []
                for p in sorted(RESULT_DIR.glob("*")):
                    if p.is_file():
                        rel = p.relative_to(ROOT).as_posix()
                        files.append(f'<li><a href="/{html.escape(rel)}">{html.escape(p.name)}</a></li>')
                body = f"""
                <html><body style="font-family:sans-serif;max-width:950px;margin:50px auto;padding:0 20px">
                <h1>BTC Breakout Success vs Failure Study</h1>
                <p><strong><a href="/{ZIP_PATH.name}">Download all results (ZIP)</a></strong></p>
                <ul>{''.join(files)}</ul>
                </body></html>
                """
            elif ERROR_PATH.exists():
                err = html.escape(ERROR_PATH.read_text(encoding="utf-8"))
                body = f"<html><body><h1>Study failed</h1><pre>{err}</pre></body></html>"
            else:
                body = "<html><body><h1>No results yet</h1></body></html>"

            self.wfile.write(body.encode("utf-8"))
            return
        return super().do_GET()

def main():
    run_study()
    port = int(os.environ.get("PORT", "8080"))
    print(f"Serving on 0.0.0.0:{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()

if __name__ == "__main__":
    main()
