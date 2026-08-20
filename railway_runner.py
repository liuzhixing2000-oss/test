import html
import os
import traceback
import zipfile
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import btc_expansion_breakout_backtest as bt

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "btc_breakout_results"
ZIP_PATH = ROOT / "btc-breakout-results.zip"
ERROR_PATH = ROOT / "run_error.txt"


def run_backtest():
    os.chdir(ROOT)
    try:
        print("=== Starting BTC Expansion Breakout Backtest ===", flush=True)
        bt.main()

        if RESULT_DIR.exists():
            with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
                for p in RESULT_DIR.rglob("*"):
                    if p.is_file():
                        z.write(p, p.relative_to(ROOT))
            print(f"=== Results ZIP ready: {ZIP_PATH.name} ===", flush=True)
        else:
            raise RuntimeError("Backtest finished but btc_breakout_results/ was not created.")

    except Exception:
        err = traceback.format_exc()
        ERROR_PATH.write_text(err, encoding="utf-8")
        print("=== BACKTEST FAILED ===", flush=True)
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
                if RESULT_DIR.exists():
                    for p in sorted(RESULT_DIR.glob("*")):
                        if p.is_file():
                            rel = p.relative_to(ROOT).as_posix()
                            files.append(f'<li><a href="/{html.escape(rel)}">{html.escape(p.name)}</a></li>')

                body = f"""
                <!doctype html>
                <html><head><meta charset="utf-8"><title>BTC Breakout Backtest</title>
                <style>body{{font-family:sans-serif;max-width:850px;margin:50px auto;padding:0 20px}}a{{font-size:18px}}code{{background:#f2f2f2;padding:2px 5px}}</style>
                </head><body>
                <h1>BTC Expansion / Breakout Backtest</h1>
                <p>Backtest completed successfully.</p>
                <p><strong><a href="/{ZIP_PATH.name}">Download all results (ZIP)</a></strong></p>
                <h2>Individual files</h2><ul>{''.join(files)}</ul>
                </body></html>
                """
            elif ERROR_PATH.exists():
                err = html.escape(ERROR_PATH.read_text(encoding="utf-8"))
                body = f"""
                <!doctype html><html><head><meta charset="utf-8"><title>Backtest Error</title></head>
                <body style="font-family:sans-serif;max-width:1000px;margin:40px auto;padding:0 20px">
                <h1>Backtest failed</h1>
                <p>The Railway service is running, but the backtest failed. The full error is below.</p>
                <p><a href="/run_error.txt">Download run_error.txt</a></p>
                <pre style="white-space:pre-wrap">{err}</pre></body></html>
                """
            else:
                body = """<!doctype html><html><body style='font-family:sans-serif;max-width:800px;margin:40px auto'><h1>No result yet</h1><p>Check Railway deployment logs.</p></body></html>"""

            self.wfile.write(body.encode("utf-8"))
            return

        return super().do_GET()


def main():
    run_backtest()
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Serving results on 0.0.0.0:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
