
import html, os, traceback, zipfile
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import btc_breakout_layered_study as study

ROOT=Path(__file__).resolve().parent
RESULT=ROOT/"btc_layered_results"
ZIP=ROOT/"btc-breakout-layered-results.zip"
ERR=ROOT/"run_error.txt"

def run():
    os.chdir(ROOT)
    try:
        print("=== Starting BTC layered breakout probability study ===", flush=True)
        study.main()
        with zipfile.ZipFile(ZIP,"w",zipfile.ZIP_DEFLATED) as z:
            for p in RESULT.rglob("*"):
                if p.is_file(): z.write(p,p.relative_to(ROOT))
        print(f"=== Results ZIP ready: {ZIP.name} ===", flush=True)
    except Exception:
        e=traceback.format_exc()
        ERR.write_text(e,encoding="utf-8")
        print(e,flush=True)

class H(SimpleHTTPRequestHandler):
    def __init__(self,*a,**kw): super().__init__(*a,directory=str(ROOT),**kw)
    def do_GET(self):
        if self.path in ("/","/index.html"):
            self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.end_headers()
            if ZIP.exists():
                links="".join(f'<li><a href="/{p.relative_to(ROOT).as_posix()}">{html.escape(p.name)}</a></li>' for p in sorted(RESULT.glob("*")) if p.is_file())
                body=f'<html><body style="font-family:sans-serif;max-width:950px;margin:50px auto"><h1>BTC Layered Breakout Study</h1><p><b><a href="/{ZIP.name}">Download all results (ZIP)</a></b></p><ul>{links}</ul></body></html>'
            elif ERR.exists():
                body=f"<html><body><pre>{html.escape(ERR.read_text())}</pre></body></html>"
            else: body="<html><body>No results yet.</body></html>"
            self.wfile.write(body.encode()); return
        super().do_GET()

run()
port=int(os.environ.get("PORT","8080"))
print(f"Serving on 0.0.0.0:{port}",flush=True)
ThreadingHTTPServer(("0.0.0.0",port),H).serve_forever()
