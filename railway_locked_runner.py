import os,zipfile,traceback
from pathlib import Path
import btc_locked_strategy_validation as s
try:
 s.main();p=Path("btc_locked_results")
 with zipfile.ZipFile("btc-locked-strategy-results.zip","w",zipfile.ZIP_DEFLATED) as z:
  [z.write(f,f) for f in p.glob("*") if f.is_file()]
 print("=== Results ZIP ready ===",flush=True)
except Exception:
 print(traceback.format_exc(),flush=True);raise
# keep Railway service alive so results remain in deployment logs/files
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
port=int(os.environ.get("PORT","8080"));ThreadingHTTPServer(("0.0.0.0",port),SimpleHTTPRequestHandler).serve_forever()
