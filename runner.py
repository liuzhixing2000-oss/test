import os,zipfile
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
import btc_ema200_response_curve as s
s.main()
with zipfile.ZipFile("btc-ema200-response-results.zip","w",zipfile.ZIP_DEFLATED) as z:
 [z.write(f,f) for f in Path("results").glob("*")]
print("=== Results ZIP ready: btc-ema200-response-results.zip ===",flush=True)
ThreadingHTTPServer(("0.0.0.0",int(os.environ.get("PORT","8080"))),SimpleHTTPRequestHandler).serve_forever()
