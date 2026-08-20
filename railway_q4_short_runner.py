import os,zipfile,traceback
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
import btc_q4_short_validation as s
ROOT=Path(__file__).resolve().parent; RESULT=ROOT/'btc_q4_short_results'; ZIP=ROOT/'btc-q4-short-results.zip'
try:
 s.main()
 with zipfile.ZipFile(ZIP,'w',zipfile.ZIP_DEFLATED) as z:
  for f in RESULT.glob('*'):
   if f.is_file(): z.write(f,f.relative_to(ROOT))
 print(f'=== Results ZIP ready: {ZIP.name} ===',flush=True)
except Exception:
 print(traceback.format_exc(),flush=True); raise
port=int(os.environ.get('PORT','8080')); print(f'Serving on 0.0.0.0:{port}',flush=True); ThreadingHTTPServer(('0.0.0.0',port),SimpleHTTPRequestHandler).serve_forever()
