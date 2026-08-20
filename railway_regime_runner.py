import os,zipfile,traceback
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
import btc_breakout_regime_study as s
try:
 s.main(); p=Path('btc_regime_results')
 with zipfile.ZipFile('btc-breakout-regime-results.zip','w',zipfile.ZIP_DEFLATED) as z:
  [z.write(f,f) for f in p.glob('*') if f.is_file()]
 print('=== Results ZIP ready: btc-breakout-regime-results.zip ===',flush=True)
except Exception:
 print(traceback.format_exc(),flush=True); raise
port=int(os.environ.get('PORT','8080')); print(f'Serving on 0.0.0.0:{port}',flush=True); ThreadingHTTPServer(('0.0.0.0',port),SimpleHTTPRequestHandler).serve_forever()
