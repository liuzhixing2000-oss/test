
import os
import traceback
import zipfile
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import btc_locked_strategy_validation as study

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "btc_locked_results"
ZIP = ROOT / "btc-locked-strategy-results.zip"

try:
    study.main()

    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for f in RESULT.glob("*"):
            if f.is_file():
                z.write(f, f.relative_to(ROOT))

    print(f"=== Results ZIP ready: {ZIP.name} ===", flush=True)

except Exception:
    print(traceback.format_exc(), flush=True)
    raise

port = int(os.environ.get("PORT", "8080"))
print(f"Serving on 0.0.0.0:{port}", flush=True)
ThreadingHTTPServer(("0.0.0.0", port), SimpleHTTPRequestHandler).serve_forever()
