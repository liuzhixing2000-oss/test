# BTC Expansion / Breakout Backtest — Railway Version

Upload these files to the GitHub repository root:

- `btc_expansion_breakout_backtest.py`
- `railway_runner.py`
- `requirements.txt`
- `railway.json`

## Deploy

1. Push the files to GitHub.
2. In Railway, create/connect a service to that GitHub repository.
3. Deploy the service.
4. The service runs the backtest once at startup.
5. In Railway service settings, go to **Networking → Public Networking → Generate Domain**.
6. Open the generated Railway domain.

When successful, the page will show **Download all results (ZIP)** plus individual result files.

If Bybit is blocked from the Railway egress IP too, the service will stay online and show the full error plus `run_error.txt` instead of crashing.
