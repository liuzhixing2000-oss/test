# BTC Breakout OOS Validation — Railway

This version locks the prior grid winner and does **no further parameter optimization**.

Locked setup:
- Volume >= 2.0x
- BB Width percentile <= 35%
- Previous 24h range <= 2%
- Stop = 1.0%
- Target = 4R
- Entry = RETEST

Validation:
- 2021-2024: historical development era
- 2025: OOS holdout
- 2026: OOS holdout
- 2025-2026 combined OOS
- 10,000 bootstrap resamples with 95% confidence intervals
- yearly and quarterly stability
- latest 7-day signal check, including exact signal and retest entry if one exists

## Railway Start Command

`python railway_oos_runner.py`

After deployment, generate/open the Railway public domain and download `btc-oos-results.zip`.

Please upload these files back to ChatGPT:
- btc_oos_summary.txt
- btc_oos_period_results.csv
- btc_oos_bootstrap.csv
- btc_oos_yearly.csv
- btc_oos_quarterly.csv
- btc_latest_signal_trades.csv
