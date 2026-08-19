# BTC Expansion / Breakout Backtest

This repository tests BTCUSDT 1H "compression -> breakout -> expansion" setups.

## Run on GitHub

1. Upload all files in this bundle to the root of your GitHub repository.
2. Keep the `.github/workflows/btc_breakout_backtest.yml` path exactly as-is.
3. Open the **Actions** tab.
4. Select **BTC Expansion Breakout Backtest**.
5. Click **Run workflow**.
6. Wait for the workflow to finish.
7. Open the completed run and download the artifact named:
   **btc-breakout-results**

## Main outputs

- `btc_breakout_summary.txt`
- `btc_breakout_trades.csv`
- `btc_breakout_yearly.csv`
- `btc_breakout_event_stats.csv`
- `btc_breakout_threshold_grid.csv`

Upload those result files back to ChatGPT for analysis.

## Notes

- The script downloads BTCUSDT 1H data from Bybit's public API.
- No Bybit API key is required.
- The default start date is 2021-01-01.
- The parameter grid can take several minutes.
