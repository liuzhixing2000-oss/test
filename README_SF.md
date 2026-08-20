# BTC Breakout Success vs Failure Study — Railway

This study does NOT optimize the previous trading rule again.

It:
- extracts broad BTC 1H breakout events,
- labels them SUCCESS / FAILURE / NEUTRAL based on the following 24h,
- compares pre-event and event-time characteristics,
- separates 2021-2024 discovery from 2025-2026 OOS,
- reports which features retain the same direction of separation out-of-sample.

Railway Start Command:

python railway_sf_runner.py

Main outputs to send back to ChatGPT:
- btc_sf_summary.txt
- btc_sf_feature_stability.csv
- btc_sf_feature_comparison_discovery.csv
- btc_sf_feature_comparison_oos.csv
- btc_sf_yearly.csv
- btc_sf_recent_events.csv
- btc_sf_events.csv
