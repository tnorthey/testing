# tnorthey/testing

This repository is used for testing purposes.

## About

A minimal README created on 2026-01-16.

## BTC/ETH move correlation script

Use `crypto_price_move_correlation.py` to find BTC moves over a window and check
whether ETH moved in the same direction during that window.

Example:

python crypto_price_move_correlation.py \
  --btc data/btc.csv \
  --eth data/eth.csv \
  --window-minutes 60 \
  --threshold-pct 1.0 \
  --output-csv btc_eth_moves.csv

The script expects CSV files with timestamp and price columns. Use the column
flags (`--btc-time-column`, `--btc-price-column`, etc.) to match your data.
