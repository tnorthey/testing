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

## CoinGecko CSV fetch

Use `fetch_coingecko_prices.py` to download BTC/ETH hourly price data from
CoinGecko and write CSV files for testing.

Example:

python fetch_coingecko_prices.py \
  --days 30 \
  --output-dir data

This writes `data/btc.csv` and `data/eth.csv` by default. Use `--coins` to
specify other CoinGecko ids, `--use-coin-id-filenames` to use the ids as
filenames, or `--granularity raw` to keep the API's native resolution.
