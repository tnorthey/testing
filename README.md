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

Use `fetch_coingecko_prices.py` to download BTC/ETH price data from CoinGecko
and write CSV files for testing.

Example:

python fetch_coingecko_prices.py \
  --days 30 \
  --output-dir data

This writes `data/btc.csv` and `data/eth.csv` by default. Use `--coins` to
specify other CoinGecko ids, or `--use-coin-id-filenames` to use the ids as
filenames.

## Reddit daily discussion comments

Use `fetch_reddit_daily_discussion_comments.py` to locate the latest Ethereum
daily discussion thread and export its comments.

Example:

python fetch_reddit_daily_discussion_comments.py \
  --output daily_comments.jsonl

To print a quick summary without writing comment data:

python fetch_reddit_daily_discussion_comments.py \
  --summary-only

To fetch a specific past daily discussion (UTC date) and summarize it:

python fetch_reddit_daily_discussion_comments.py \
  --date 2026-01-10 \
  --summary-only

To cache fetched JSON locally (and reuse it on subsequent runs):

python fetch_reddit_daily_discussion_comments.py \
  --date 2026-01-10 \
  --summary-only \
  --cache-dir reddit_cache

To force a refetch even if cached data exists:

python fetch_reddit_daily_discussion_comments.py \
  --date 2026-01-10 \
  --summary-only \
  --cache-dir reddit_cache \
  --refresh-cache

To fetch the daily discussion from N days ago (UTC) and summarize it:

python fetch_reddit_daily_discussion_comments.py \
  --days-ago 7 \
  --summary-only

To summarize the last N days of daily discussions (UTC):

python fetch_reddit_daily_discussion_comments.py \
  --past-days 14 \
  --summary-only \
  --cache-dir reddit_cache

To build a CSV of `date_utc -> upvotes` for the best *comment* whose body is exactly
`Ethereum` / `Ethereum!` / `Ethereum.` (case-insensitive) per day (uses your cached
`ethereum_post_*.json` files):

python fetch_reddit_daily_discussion_comments.py \
  --past-days 30 \
  --cache-dir reddit_cache \
  --ethereum-upvotes-csv ethereum_upvotes.csv

You can also pass a specific post URL or id if you already know it:

python fetch_reddit_daily_discussion_comments.py \
  --post-url "https://www.reddit.com/r/ethereum/comments/abcdef/daily_discussion/" \
  --format csv \
  --output daily_comments.csv
