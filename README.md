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

To use a local Ollama model for an AI summary:

python fetch_reddit_daily_discussion_comments.py \
  --summary-only \
  --summary-ollama \
  --ollama-model llama3

To use the ChatGPT API for an AI summary:

export OPENAI_API_KEY="your-key-here"

python fetch_reddit_daily_discussion_comments.py \
  --summary-only \
  --summary-chatgpt \
  --openai-model gpt-4o-mini

## Discord channel messages

Use `fetch_reddit_daily_discussion_comments.py` with `--source discord` to read
messages from a Discord channel. You will need a bot token with access to the
channel.

Example:

export DISCORD_TOKEN="your-discord-bot-token"

python fetch_reddit_daily_discussion_comments.py \
  --source discord \
  --discord-channel-id 123456789012345678 \
  --output discord_messages.jsonl

To summarize Discord messages locally with Ollama:

python fetch_reddit_daily_discussion_comments.py \
  --source discord \
  --discord-channel-id 123456789012345678 \
  --summary-only \
  --summary-ollama

You can also pass a specific post URL or id if you already know it:

python fetch_reddit_daily_discussion_comments.py \
  --post-url "https://www.reddit.com/r/ethereum/comments/abcdef/daily_discussion/" \
  --format csv \
  --output daily_comments.csv
