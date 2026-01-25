#!/bin/bash

CACHE="$HOME/Nextcloud2/reddit_cache/ethereum_daily"

python fetch_reddit_daily_discussion_comments.py \
  --date 2025-12-09 \
  --cache-dir $CACHE \
  --refresh-cache \
  --ethereum-upvotes-csv ethereum_upvotes.csv
  
#--past-days 60 \
