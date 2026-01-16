#!/usr/bin/env python3
"""
Fetch historical price data from CoinGecko and write CSVs (hourly by default).

CoinGecko API docs: https://www.coingecko.com/en/api/documentation
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Sequence


COINGECKO_API_BASE = "https://api.coingecko.com/api/v3"
USER_AGENT = "crypto-price-correlation-script"
SYMBOL_FILENAMES = {
    "bitcoin": "btc",
    "ethereum": "eth",
}
SECONDS_PER_DAY = 86400
SECONDS_PER_HOUR = 3600
MAX_RANGE_DAYS = 90


@dataclass(frozen=True)
class MarketChart:
    prices: Sequence[Sequence[float]]


def parse_timestamp(value: str) -> int:
    value = value.strip()
    if not value:
        raise ValueError("Empty timestamp value")

    try:
        numeric = float(value)
    except ValueError:
        numeric = None

    if numeric is not None:
        if numeric > 1e12:
            return int(numeric / 1000)
        return int(numeric)

    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        parsed = None

    if parsed is None:
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                parsed = dt.datetime.strptime(value, fmt)
                break
            except ValueError:
                continue

    if parsed is None:
        raise ValueError(f"Unsupported timestamp format: {value}")

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)

    return int(parsed.timestamp())


def parse_coin_list(value: str) -> list[str]:
    coins = [item.strip() for item in value.split(",") if item.strip()]
    if not coins:
        raise ValueError("No coin ids provided")
    return coins


def build_url(coin_id: str, vs_currency: str, days: float | None, start: int | None, end: int | None) -> str:
    if days is None:
        if start is None or end is None:
            raise ValueError("Both start and end are required for range fetch.")
        path = f"/coins/{coin_id}/market_chart/range"
        params = {"vs_currency": vs_currency, "from": str(start), "to": str(end)}
    else:
        path = f"/coins/{coin_id}/market_chart"
        params = {"vs_currency": vs_currency, "days": str(days)}

    return f"{COINGECKO_API_BASE}{path}?{urllib.parse.urlencode(params)}"


def fetch_market_chart(url: str) -> MarketChart:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"CoinGecko request failed: {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"CoinGecko request failed: {exc.reason}") from exc

    data = json.loads(payload)
    prices = data.get("prices")
    if not isinstance(prices, list):
        raise RuntimeError("Unexpected response from CoinGecko (missing prices array).")
    return MarketChart(prices=prices)


def resolve_output_path(coin_id: str, output_dir: str, use_symbol: bool) -> str:
    filename = SYMBOL_FILENAMES.get(coin_id, coin_id) if use_symbol else coin_id
    return os.path.join(output_dir, f"{filename}.csv")


def normalize_prices(prices: Iterable[Sequence[float]]) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    for entry in prices:
        if len(entry) < 2:
            continue
        timestamp_ms, price = entry[0], entry[1]
        try:
            ts_seconds = int(float(timestamp_ms) / 1000)
            price_value = float(price)
        except (TypeError, ValueError):
            continue
        rows.append((ts_seconds, price_value))

    rows.sort(key=lambda item: item[0])

    deduped: list[tuple[int, float]] = []
    for ts_seconds, price_value in rows:
        if deduped and deduped[-1][0] == ts_seconds:
            deduped[-1] = (ts_seconds, price_value)
        else:
            deduped.append((ts_seconds, price_value))
    return deduped


def downsample_to_hourly(rows: Sequence[tuple[int, float]]) -> list[tuple[int, float]]:
    hourly: list[tuple[int, float]] = []
    current_bucket: int | None = None
    for ts_seconds, price_value in rows:
        bucket = ts_seconds - (ts_seconds % SECONDS_PER_HOUR)
        if bucket != current_bucket:
            hourly.append((bucket, price_value))
            current_bucket = bucket
        else:
            hourly[-1] = (bucket, price_value)
    return hourly


def fetch_prices_range_chunked(
    coin_id: str,
    vs_currency: str,
    start: int,
    end: int,
    chunk_days: int,
) -> list[Sequence[float]]:
    if start >= end:
        raise ValueError("Start must be before end.")
    chunk_seconds = chunk_days * SECONDS_PER_DAY
    prices: list[Sequence[float]] = []
    current_start = start
    while current_start < end:
        current_end = min(current_start + chunk_seconds, end)
        url = build_url(coin_id, vs_currency, None, current_start, current_end)
        chart = fetch_market_chart(url)
        prices.extend(chart.prices)
        current_start = current_end
    return prices


def resolve_range(
    days: float | None,
    start: int | None,
    end: int | None,
) -> tuple[int, int]:
    if start is not None or end is not None:
        if start is None or end is None:
            raise ValueError("Both start and end are required for range fetch.")
        if start >= end:
            raise ValueError("Start must be before end.")
        return start, end
    if days is None:
        raise ValueError("Days must be provided when start/end not supplied.")
    end_ts = int(time.time())
    start_ts = end_ts - int(days * SECONDS_PER_DAY)
    if start_ts >= end_ts:
        raise ValueError("Days must be positive.")
    return start_ts, end_ts


def fetch_prices(
    coin_id: str,
    vs_currency: str,
    days: float | None,
    start: int | None,
    end: int | None,
    granularity: str,
) -> list[tuple[int, float]]:
    if granularity == "hourly":
        start_ts, end_ts = resolve_range(days, start, end)
        raw_prices = fetch_prices_range_chunked(
            coin_id=coin_id,
            vs_currency=vs_currency,
            start=start_ts,
            end=end_ts,
            chunk_days=MAX_RANGE_DAYS,
        )
        rows = normalize_prices(raw_prices)
        return downsample_to_hourly(rows)

    if start is not None or end is not None:
        if start is None or end is None:
            raise ValueError("Both start and end are required for range fetch.")
        url = build_url(coin_id, vs_currency, None, start, end)
    else:
        if days is None:
            raise ValueError("Days must be provided when start/end not supplied.")
        url = build_url(coin_id, vs_currency, days, None, None)

    chart = fetch_market_chart(url)
    return normalize_prices(chart.prices)


def write_prices_csv(rows: Iterable[tuple[int, float]], path: str) -> int:
    sorted_rows = sorted(rows, key=lambda item: item[0])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "close"])
        for ts_seconds, price_value in sorted_rows:
            writer.writerow([ts_seconds, price_value])

    return len(sorted_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch price data from CoinGecko and write CSVs."
    )
    parser.add_argument(
        "--coins",
        default="bitcoin,ethereum",
        help="Comma-separated CoinGecko coin ids (default: bitcoin,ethereum).",
    )
    parser.add_argument(
        "--vs-currency",
        default="usd",
        help="Fiat or crypto quote currency (default: usd).",
    )
    parser.add_argument(
        "--days",
        type=float,
        default=7.0,
        help="Number of days back from now (default: 7).",
    )
    parser.add_argument(
        "--start",
        help="Range start timestamp (epoch or ISO). Requires --end.",
    )
    parser.add_argument(
        "--end",
        help="Range end timestamp (epoch or ISO). Requires --start.",
    )
    parser.add_argument(
        "--granularity",
        choices=("hourly", "raw"),
        default="hourly",
        help="Output granularity: hourly closes or raw API data (default: hourly).",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for CSVs (default: data).",
    )
    parser.add_argument(
        "--use-coin-id-filenames",
        action="store_true",
        help="Use CoinGecko ids for filenames instead of symbols.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        coin_ids = parse_coin_list(args.coins)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    start = end = None
    if args.start or args.end:
        if not args.start or not args.end:
            print("Error: --start and --end must be provided together.", file=sys.stderr)
            return 1
        try:
            start = parse_timestamp(args.start)
            end = parse_timestamp(args.end)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    os.makedirs(args.output_dir, exist_ok=True)

    for coin_id in coin_ids:
        try:
            rows = fetch_prices(
                coin_id=coin_id,
                vs_currency=args.vs_currency,
                days=args.days,
                start=start,
                end=end,
                granularity=args.granularity,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"Error fetching {coin_id}: {exc}", file=sys.stderr)
            return 1
        output_path = resolve_output_path(
            coin_id=coin_id,
            output_dir=args.output_dir,
            use_symbol=not args.use_coin_id_filenames,
        )
        row_count = write_prices_csv(rows, output_path)
        print(f"Wrote {row_count} rows to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
