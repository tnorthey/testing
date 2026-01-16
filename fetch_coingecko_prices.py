#!/usr/bin/env python3
"""
Fetch historical price data from CoinGecko and write CSVs.

CoinGecko API docs: https://www.coingecko.com/en/api/documentation
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
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


def write_prices_csv(prices: Iterable[Sequence[float]], path: str) -> int:
    rows = []
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

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "close"])
        for ts_seconds, price_value in rows:
            writer.writerow([ts_seconds, price_value])

    return len(rows)


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
    days: float | None = args.days
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
        days = None

    os.makedirs(args.output_dir, exist_ok=True)

    for coin_id in coin_ids:
        url = build_url(coin_id, args.vs_currency, days, start, end)
        try:
            chart = fetch_market_chart(url)
        except RuntimeError as exc:
            print(f"Error fetching {coin_id}: {exc}", file=sys.stderr)
            return 1
        output_path = resolve_output_path(
            coin_id=coin_id,
            output_dir=args.output_dir,
            use_symbol=not args.use_coin_id_filenames,
        )
        row_count = write_prices_csv(chart.prices, output_path)
        print(f"Wrote {row_count} rows to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
