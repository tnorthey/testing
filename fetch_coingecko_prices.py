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
COINGLASS_API_BASE = "https://open-api.coinglass.com"
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


def parse_params(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    parsed = urllib.parse.parse_qs(value, keep_blank_values=True)
    return {key: values[-1] if values else "" for key, values in parsed.items()}


def build_coinglass_url(
    endpoint: str,
    params: dict[str, str],
    api_base: str,
) -> str:
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        base_url = endpoint
    else:
        endpoint = endpoint.lstrip("/")
        base_url = f"{api_base}/{endpoint}"
    if params:
        return f"{base_url}?{urllib.parse.urlencode(params)}"
    return base_url


def get_coinglass_api_key(value: str | None) -> str:
    api_key = value or os.getenv("COINGLASS_API_KEY", "")
    if not api_key:
        raise ValueError("Coinglass API key missing. Set COINGLASS_API_KEY or --coinglass-api-key.")
    return api_key.strip()


def fetch_coinglass_json(url: str, api_key: str, api_header: str, timeout: int) -> object:
    headers = {
        "User-Agent": USER_AGENT,
        api_header: api_key,
    }
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
        message = error_body.strip() or f"{exc.code} {exc.reason}"
        raise RuntimeError(f"Coinglass request failed: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Coinglass request failed: {exc.reason}") from exc
    return json.loads(payload)


def resolve_data_node(payload: object, data_key: str | None) -> object:
    if data_key:
        node: object = payload
        for part in data_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return None
        return node
    if isinstance(payload, dict):
        for key in ("data", "result", "list"):
            if key in payload:
                return payload[key]
    return payload


def select_value(data: dict[str, object], keys: Sequence[str]) -> object | None:
    for key in keys:
        if key in data:
            return data[key]
    return None


def parse_series_rows(
    payload: object,
    data_key: str | None,
    time_field: str,
    value_field: str,
) -> list[tuple[int, float]]:
    node = resolve_data_node(payload, data_key)
    if node is None:
        raise RuntimeError("Unable to locate data node in Coinglass response.")

    time_keys = (time_field, "timestamp", "time", "t", "date", "datetime")
    value_keys = (value_field, "value", "v", "close")
    rows: list[tuple[int, float]] = []

    if isinstance(node, dict):
        time_values = select_value(node, time_keys)
        value_values = select_value(node, value_keys)
        if isinstance(time_values, list) and isinstance(value_values, list):
            for raw_time, raw_value in zip(time_values, value_values):
                try:
                    ts_seconds = parse_timestamp(str(raw_time))
                    value = float(raw_value)
                except (ValueError, TypeError):
                    continue
                rows.append((ts_seconds, value))
            return rows

    if isinstance(node, list):
        for item in node:
            raw_time = raw_value = None
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                raw_time, raw_value = item[0], item[1]
            elif isinstance(item, dict):
                raw_time = select_value(item, time_keys)
                raw_value = select_value(item, value_keys)
            if raw_time is None or raw_value is None:
                continue
            try:
                ts_seconds = parse_timestamp(str(raw_time))
                value = float(raw_value)
            except (ValueError, TypeError):
                continue
            rows.append((ts_seconds, value))
        return rows

    raise RuntimeError("Unexpected data format from Coinglass response.")


def resolve_coinglass_output_path(metric: str, output_dir: str) -> str:
    safe_metric = metric.strip() or "coinglass_series"
    safe_metric = safe_metric.replace(" ", "_")
    return os.path.join(output_dir, f"{safe_metric}.csv")


def write_series_csv(rows: Iterable[tuple[int, float]], path: str) -> int:
    sorted_rows = sorted(rows, key=lambda item: item[0])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "value"])
        for ts_seconds, value in sorted_rows:
            writer.writerow([ts_seconds, value])
    return len(sorted_rows)


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
        description="Fetch price data from CoinGecko or Coinglass and write CSVs."
    )
    parser.add_argument(
        "--source",
        default="coingecko",
        choices=("coingecko", "coinglass"),
        help="Data source to fetch (default: coingecko).",
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
    parser.add_argument(
        "--coinglass-endpoint",
        help="Coinglass endpoint path or full URL.",
    )
    parser.add_argument(
        "--coinglass-metric",
        default="btc_2y_ma_multiplier",
        help="Metric name used for the output filename (default: btc_2y_ma_multiplier).",
    )
    parser.add_argument(
        "--coinglass-api-key",
        help="Coinglass API key (defaults to COINGLASS_API_KEY).",
    )
    parser.add_argument(
        "--coinglass-api-header",
        default="coinglassSecret",
        help="Header name for Coinglass API key (default: coinglassSecret).",
    )
    parser.add_argument(
        "--coinglass-params",
        help="Query params for Coinglass endpoint (key=value&key2=value2).",
    )
    parser.add_argument(
        "--coinglass-data-key",
        default="data",
        help="Dot-separated key to the data array in the response (default: data).",
    )
    parser.add_argument(
        "--coinglass-time-field",
        default="timestamp",
        help="Field name for timestamps within each data row (default: timestamp).",
    )
    parser.add_argument(
        "--coinglass-value-field",
        default="value",
        help="Field name for values within each data row (default: value).",
    )
    parser.add_argument(
        "--coinglass-output",
        help="Output CSV path for Coinglass data (defaults to output dir + metric).",
    )
    parser.add_argument(
        "--coinglass-timeout",
        type=int,
        default=30,
        help="Coinglass request timeout in seconds (default: 30).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.source == "coinglass":
        if not args.coinglass_endpoint:
            print("Error: --coinglass-endpoint is required for Coinglass source.", file=sys.stderr)
            return 1
        try:
            api_key = get_coinglass_api_key(args.coinglass_api_key)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        params = parse_params(args.coinglass_params)
        url = build_coinglass_url(args.coinglass_endpoint, params, COINGLASS_API_BASE)
        try:
            payload = fetch_coinglass_json(
                url=url,
                api_key=api_key,
                api_header=args.coinglass_api_header,
                timeout=args.coinglass_timeout,
            )
            data_key = args.coinglass_data_key or None
            rows = parse_series_rows(
                payload=payload,
                data_key=data_key,
                time_field=args.coinglass_time_field,
                value_field=args.coinglass_value_field,
            )
        except RuntimeError as exc:
            print(f"Error fetching Coinglass data: {exc}", file=sys.stderr)
            return 1
        if not rows:
            print("Error: No usable data rows returned from Coinglass.", file=sys.stderr)
            return 1
        output_path = args.coinglass_output or resolve_coinglass_output_path(
            args.coinglass_metric, args.output_dir
        )
        row_count = write_series_csv(rows, output_path)
        print(f"Wrote {row_count} rows to {output_path}")
        return 0

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
