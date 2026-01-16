#!/usr/bin/env python3
"""
Analyze BTC moves and whether ETH follows within a window.

Expected CSV columns include a timestamp and a price column. Use CLI
flags to match your dataset's column names and formats.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from bisect import bisect_left
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PriceSeries:
    times: Tuple[int, ...]
    prices: Tuple[float, ...]


@dataclass(frozen=True)
class MoveEvent:
    start_ts: int
    end_ts: int
    btc_change_pct: float
    eth_change_pct: Optional[float]
    eth_followed: Optional[bool]


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
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M",
        ):
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


def load_series(
    path: str,
    time_column: str,
    price_column: str,
    delimiter: str,
) -> PriceSeries:
    times: list[int] = []
    prices: list[float] = []

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"No header row found in {path}")
        if time_column not in reader.fieldnames:
            raise ValueError(f"Missing time column '{time_column}' in {path}")
        if price_column not in reader.fieldnames:
            raise ValueError(f"Missing price column '{price_column}' in {path}")

        for row in reader:
            raw_time = row.get(time_column, "").strip()
            raw_price = row.get(price_column, "").strip()
            if not raw_time or not raw_price:
                continue
            try:
                ts = parse_timestamp(raw_time)
                price = float(raw_price)
            except ValueError:
                continue
            times.append(ts)
            prices.append(price)

    if not times:
        raise ValueError(f"No valid rows found in {path}")

    combined = sorted(zip(times, prices))
    sorted_times, sorted_prices = zip(*combined)
    return PriceSeries(tuple(sorted_times), tuple(sorted_prices))


def closest_price(
    series: PriceSeries,
    target_ts: int,
    max_gap_seconds: int,
) -> Optional[float]:
    times = series.times
    prices = series.prices
    idx = bisect_left(times, target_ts)
    candidates: list[tuple[int, float]] = []
    if idx < len(times):
        candidates.append((abs(times[idx] - target_ts), prices[idx]))
    if idx > 0:
        candidates.append((abs(times[idx - 1] - target_ts), prices[idx - 1]))
    if not candidates:
        return None
    gap, price = min(candidates, key=lambda item: item[0])
    if gap <= max_gap_seconds:
        return price
    return None


def compute_events(
    btc: PriceSeries,
    eth: PriceSeries,
    window_seconds: int,
    threshold_pct: float,
    eth_threshold_pct: float,
    max_gap_seconds: int,
) -> Sequence[MoveEvent]:
    events: list[MoveEvent] = []
    for start_ts, btc_start in zip(btc.times, btc.prices):
        end_ts = start_ts + window_seconds
        btc_end = closest_price(btc, end_ts, max_gap_seconds)
        if btc_end is None:
            continue
        btc_change_pct = (btc_end - btc_start) / btc_start * 100
        if abs(btc_change_pct) < threshold_pct:
            continue

        eth_start = closest_price(eth, start_ts, max_gap_seconds)
        eth_end = closest_price(eth, end_ts, max_gap_seconds)
        eth_change_pct: Optional[float] = None
        eth_followed: Optional[bool] = None
        if eth_start is not None and eth_end is not None:
            eth_change_pct = (eth_end - eth_start) / eth_start * 100
            same_direction = (btc_change_pct >= 0 and eth_change_pct >= 0) or (
                btc_change_pct < 0 and eth_change_pct < 0
            )
            eth_followed = same_direction and abs(eth_change_pct) >= eth_threshold_pct

        events.append(
            MoveEvent(
                start_ts=start_ts,
                end_ts=end_ts,
                btc_change_pct=btc_change_pct,
                eth_change_pct=eth_change_pct,
                eth_followed=eth_followed,
            )
        )
    return events


def format_ts(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).isoformat()


def write_csv(events: Iterable[MoveEvent], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "start_time_utc",
                "end_time_utc",
                "btc_change_pct",
                "eth_change_pct",
                "eth_followed",
            ]
        )
        for event in events:
            writer.writerow(
                [
                    format_ts(event.start_ts),
                    format_ts(event.end_ts),
                    f"{event.btc_change_pct:.4f}",
                    "" if event.eth_change_pct is None else f"{event.eth_change_pct:.4f}",
                    "" if event.eth_followed is None else str(event.eth_followed),
                ]
            )


def report_summary(events: Sequence[MoveEvent]) -> None:
    total = len(events)
    followed = sum(1 for event in events if event.eth_followed is True)
    not_followed = sum(1 for event in events if event.eth_followed is False)
    unknown = total - followed - not_followed
    follow_rate = (followed / total * 100) if total else 0.0

    print("BTC move events:", total)
    print("ETH followed:", followed)
    print("ETH did not follow:", not_followed)
    print("ETH unknown:", unknown)
    print(f"ETH follow rate: {follow_rate:.2f}%")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find BTC moves and check if ETH follows within a window."
    )
    parser.add_argument("--btc", required=True, help="BTC CSV path")
    parser.add_argument("--eth", required=True, help="ETH CSV path")
    parser.add_argument(
        "--window-minutes",
        type=float,
        default=60.0,
        help="Window size in minutes (default: 60).",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=1.0,
        help="BTC percent change threshold (default: 1.0).",
    )
    parser.add_argument(
        "--eth-threshold-pct",
        type=float,
        default=0.0,
        help="ETH percent change threshold (default: 0.0).",
    )
    parser.add_argument(
        "--max-gap-minutes",
        type=float,
        default=5.0,
        help="Max timestamp gap allowed for matching (default: 5).",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="CSV delimiter (default: ',').",
    )
    parser.add_argument(
        "--btc-time-column",
        default="timestamp",
        help="BTC time column name (default: timestamp).",
    )
    parser.add_argument(
        "--btc-price-column",
        default="close",
        help="BTC price column name (default: close).",
    )
    parser.add_argument(
        "--eth-time-column",
        default="timestamp",
        help="ETH time column name (default: timestamp).",
    )
    parser.add_argument(
        "--eth-price-column",
        default="close",
        help="ETH price column name (default: close).",
    )
    parser.add_argument(
        "--output-csv",
        help="Write detailed events to CSV.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    btc = load_series(
        path=args.btc,
        time_column=args.btc_time_column,
        price_column=args.btc_price_column,
        delimiter=args.delimiter,
    )
    eth = load_series(
        path=args.eth,
        time_column=args.eth_time_column,
        price_column=args.eth_price_column,
        delimiter=args.delimiter,
    )

    window_seconds = int(args.window_minutes * 60)
    max_gap_seconds = int(args.max_gap_minutes * 60)

    events = compute_events(
        btc=btc,
        eth=eth,
        window_seconds=window_seconds,
        threshold_pct=args.threshold_pct,
        eth_threshold_pct=args.eth_threshold_pct,
        max_gap_seconds=max_gap_seconds,
    )

    report_summary(events)

    if args.output_csv:
        write_csv(events, args.output_csv)
        print(f"Wrote event details to {args.output_csv}")
    else:
        print("start_time_utc,end_time_utc,btc_change_pct,eth_change_pct,eth_followed")
        for event in events:
            eth_change = "" if event.eth_change_pct is None else f"{event.eth_change_pct:.4f}"
            eth_followed = "" if event.eth_followed is None else str(event.eth_followed)
            print(
                f"{format_ts(event.start_ts)},"
                f"{format_ts(event.end_ts)},"
                f"{event.btc_change_pct:.4f},"
                f"{eth_change},"
                f"{eth_followed}"
            )


if __name__ == "__main__":
    main()
