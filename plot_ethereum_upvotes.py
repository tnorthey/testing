#!/usr/bin/env python3
"""
Plot Ethereum upvotes CSV (date_utc vs upvotes).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value.strip())


def read_upvotes(csv_path: Path) -> tuple[list[dt.date], list[int]]:
    dates: list[dt.date] = []
    upvotes: list[int] = []

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")
        if "date_utc" not in reader.fieldnames or "upvotes" not in reader.fieldnames:
            raise ValueError("CSV must include date_utc and upvotes columns.")

        for row in reader:
            date_raw = (row.get("date_utc") or "").strip()
            upvotes_raw = (row.get("upvotes") or "").strip()
            if not date_raw or not upvotes_raw:
                continue
            try:
                date_value = parse_date(date_raw)
                upvotes_value = int(upvotes_raw)
            except ValueError:
                continue
            dates.append(date_value)
            upvotes.append(upvotes_value)

    return dates, upvotes


def read_eth_prices(csv_path: Path) -> tuple[list[dt.datetime], list[float]]:
    timestamps: list[dt.datetime] = []
    prices: list[float] = []

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("ETH CSV has no header row.")
        if "timestamp" not in reader.fieldnames or "close" not in reader.fieldnames:
            raise ValueError("ETH CSV must include timestamp and close columns.")

        for row in reader:
            ts_raw = (row.get("timestamp") or "").strip()
            close_raw = (row.get("close") or "").strip()
            if not ts_raw or not close_raw:
                continue
            try:
                ts_value = int(ts_raw)
                close_value = float(close_raw)
            except ValueError:
                continue
            timestamps.append(
                dt.datetime.fromtimestamp(ts_value, tz=dt.timezone.utc)
            )
            prices.append(close_value)

    return timestamps, prices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot date vs upvotes from CSV.")
    parser.add_argument(
        "--input",
        default="ethereum_upvotes.csv",
        help="Input CSV path (default: ethereum_upvotes.csv).",
    )
    parser.add_argument(
        "--eth-csv",
        help="Optional ETH CSV with timestamp,close columns for y2 axis.",
    )
    parser.add_argument(
        "--output",
        help="Optional output image path (e.g., plot.png). If omitted, show window.",
    )
    parser.add_argument(
        "--title",
        default="Ethereum comment upvotes",
        help="Plot title.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    csv_path = Path(args.input).expanduser()
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    try:
        dates, upvotes = read_upvotes(csv_path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not dates:
        print("Error: No valid date/upvotes rows found in CSV.", file=sys.stderr)
        return 1

    eth_times: list[dt.datetime] = []
    eth_prices: list[float] = []
    if args.eth_csv:
        eth_path = Path(args.eth_csv).expanduser()
        if not eth_path.exists():
            print(f"Error: ETH CSV not found: {eth_path}", file=sys.stderr)
            return 1
        try:
            eth_times, eth_prices = read_eth_prices(eth_path)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required (pip install matplotlib).", file=sys.stderr)
        return 1

    upvote_times = [
        dt.datetime.combine(date, dt.time(12, 0), tzinfo=dt.timezone.utc) for date in dates
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(upvote_times, upvotes, marker="o", linewidth=1.5, color="#1f77b4")
    ax.set_title(args.title)
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Upvotes")
    ax.grid(True, alpha=0.3)

    if eth_times and eth_prices:
        ax2 = ax.twinx()
        ax2.plot(eth_times, eth_prices, color="#ff7f0e", linewidth=1.2, alpha=0.8)
        ax2.set_ylabel("ETH Close (USD)")

    fig.autofmt_xdate()

    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
