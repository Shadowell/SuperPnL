#!/usr/bin/env python3
"""Download OKX spot 1m candles for SuperPnL.

This script intentionally uses only Python's standard library so it can run on
minimal servers without pandas/pyarrow installed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import gzip
import json
import os
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path


BASE_URL = "https://www.okx.com"
BAR_MS = 60_000
STABLE_BASES = {
    "USDT",
    "USDC",
    "DAI",
    "USDG",
    "PYUSD",
    "TUSD",
    "FDUSD",
    "EURT",
    "RLUSD",
    "USDE",
    "USD1",
}
RATE_LIMITER: "RateLimiter | None" = None


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            if now < self.next_at:
                time.sleep(self.next_at - now)
                now = time.monotonic()
            self.next_at = now + self.min_interval


def utc_now_ms() -> int:
    return int(time.time() * 1000)


def ms_to_utc(ms: int | str | None) -> str | None:
    if ms in (None, ""):
        return None
    return dt.datetime.fromtimestamp(int(ms) / 1000, tz=dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def request_json(path: str, params: dict[str, str | int], max_retries: int = 6) -> dict:
    if RATE_LIMITER is not None:
        RATE_LIMITER.wait()
    query = urllib.parse.urlencode(params)
    url = f"{BASE_URL}{path}?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "SuperPnL-data/0.1"})
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("code") == "0":
                return payload
            raise RuntimeError(f"OKX code={payload.get('code')} msg={payload.get('msg')}")
        except Exception as exc:  # noqa: BLE001 - retry network and OKX throttle errors.
            last_error = exc
            sleep_s = min(20.0, 0.8 * (2**attempt))
            time.sleep(sleep_s)
    raise RuntimeError(f"request failed after retries: {url}") from last_error


def as_float(value: str | None) -> float:
    try:
        return float(value or 0)
    except ValueError:
        return 0.0


def get_live_usdt_spot_universe() -> tuple[dict[str, dict], list[dict]]:
    instruments = request_json("/api/v5/public/instruments", {"instType": "SPOT"})["data"]
    live = {
        item["instId"]: item
        for item in instruments
        if item.get("state") == "live" and item.get("instId", "").endswith("-USDT")
    }
    tickers = request_json("/api/v5/market/tickers", {"instType": "SPOT"})["data"]
    candidates = []
    for ticker in tickers:
        inst_id = ticker.get("instId")
        if inst_id not in live:
            continue
        base = inst_id.split("-")[0]
        if base in STABLE_BASES:
            continue
        candidates.append(ticker)
    candidates.sort(key=lambda item: as_float(item.get("volCcy24h")), reverse=True)
    return live, candidates


def has_history_before(inst_id: str, timestamp_ms: int) -> bool:
    payload = request_json(
        "/api/v5/market/history-candles",
        {"instId": inst_id, "bar": "1m", "limit": "1", "after": str(timestamp_ms)},
    )
    return any(len(row) >= 9 and row[8] == "1" for row in payload.get("data") or [])


def select_top_symbols(top_n: int, start_ms: int) -> list[str]:
    live, candidates = get_live_usdt_spot_universe()
    selected: list[str] = []
    for ticker in candidates:
        inst_id = ticker["instId"]
        list_time = int(live[inst_id].get("listTime") or 0)
        if list_time and list_time > start_ms:
            continue
        if not has_history_before(inst_id, start_ms + BAR_MS):
            continue
        selected.append(inst_id)
        if len(selected) >= top_n:
            break
        time.sleep(0.05)
    return selected


def fetch_symbol_rows(inst_id: str, start_ms: int, end_ms: int, sleep_s: float) -> list[list]:
    rows_by_ts: dict[int, list] = {}
    cursor: int | None = None
    pages = 0
    while True:
        params = {"instId": inst_id, "bar": "1m", "limit": "300"}
        if cursor is not None:
            params["after"] = str(cursor)
        payload = request_json("/api/v5/market/history-candles", params)
        raw_rows = payload.get("data") or []
        confirmed = [row for row in raw_rows if len(row) >= 9 and row[8] == "1"]
        if not confirmed:
            break
        pages += 1
        min_ts = min(int(row[0]) for row in confirmed)
        for row in confirmed:
            ts = int(row[0])
            if start_ms <= ts <= end_ms:
                rows_by_ts[ts] = row
        if min_ts < start_ms:
            break
        if cursor == min_ts:
            break
        cursor = min_ts
        if sleep_s:
            time.sleep(sleep_s)
    rows = [rows_by_ts[ts] for ts in sorted(rows_by_ts)]
    print(
        f"{inst_id}: pages={pages} rows={len(rows)} "
        f"start={ms_to_utc(rows[0][0]) if rows else None} "
        f"end={ms_to_utc(rows[-1][0]) if rows else None}",
        flush=True,
    )
    return rows


def write_symbol_csv_gz(path: Path, rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        with gzip.open(tmp_name, "wt", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "volume_ccy",
                    "amount",
                    "confirm",
                ]
            )
            for row in rows:
                writer.writerow(row[:9])
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def download_one_symbol(
    index: int,
    total: int,
    symbol: str,
    csv_dir: Path,
    start_ms: int,
    end_ms: int,
    sleep_s: float,
    force: bool,
) -> dict:
    path = csv_dir / f"{symbol}.csv.gz"
    if path.exists() and not force:
        print(f"[{index}/{total}] skip existing {symbol}", flush=True)
        return {"symbol": symbol, "path": str(path), "skipped": True}

    print(f"[{index}/{total}] downloading {symbol}", flush=True)
    rows = fetch_symbol_rows(symbol, start_ms, end_ms, sleep_s)
    expected = (end_ms - start_ms) // BAR_MS + 1
    coverage = len(rows) / expected if expected else 0
    write_symbol_csv_gz(path, rows)
    return {
        "symbol": symbol,
        "path": str(path),
        "rows": len(rows),
        "coverage": coverage,
        "first_utc": ms_to_utc(rows[0][0]) if rows else None,
        "last_utc": ms_to_utc(rows[-1][0]) if rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--sleep", type=float, default=0.06)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rate", type=float, default=8.0, help="global OKX requests per second")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    global RATE_LIMITER
    RATE_LIMITER = RateLimiter(args.rate)

    out = Path(args.out)
    csv_dir = out / "csv"
    out.mkdir(parents=True, exist_ok=True)
    end_ms = (utc_now_ms() // BAR_MS - 1) * BAR_MS
    start_ms = end_ms - args.days * 24 * 60 * BAR_MS

    print("SuperPnL OKX 1m download", flush=True)
    print(f"out={out}", flush=True)
    print(f"window={ms_to_utc(start_ms)} -> {ms_to_utc(end_ms)}", flush=True)
    print(
        f"top={args.top} days={args.days} workers={args.workers} rate={args.rate}/s",
        flush=True,
    )

    symbols = select_top_symbols(args.top, start_ms)
    if len(symbols) < args.top:
        raise RuntimeError(f"only selected {len(symbols)} symbols: {symbols}")
    print("symbols=" + ",".join(symbols), flush=True)

    metadata = {
        "source": "OKX public API",
        "exchange": "okx",
        "instrument_type": "SPOT",
        "bar_size": "1m",
        "top_n": args.top,
        "days": args.days,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "start_utc": ms_to_utc(start_ms),
        "end_utc": ms_to_utc(end_ms),
        "symbols": symbols,
        "files": [],
        "candle_fields": [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "volume_ccy",
            "amount",
            "confirm",
        ],
        "notes": "amount maps to OKX candle volCcyQuote for USDT pairs.",
    }

    if args.workers <= 1:
        for index, symbol in enumerate(symbols, 1):
            metadata["files"].append(
                download_one_symbol(
                    index,
                    len(symbols),
                    symbol,
                    csv_dir,
                    start_ms,
                    end_ms,
                    args.sleep,
                    args.force,
                )
            )
    else:
        results: dict[str, dict] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_symbol = {
                executor.submit(
                    download_one_symbol,
                    index,
                    len(symbols),
                    symbol,
                    csv_dir,
                    start_ms,
                    end_ms,
                    args.sleep,
                    args.force,
                ): symbol
                for index, symbol in enumerate(symbols, 1)
            }
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                results[symbol] = future.result()
                print(f"completed {symbol}", flush=True)
        metadata["files"].extend(results[symbol] for symbol in symbols if symbol in results)

    metadata_path = out / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(f"metadata={metadata_path}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
