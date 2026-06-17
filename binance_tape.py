"""
Binance free historical aggTrades -> real per-instrument order-flow bars.

Downloads monthly (or daily) aggTrades flat files from data.binance.vision
(USDT-M futures) through an HTTP proxy and STREAM-aggregates them into bars that
strategy.py / wyckoff.py can read directly, plus a footprint sidecar:

    <SYM>_<bar>.csv       timestamp,open,high,low,close,volume,buy_vol,sell_vol,delta,cvd,trades
    <SYM>_<bar>_fp.jsonl  {"ts":.., "levels":{price_bin:[buy,sell], ...}}

aggTrades columns (um futures):
    agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker
`is_buyer_maker == true` => the taker was the SELLER (taker-sell); else taker-buy.
So this is REAL per-instrument aggressor flow — the thing the OKX ccy-level data
lacked. From it we get delta, CVD, and volume-by-price footprint for Wyckoff.

The tape is large (BTC ~hundreds of MB/month, billions of trades), so files are
streamed line-by-line and deleted after processing; CVD runs continuously across
months. This is the cheapest path to a real Gate-0 sample (no vendor needed).

Usage:
  # smoke test on a single day
  py binance_tape.py --proxy http://127.0.0.1:7890 --symbols BTCUSDT --bar 15m \
      --daily 2026-01-15 --out data_binance
  # full pull (months range, multiple symbols) — run it long, like the recorder
  py binance_tape.py --proxy http://127.0.0.1:7890 \
      --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT DOGEUSDT --bar 15m \
      --start 2025-01 --end 2025-12 --out data_binance
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile

BASE = "https://data.binance.vision/data/futures/um"
COLS = ["timestamp", "open", "high", "low", "close", "volume",
        "buy_vol", "sell_vol", "delta", "cvd", "trades"]
BAR_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
          "30m": 1_800_000, "1h": 3_600_000}


class BarAgg:
    def __init__(self, bar_ms, cvd0=0.0):
        self.bar_ms = bar_ms
        self.cvd = cvd0
        self.bucket = None
        self.o = self.h = self.l = self.c = None
        self.vol = self.buy = self.sell = 0.0
        self.n = 0
        self.levels = {}

    def _start(self, bucket, px):
        self.bucket = bucket
        self.o = self.h = self.l = self.c = px
        self.vol = self.buy = self.sell = 0.0
        self.n = 0
        self.levels = {}

    def add(self, ts, px, qty, taker_buy):
        bucket = (ts // self.bar_ms) * self.bar_ms
        finalized = None
        if self.bucket is None:
            self._start(bucket, px)
        elif bucket > self.bucket:
            finalized = self._finalize()
            self._start(bucket, px)
        elif bucket < self.bucket:
            return None                       # out-of-order (shouldn't happen)
        self.c = px
        if px > self.h: self.h = px
        if px < self.l: self.l = px
        self.vol += qty
        self.n += 1
        if taker_buy:
            self.buy += qty
        else:
            self.sell += qty
        step = 10.0 ** (math.floor(math.log10(px)) - 3) if px > 0 else 1.0
        binp = round(round(px / step) * step, 12)
        lvl = self.levels.setdefault(binp, [0.0, 0.0])
        lvl[0 if taker_buy else 1] += qty
        return finalized

    def _finalize(self):
        delta = self.buy - self.sell
        self.cvd += delta
        return {"timestamp": self.bucket, "open": self.o, "high": self.h,
                "low": self.l, "close": self.c, "volume": self.vol,
                "buy_vol": self.buy, "sell_vol": self.sell, "delta": delta,
                "cvd": self.cvd, "trades": self.n, "_levels": self.levels}


def months_range(start, end):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def download(url, dest, opener, retries=5):
    """Stream a (large) file to dest. Returns True, or False on 404."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with opener.open(req, timeout=60) as r, open(dest + ".part", "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(dest + ".part", dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            time.sleep(1 + attempt)
        except Exception:
            time.sleep(1 + attempt)
    raise RuntimeError(f"download failed after {retries}: {url}")


def aggregate_file(zip_path, agg, write_row):
    """Stream the aggTrades CSV inside zip_path through `agg`, writing bars."""
    rows = 0
    with zipfile.ZipFile(zip_path) as z:
        inner = z.namelist()[0]
        with z.open(inner) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            for row in csv.reader(text):
                if not row:
                    continue
                try:
                    px = float(row[1])
                except ValueError:
                    continue                  # header line
                qty = float(row[2])
                ts = int(row[5])
                taker_buy = row[6].strip().lower() not in ("true", "1")  # buyer-maker => taker sell
                fin = agg.add(ts, px, qty, taker_buy)
                if fin is not None:
                    write_row(fin)
                rows += 1
                if rows % 20_000_000 == 0:
                    print(f"    ...{rows//1_000_000}M trades", file=sys.stderr)
    return rows


def process_symbol(sym, periods, daily, bar, outdir, opener):
    bar_ms = BAR_MS[bar]
    os.makedirs(outdir, exist_ok=True)
    main = open(os.path.join(outdir, f"{sym}_{bar}.csv"), "w", buffering=1)
    fp = open(os.path.join(outdir, f"{sym}_{bar}_fp.jsonl"), "w", buffering=1)
    main.write(",".join(COLS) + "\n")
    agg = BarAgg(bar_ms)
    bars = [0]

    def write_row(row):
        levels = row.pop("_levels", {})
        main.write(",".join(str(row[c]) for c in COLS) + "\n")
        fpd = {f"{k:g}": [round(v[0], 6), round(v[1], 6)] for k, v in levels.items()}
        fp.write(json.dumps({"ts": row["timestamp"], "levels": fpd}) + "\n")
        bars[0] += 1

    sub = "daily" if daily else "monthly"
    total_rows = 0
    for p in periods:
        url = f"{BASE}/{sub}/aggTrades/{sym}/{sym}-aggTrades-{p}.zip"
        tmp = os.path.join(outdir, f".{sym}-{p}.zip")
        print(f"  {sym} {p}: downloading ...", file=sys.stderr)
        if not download(url, tmp, opener):
            print(f"  {sym} {p}: 404 (not listed) — skipped", file=sys.stderr)
            continue
        size_mb = os.path.getsize(tmp) / 1e6
        print(f"  {sym} {p}: {size_mb:.0f}MB, aggregating ...", file=sys.stderr)
        total_rows += aggregate_file(tmp, agg, write_row)
        os.remove(tmp)
    if agg.bucket is not None:                # flush trailing bar
        write_row(agg._finalize())
    main.close(); fp.close()
    print(f"  {sym}: {total_rows:,} trades -> {bars[0]} {bar} bars", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Binance aggTrades -> order-flow bars")
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    ap.add_argument("--bar", default="15m", choices=list(BAR_MS))
    ap.add_argument("--start", help="YYYY-MM (monthly range start)")
    ap.add_argument("--end", help="YYYY-MM (monthly range end)")
    ap.add_argument("--daily", nargs="*", help="specific YYYY-MM-DD daily files")
    ap.add_argument("--out", default="data_binance")
    ap.add_argument("--proxy", default=None)
    args = ap.parse_args()

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": args.proxy, "https": args.proxy})
        if args.proxy else urllib.request.ProxyHandler({}))

    if args.daily:
        periods, daily = args.daily, True
    elif args.start and args.end:
        periods, daily = months_range(args.start, args.end), False
    else:
        ap.error("provide either --daily YYYY-MM-DD ... or --start/--end YYYY-MM")

    print(f"symbols={args.symbols} bar={args.bar} periods={periods} -> {args.out}/")
    for sym in args.symbols:
        process_symbol(sym, periods, daily, args.bar, args.out, opener)


if __name__ == "__main__":
    main()
