"""
OKX trades-websocket recorder -> real per-instrument order-flow bars.

Subscribes to the public `trades` channel for a set of instruments and
aggregates the live taker tape into time bars, writing one append-only CSV per
instrument in a schema strategy.py can read directly (plus flow columns):

    timestamp, open, high, low, close, volume, buy_vol, sell_vol, delta, cvd, trades

  * `side` on each OKX trade is the TAKER/aggressor side -> buy_vol / sell_vol.
  * delta = buy_vol - sell_vol ;  cvd = running cumulative delta (resumed from
    the existing file on restart, so CVD is continuous across runs).

This is the data foundation for the Wyckoff Spring/Upthrust detector: real,
per-instrument effort-vs-result flow that the currency-level Rubik proxy could
never provide. It must run continuously to accumulate history (you cannot
backfill a websocket), so run it on a box that stays up.

Usage:
  py flow_recorder.py --proxy http://127.0.0.1:7890           # 1m bars, forever
  py flow_recorder.py --proxy http://127.0.0.1:7890 --bar-sec 5 --seconds 30   # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.parse

import websocket  # websocket-client

WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
DEFAULT_INSTS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP",
                 "BNB-USDT-SWAP", "DOGE-USDT-SWAP"]
COLS = ["timestamp", "open", "high", "low", "close", "volume",
        "buy_vol", "sell_vol", "delta", "cvd", "trades"]


class BarAgg:
    """Accumulates trades into one bar; rolls over on bucket change."""
    def __init__(self, bar_ms: int, cvd0: float = 0.0):
        self.bar_ms = bar_ms
        self.cvd = cvd0
        self.bucket = None
        self.o = self.h = self.l = self.c = None
        self.vol = self.buy = self.sell = 0.0
        self.n = 0

    def _start(self, bucket: int, px: float):
        self.bucket = bucket
        self.o = self.h = self.l = self.c = px
        self.vol = self.buy = self.sell = 0.0
        self.n = 0

    def add(self, ts: int, px: float, sz: float, side: str):
        """Return a finalized bar dict if this trade rolled the bucket, else None."""
        bucket = (ts // self.bar_ms) * self.bar_ms
        finalized = None
        if self.bucket is None:
            self._start(bucket, px)
        elif bucket > self.bucket:
            finalized = self._finalize()
            self._start(bucket, px)
        elif bucket < self.bucket:
            return None                      # late out-of-order trade: drop
        # update current bar
        self.c = px
        self.h = max(self.h, px)
        self.l = min(self.l, px)
        self.vol += sz
        self.n += 1
        if side == "buy":
            self.buy += sz
        else:
            self.sell += sz
        return finalized

    def _finalize(self) -> dict:
        delta = self.buy - self.sell
        self.cvd += delta
        return {"timestamp": self.bucket, "open": self.o, "high": self.h,
                "low": self.l, "close": self.c, "volume": self.vol,
                "buy_vol": self.buy, "sell_vol": self.sell, "delta": delta,
                "cvd": self.cvd, "trades": self.n}


class Recorder:
    def __init__(self, insts, bar_sec, outdir):
        self.insts = insts
        self.bar_ms = bar_sec * 1000
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.files = {}
        self.aggs = {}
        for inst in insts:
            path = os.path.join(outdir, f"{inst}_{bar_sec}s.csv")
            cvd0, new = self._resume_cvd(path)
            self.files[inst] = open(path, "a", buffering=1)  # line-buffered
            if new:
                self.files[inst].write(",".join(COLS) + "\n")
            self.aggs[inst] = BarAgg(self.bar_ms, cvd0)
        self.bars_written = 0

    def _resume_cvd(self, path):
        """Resume running CVD from the last row of an existing file."""
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return 0.0, True
        last = None
        with open(path) as f:
            for line in f:
                if line.strip():
                    last = line
        try:
            return float(last.split(",")[COLS.index("cvd")]), False
        except Exception:
            return 0.0, False

    def on_trade(self, tr: dict):
        inst = tr["instId"]
        agg = self.aggs.get(inst)
        if agg is None:
            return
        row = agg.add(int(tr["ts"]), float(tr["px"]), float(tr["sz"]), tr["side"])
        if row is not None:
            self.files[inst].write(",".join(str(row[c]) for c in COLS) + "\n")
            self.bars_written += 1

    def close(self):
        for f in self.files.values():
            f.close()


def run(proxy, insts, bar_sec, outdir, seconds):
    rec = Recorder(insts, bar_sec, outdir)
    purl = urllib.parse.urlparse(proxy) if proxy else None

    def on_open(ws):
        ws.send(json.dumps({"op": "subscribe",
                            "args": [{"channel": "trades", "instId": i} for i in insts]}))
        print(f"subscribed: {insts}  ({bar_sec}s bars -> {outdir}/)")

        def keepalive():                     # OKX drops idle conns after 30s
            while ws.keep_running:
                time.sleep(20)
                try:
                    ws.send("ping")
                except Exception:
                    break
        threading.Thread(target=keepalive, daemon=True).start()

    def on_message(ws, msg):
        if msg == "pong":
            return
        try:
            d = json.loads(msg)
        except Exception:
            return
        if "event" in d:                     # subscribe ack / error
            if d.get("event") == "error":
                print("WS error:", d)
            return
        if d.get("arg", {}).get("channel") == "trades":
            for tr in d.get("data", []):
                rec.on_trade(tr)

    def on_error(ws, err):
        print("WS error:", err)

    ws = websocket.WebSocketApp(WS_URL, on_open=on_open, on_message=on_message,
                                on_error=on_error)
    if seconds:                              # bounded run (smoke test)
        threading.Timer(seconds, ws.close).start()

    kw = dict(reconnect=5)
    if purl:
        kw.update(http_proxy_host=purl.hostname, http_proxy_port=purl.port,
                  proxy_type="http")
    try:
        ws.run_forever(**kw)
    finally:
        rec.close()
        print(f"stopped. bars written this run: {rec.bars_written}")


def main():
    ap = argparse.ArgumentParser(description="OKX trades -> order-flow bars recorder")
    ap.add_argument("--proxy", default=None, help="http://127.0.0.1:7890")
    ap.add_argument("--instruments", nargs="*", default=DEFAULT_INSTS)
    ap.add_argument("--bar-sec", type=int, default=60, help="bar size in seconds")
    ap.add_argument("--out", default="data_live")
    ap.add_argument("--seconds", type=int, default=0, help="run duration (0 = forever)")
    args = ap.parse_args()
    run(args.proxy, args.instruments, args.bar_sec, args.out, args.seconds)


if __name__ == "__main__":
    main()
