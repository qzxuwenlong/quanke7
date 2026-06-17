"""
OKX V5 market-data fetcher for the LSOF strategy.

Produces a CSV with the exact schema strategy.py expects:
    timestamp, open, high, low, close, volume, buy_vol, sell_vol

Data sources (public endpoints, no API key required):
  * OHLC  : GET /api/v5/market/history-candles  (keeps confirm=="1" only)
  * Flow  : GET /api/v5/market/history-trades   (each trade's `side` is the
            TAKER/aggressor side -> bucket into candles for buy_vol/sell_vol)

OKX has no per-instrument, per-candle taker-volume endpoint, so aggregating the
trade tape is the faithful way to get order flow. `volume` is taken from the
tape (buy_vol + sell_vol) so the three volume columns are always consistent.

No-repaint at the data layer:
  * only CLOSED candles (confirm=="1") are written;
  * candle `ts` is the bar OPEN time (OKX convention), and trades are bucketed
    by floor(ts / bar_ms) so each trade lands in the bar it occurred in.

Usage:
    py okx_fetch.py --inst BTC-USDT-SWAP --bar 15m \
        --start 2024-01-01 --end 2024-02-01 --out data.csv

    # OHLC only (skip the trade tape) if you just want to smoke-test candles:
    py okx_fetch.py --inst BTC-USDT-SWAP --bar 15m --start 2024-01-01 \
        --end 2024-01-02 --no-flow --out ohlc_only.csv

Then:
    py strategy.py --csv data.csv --tick <instrument_tick_size>

Requires: pandas (stdlib urllib is used for HTTP — no `requests` dependency).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

# bar string -> milliseconds. OKX uses lowercase for sub-hour, uppercase H/D/W/M.
BAR_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1H": 3_600_000, "2H": 7_200_000, "4H": 14_400_000,
    "6H": 21_600_000, "12H": 43_200_000, "1D": 86_400_000, "1W": 604_800_000,
}


def to_ms(s: str) -> int:
    """Parse 'YYYY-MM-DD' or epoch-ms/seconds into epoch-ms (UTC)."""
    s = str(s).strip()
    if s.isdigit():
        v = int(s)
        return v if v > 10_000_000_000 else v * 1000  # seconds -> ms heuristic
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)


class OKX:
    def __init__(self, base: str = "https://www.okx.com", pause: float = 0.25,
                 timeout: float = 20.0, retries: int = 5, proxy: str | None = None):
        self.base = base.rstrip("/")
        self.pause = pause
        self.timeout = timeout
        self.retries = retries
        # Route through an HTTP(S) proxy (e.g. Clash on 127.0.0.1:7890) when set.
        # ProxyHandler tunnels https via CONNECT automatically.
        if proxy:
            self.opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
        else:
            self.opener = urllib.request.build_opener()

    def _get(self, path: str, params: dict) -> list:
        url = f"{self.base}{path}?{urllib.parse.urlencode(params)}"
        last_err = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "lsof-fetch/1.0"})
                with self.opener.open(req, timeout=self.timeout) as r:
                    payload = json.loads(r.read().decode())
                if payload.get("code") not in ("0", 0):
                    # 50011 = rate limited; back off and retry
                    if str(payload.get("code")) == "50011":
                        time.sleep(1.0 + attempt)
                        continue
                    raise RuntimeError(f"OKX error {payload.get('code')}: {payload.get('msg')}")
                time.sleep(self.pause)
                return payload.get("data", [])
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (429, 503):
                    time.sleep(1.0 + attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                time.sleep(1.0 + attempt)
        raise RuntimeError(f"GET {path} failed after {self.retries} retries: {last_err}")

    # --- candles (paginate backward via `after` = older-than-ts) ---------- #
    def candles(self, inst: str, bar: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        rows, after = [], end_ms
        while True:
            data = self._get("/api/v5/market/history-candles",
                             {"instId": inst, "bar": bar, "after": after, "limit": 100})
            if not data:
                break
            rows.extend(data)
            oldest = int(data[-1][0])  # response is newest-first
            if oldest <= start_ms or len(data) < 100:
                break
            after = oldest
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "vol", "confirm"])
        df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "vol", "volCcy", "volCcyQuote", "confirm"])
        df["ts"] = df["ts"].astype("int64")
        df = df[(df.ts >= start_ms) & (df.ts < end_ms)]
        df = df[df["confirm"] == "1"]                    # CLOSED candles only
        df = df.drop_duplicates("ts").sort_values("ts")
        out = pd.DataFrame({
            "timestamp": df["ts"].to_numpy(),
            "open": df["o"].astype(float).to_numpy(),
            "high": df["h"].astype(float).to_numpy(),
            "low": df["l"].astype(float).to_numpy(),
            "close": df["c"].astype(float).to_numpy(),
        })
        return out.reset_index(drop=True)

    # --- trades (anchor by ts, then paginate older via tradeId) ----------- #
    def trades(self, inst: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        rows = []
        # anchor: newest trades at/under end_ms (type=2 -> after is a timestamp)
        data = self._get("/api/v5/market/history-trades",
                         {"instId": inst, "type": "2", "after": end_ms, "limit": 100})
        n_batches = 0
        while data:
            rows.extend(data)
            n_batches += 1
            oldest_ts = int(data[-1]["ts"])
            oldest_id = min(int(d["tradeId"]) for d in data)
            if oldest_ts <= start_ms or len(data) < 100:
                break
            # paginate strictly older by tradeId to avoid duplicate-ts stalls
            data = self._get("/api/v5/market/history-trades",
                             {"instId": inst, "type": "1", "after": oldest_id, "limit": 100})
            if n_batches % 25 == 0:
                print(f"  ...fetched {len(rows)} trades, at ts={oldest_ts}", file=sys.stderr)
        if not rows:
            return pd.DataFrame(columns=["ts", "sz", "side"])
        df = pd.DataFrame(rows)
        df["ts"] = df["ts"].astype("int64")
        df["sz"] = df["sz"].astype(float)
        df = df[(df.ts >= start_ms) & (df.ts < end_ms)]
        return df[["ts", "sz", "side"]].drop_duplicates()

    # --- Rubik taker volume (ccy-level historical flow; lightweight) ------ #
    def taker_volume(self, ccy: str, inst_type: str, period: str,
                     start_ms: int, end_ms: int) -> pd.DataFrame:
        """GET /rubik/stat/taker-volume -> rows of [ts, sellVol, buyVol].
        Currency-level aggressor volume; paginated backward via `end`."""
        # NOTE: only `end` is sent. Passing `begin` beyond the series retention
        # (5m ~ 48h, 1H ~ 30d, 1D longer) triggers OKX 50030 "Illegal time
        # range"; instead we page back via `end` and filter client-side, so an
        # out-of-retention start degrades to partial coverage + a warning.
        rows, end = [], end_ms
        for _ in range(500):  # safety bound
            data = self._get("/api/v5/rubik/stat/taker-volume",
                             {"ccy": ccy, "instType": inst_type, "period": period,
                              "end": end})
            if not data:
                break
            rows.extend(data)
            oldest = int(data[-1][0])
            if oldest <= start_ms or oldest == end:  # reached start / no progress
                break
            end = oldest
        if not rows:
            return pd.DataFrame(columns=["ts", "buy_vol", "sell_vol"])
        df = pd.DataFrame(rows, columns=["ts", "sellVol", "buyVol"])
        df["ts"] = df["ts"].astype("int64")
        df = df[(df.ts >= start_ms) & (df.ts < end_ms)].drop_duplicates("ts")
        return pd.DataFrame({
            "ts": df["ts"].to_numpy(),
            "buy_vol": df["buyVol"].astype(float).to_numpy(),    # taker buy
            "sell_vol": df["sellVol"].astype(float).to_numpy(),  # taker sell
        }).sort_values("ts").reset_index(drop=True)


def derive_ccy_type(inst: str) -> tuple[str, str]:
    """BTC-USDT-SWAP -> (BTC, CONTRACTS); BTC-USDT -> (BTC, SPOT)."""
    parts = inst.split("-")
    return parts[0], ("CONTRACTS" if len(parts) >= 3 else "SPOT")


def rubik_period_for(bar: str) -> str:
    """Pick the Rubik period to resample up into `bar` (must be >= 5m)."""
    bms = BAR_MS[bar]
    if bms < BAR_MS["5m"]:
        raise ValueError(f"--flow rubik needs bar >= 5m (got {bar}); use --flow tape")
    if bms < BAR_MS["1H"]:
        return "5m"
    if bms < BAR_MS["1D"]:
        return "1H"
    return "1D"


def resample_flow(flow: pd.DataFrame, bar_ms: int) -> pd.DataFrame:
    """Sum per-period buy/sell volume into bar buckets (bar OPEN time)."""
    if flow.empty:
        return pd.DataFrame(columns=["timestamp", "buy_vol", "sell_vol"])
    f = flow.copy()
    f["timestamp"] = (f["ts"] // bar_ms) * bar_ms
    return f.groupby("timestamp")[["buy_vol", "sell_vol"]].sum().reset_index()


def bucket_flow(trades: pd.DataFrame, bar_ms: int) -> pd.DataFrame:
    """Aggregate the taker tape into per-bar buy/sell volume."""
    if trades.empty:
        return pd.DataFrame(columns=["timestamp", "buy_vol", "sell_vol"])
    t = trades.copy()
    t["timestamp"] = (t["ts"] // bar_ms) * bar_ms      # bar OPEN time
    t["buy_vol"] = t["sz"].where(t["side"] == "buy", 0.0)   # taker buy
    t["sell_vol"] = t["sz"].where(t["side"] == "sell", 0.0)  # taker sell
    g = t.groupby("timestamp")[["buy_vol", "sell_vol"]].sum().reset_index()
    return g


OUT_COLS = ["timestamp", "open", "high", "low", "close", "volume", "buy_vol", "sell_vol"]


def build(inst: str, bar: str, start_ms: int, end_ms: int, base: str,
          flow_mode: str = "rubik", proxy: str | None = None,
          ccy: str | None = None, inst_type: str | None = None) -> pd.DataFrame:
    if bar not in BAR_MS:
        raise ValueError(f"unsupported bar '{bar}'. Choose from {list(BAR_MS)}")
    api = OKX(base=base, proxy=proxy)
    print(f"Fetching candles {inst} {bar} ...", file=sys.stderr)
    candles = api.candles(inst, bar, start_ms, end_ms)
    if candles.empty:
        raise RuntimeError("no candles returned (check inst/bar/date range or region access)")
    print(f"  {len(candles)} closed candles", file=sys.stderr)

    if flow_mode == "none":
        candles["volume"] = candles["buy_vol"] = candles["sell_vol"] = float("nan")
        return candles[OUT_COLS]

    if flow_mode == "rubik":
        period = rubik_period_for(bar)
        c, it = derive_ccy_type(inst)
        ccy, inst_type = ccy or c, inst_type or it
        print(f"Fetching Rubik taker-volume ({ccy}/{inst_type}, {period} -> {bar}) ...",
              file=sys.stderr)
        tv = api.taker_volume(ccy, inst_type, period, start_ms, end_ms)
        print(f"  {len(tv)} {period} flow points", file=sys.stderr)
        flow = resample_flow(tv, BAR_MS[bar])
    elif flow_mode == "tape":
        print("Fetching trade tape for order flow (heavy on liquid perps) ...",
              file=sys.stderr)
        trades = api.trades(inst, start_ms, end_ms)
        print(f"  {len(trades)} trades", file=sys.stderr)
        flow = bucket_flow(trades, BAR_MS[bar])
    else:
        raise ValueError(f"unknown flow_mode '{flow_mode}'")

    df = candles.merge(flow, on="timestamp", how="left")
    df[["buy_vol", "sell_vol"]] = df[["buy_vol", "sell_vol"]].fillna(0.0)
    df["volume"] = df["buy_vol"] + df["sell_vol"]      # consistent with flow source

    covered = (df["volume"] > 0).mean() * 100
    print(f"  flow coverage: {covered:.1f}% of candles have flow data", file=sys.stderr)
    if covered < 90:
        print("  WARNING: low flow coverage — flow source may not span the full "
              "window. Order-flow signals (S4) need buy/sell volume; backtest only "
              "the covered range.", file=sys.stderr)
    return df[OUT_COLS]


def main():
    ap = argparse.ArgumentParser(description="Fetch OKX OHLC + order flow -> LSOF CSV")
    ap.add_argument("--inst", required=True, help="e.g. BTC-USDT-SWAP or BTC-USDT")
    ap.add_argument("--bar", default="15m", help=f"one of {list(BAR_MS)}")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD or epoch")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD or epoch")
    ap.add_argument("--out", default="okx_data.csv")
    ap.add_argument("--base", default="https://www.okx.com",
                    help="API base; try https://aws.okx.com if blocked")
    ap.add_argument("--proxy", default=None,
                    help="HTTP(S) proxy, e.g. http://127.0.0.1:7890")
    ap.add_argument("--flow", choices=["rubik", "tape", "none"], default="rubik",
                    help="flow source: rubik=ccy taker-volume (default, fast, "
                         "spans history); tape=per-trade aggregation (precise but "
                         "heavy, short windows only); none=OHLC only")
    ap.add_argument("--no-flow", action="store_true", help="alias for --flow none")
    ap.add_argument("--ccy", default=None, help="override Rubik ccy (default from inst)")
    ap.add_argument("--inst-type", default=None, help="override SPOT/CONTRACTS")
    args = ap.parse_args()

    start_ms, end_ms = to_ms(args.start), to_ms(args.end)
    if end_ms <= start_ms:
        ap.error("--end must be after --start")

    flow_mode = "none" if args.no_flow else args.flow
    df = build(args.inst, args.bar, start_ms, end_ms, args.base, flow_mode,
               args.proxy, args.ccy, args.inst_type)
    df.to_csv(args.out, index=False)
    span = f"{pd.Timestamp(df.timestamp.iloc[0], unit='ms', tz='UTC')} -> " \
           f"{pd.Timestamp(df.timestamp.iloc[-1], unit='ms', tz='UTC')}"
    print(f"\nWrote {len(df)} rows to {args.out}\n  span: {span}")
    print(f"  next: py strategy.py --csv {args.out} --tick <tick_size>")


if __name__ == "__main__":
    main()
