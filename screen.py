"""
Screen OKX USDT perpetuals for instruments that MATCH the LSOF strategy.

The filter is objective and programmable — same ethos as the signals, every gate
is a number, no discretion:

  * settleCcy == USDT, instType == SWAP, state == live   (linear perps)
  * 24h USD turnover  >= MIN_TURNOVER_USD   (clean order flow, low slippage)
  * open interest USD >= MIN_OI_USD         (real participation / book depth)
  * MIN_RANGE_PCT <= (high24h-low24h)/last <= MAX_RANGE_PCT
        lower bound: enough daily range to reach multi-R targets
        upper bound: exclude degenerate / illiquid hyper-volatility
  * tickSz/last <= MAX_REL_TICK             (tick fine enough for stop/R math)
  * listed >= MIN_AGE_DAYS ago              (history for warmup + flow window)
  * Rubik taker-volume exists for the base ccy (REQUIRED — S4 uses ccy flow)

Liquidity math verified against live OKX fields:
  turnover_usd = volCcy24h * last          (volCcy24h is base-ccy volume)
  oi_usd       = open-interest `oiUsd`     (provided directly)

Usage:
  py screen.py --proxy http://127.0.0.1:7890 --out universe.csv
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from okx_fetch import OKX

# --- objective thresholds (override via CLI) ------------------------------- #
MIN_TURNOVER_USD = 50_000_000
MIN_OI_USD = 20_000_000
MIN_RANGE_PCT = 0.015
MAX_RANGE_PCT = 0.20
MAX_REL_TICK = 5e-4
MIN_AGE_DAYS = 30
MIN_FLOW_PTS = 100          # 1H Rubik taker-volume points the ccy must return


def _f(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def screen(api: OKX, inst_type: str, settle: str,
           min_turnover: float, min_oi: float) -> pd.DataFrame:
    instruments = api._get("/api/v5/public/instruments", {"instType": inst_type})
    tickers = {d["instId"]: d for d in api._get("/api/v5/market/tickers", {"instType": inst_type})}
    ois = {d["instId"]: d for d in api._get("/api/v5/public/open-interest", {"instType": inst_type})}
    now_ms = int(time.time() * 1000)

    rows = []
    for ins in instruments:
        iid = ins["instId"]
        if ins.get("settleCcy") != settle or ins.get("state") != "live":
            continue
        tk, oi = tickers.get(iid), ois.get(iid)
        if not tk or not oi:
            continue
        last = _f(tk.get("last"))
        if last <= 0:
            continue
        rows.append(dict(
            instId=iid, ccy=iid.split("-")[0], last=last,
            turnover_usd=_f(tk.get("volCcy24h")) * last,
            oi_usd=_f(oi.get("oiUsd")),
            range_pct=(_f(tk.get("high24h")) - _f(tk.get("low24h"))) / last,
            rel_tick=_f(ins.get("tickSz")) / last,
            age_days=(now_ms - _f(ins.get("listTime"))) / 86_400_000,
            tickSz=ins.get("tickSz"),
        ))
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # quantitative gates (everything except Rubik flow)
    df["q_pass"] = (
        (df.turnover_usd >= min_turnover) & (df.oi_usd >= min_oi) &
        (df.range_pct >= MIN_RANGE_PCT) & (df.range_pct <= MAX_RANGE_PCT) &
        (df.rel_tick <= MAX_REL_TICK) & (df.age_days >= MIN_AGE_DAYS)
    )

    # Rubik flow availability per ccy — only for q_pass rows, to bound calls
    flow_pts = {}
    cands = sorted(df.loc[df.q_pass, "ccy"].unique())
    print(f"checking Rubik flow for {len(cands)} candidate ccys ...", file=sys.stderr)
    for ccy in cands:
        try:
            data = api._get("/api/v5/rubik/stat/taker-volume",
                            {"ccy": ccy, "instType": "CONTRACTS", "period": "1H"})
            flow_pts[ccy] = len(data)
        except Exception:
            flow_pts[ccy] = 0
    df["flow_pts"] = df.ccy.map(flow_pts).fillna(0).astype(int)
    df["flow_ok"] = df.flow_pts >= MIN_FLOW_PTS
    df["match"] = df.q_pass & df.flow_ok

    return df.sort_values(["match", "turnover_usd"], ascending=[False, False]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Screen OKX perps for LSOF-suitable pairs")
    ap.add_argument("--proxy", default=None, help="e.g. http://127.0.0.1:7890")
    ap.add_argument("--base", default="https://www.okx.com")
    ap.add_argument("--inst-type", default="SWAP")
    ap.add_argument("--settle", default="USDT")
    ap.add_argument("--min-turnover", type=float, default=MIN_TURNOVER_USD)
    ap.add_argument("--min-oi", type=float, default=MIN_OI_USD)
    ap.add_argument("--out", default="universe.csv")
    args = ap.parse_args()

    api = OKX(base=args.base, proxy=args.proxy)
    df = screen(api, args.inst_type, args.settle, args.min_turnover, args.min_oi)
    if df.empty:
        print("no instruments returned (check region/proxy)")
        return

    df.to_csv(args.out, index=False)
    matched = df[df.match]
    print(f"\n{len(matched)} / {len(df)} {args.settle} {args.inst_type} match the strategy")
    print(f"(thresholds: turnover>=${args.min_turnover/1e6:.0f}M  oi>=${args.min_oi/1e6:.0f}M  "
          f"range {MIN_RANGE_PCT*100:.1f}-{MAX_RANGE_PCT*100:.0f}%  relTick<={MAX_REL_TICK*1e4:.0f}bps)\n")

    show = matched.copy()
    show["turn$M"] = (show.turnover_usd / 1e6).round(0).astype(int)
    show["oi$M"] = (show.oi_usd / 1e6).round(0).astype(int)
    show["range%"] = (show.range_pct * 100).round(2)
    show["tick_bps"] = (show.rel_tick * 1e4).round(2)
    cols = ["instId", "last", "turn$M", "oi$M", "range%", "tick_bps", "flow_pts"]
    print(show[cols].to_string(index=False))
    print(f"\nFull table (incl. rejects) -> {args.out}")
    print(f"Backtest one:  py okx_fetch.py --inst {matched.iloc[0].instId} --bar 1H "
          f"--start <d> --end <d> --proxy {args.proxy or '...'} --out d.csv "
          f"&& py strategy.py --csv d.csv --tick {matched.iloc[0].tickSz}")


if __name__ == "__main__":
    main()
