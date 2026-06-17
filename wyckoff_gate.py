"""
Gate the Wyckoff signal on REAL per-instrument Binance flow (data_binance/).

Run after binance_tape.py has populated data_binance/. Reports BOTH Gate-0 criteria:
  (a) random-baseline — does the signal beat random selection? (vol-normalized MFE,
      p<0.05) — the falsification gate.
  (b) cost-aware backtest — net expectancy/return after Binance fees + slippage
      (funding NOT yet modeled — treat (b) as slightly optimistic).

Usage:
  py wyckoff_gate.py                 # 15m bars, ./data_binance
  py wyckoff_gate.py 15m data_binance_snap
"""
from __future__ import annotations

import glob
import os
import sys
from dataclasses import replace

from strategy import Config, load_csv
from portfolio import PortfolioBacktester
from random_baseline import real_signal_mfe, monte_carlo_compare
from wyckoff import WyckoffBacktester

# real Binance USDT-M futures tick sizes (fetched from fapi exchangeInfo)
BINANCE_TICKS = {"BTCUSDT": 0.1, "ETHUSDT": 0.01, "BNBUSDT": 0.01,
                 "DOGEUSDT": 1e-05, "SOLUSDT": 0.01}


def load_binance(d="data_binance", bar="15m", min_bars=200):
    data = {}
    for f in sorted(glob.glob(os.path.join(d, f"*_{bar}.csv"))):
        sym = os.path.basename(f).split(f"_{bar}.csv")[0]
        tick = BINANCE_TICKS.get(sym)
        if tick is None:
            print(f"  skip {sym}: no tick size known", file=sys.stderr)
            continue
        try:
            df = load_csv(f)
        except Exception as e:
            print(f"  skip {sym}: {e}", file=sys.stderr)
            continue
        if len(df) >= min_bars:
            data[sym] = (df, tick)
        else:
            print(f"  {sym}: only {len(df)} bars (<{min_bars}) — skipped", file=sys.stderr)
    return data


def main():
    bar = sys.argv[1] if len(sys.argv) > 1 else "15m"
    d = sys.argv[2] if len(sys.argv) > 2 else "data_binance"
    data = load_binance(d, bar)
    if not data:
        print("no usable data_binance files yet — let the pull run longer.")
        return
    print("loaded: " + ", ".join(f"{s}:{len(df)}b" for s, (df, _) in data.items()))

    # (a) signal census + random-baseline significance ----------------------
    real = real_signal_mfe(data, bt_class=WyckoffBacktester)
    n = len(real)
    print(f"\nWyckoff signals: {n} across {real.symbol.nunique() if n else 0} symbols")
    if n < 10:
        print("too few signals to gate yet — need more history. (plumbing OK.)")
        return
    print(f"MFE: mean={real.mfe_R.mean():.2f}R median={real.mfe_R.median():.2f}R  "
          f">=1R:{(real.mfe_R>=1).mean():.0%} >=2R:{(real.mfe_R>=2).mean():.0%}")
    print("\n=== (a) random-baseline gate (signal selection vs chance) ===")
    monte_carlo_compare(data, real)

    # (b) cost-aware backtest with Binance fees ------------------------------
    print("\n=== (b) cost-aware backtest (Binance taker 0.04% / maker 0.02%; funding NOT modeled) ===")
    cfg = replace(Config(), FEE_TAKER=0.0004, FEE_MAKER=0.0002)
    res = PortfolioBacktester(data, cfg, bt_class=WyckoffBacktester).run()
    o = res.get("overall", {})
    print(f"trades={o.get('trades')}  net_return%={res.get('total_return_pct')}  "
          f"maxDD%={res.get('max_drawdown_pct')}  expectancyR={o.get('expR')}  PF={o.get('pf')}")
    print("\nPASS = (a) beats random p<0.05 on vol-norm MFE  AND  (b) net-positive after costs.")


if __name__ == "__main__":
    main()
