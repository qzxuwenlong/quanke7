"""
MFE/MAE + exit-geometry study for the LSOF strategy.

Answers one question rigorously, WITHOUT curve-fitting: is the strategy's loss
caused by bad EXITS (signal moves in our favor but we capture too little) or by
a weak SIGNAL (price doesn't move our way enough, no exit rule can save it)?

Method (measurement, not optimization):
  * Strip all caps and TPs/time-stops -> every signal becomes one trade held to
    the structural stop. This is a pure SIGNAL CENSUS.
  * Record MFE (max favorable excursion, in R) and MAE (max adverse) per trade.
  * Compute the ORACLE expectancy of a single fixed +N target (exit at +N if the
    trade ever reached it before the stop, else take the stop outcome). Sweep N.
  * Split in-sample / out-of-sample by median entry time to check stability.

If oracle E(N) is positive for some stable N -> geometry was the bottleneck.
If oracle E(N) is negative for ALL N, in and out of sample -> the SIGNAL has no
edge and no exit rule can rescue it (look at the input/signal, not parameters).

Usage:  py mfe_study.py            # uses ./cache + universe.csv
"""
from __future__ import annotations

import glob
from dataclasses import replace

import numpy as np
import pandas as pd

from portfolio import PortfolioBacktester
from strategy import Config, load_csv

BIG = 10**9


def load_cache(cache="cache", universe="universe.csv", bar="1H"):
    ticks = dict(zip(*[pd.read_csv(universe)[c] for c in ("instId", "tickSz")]))
    data = {}
    for f in glob.glob(f"{cache}/*_{bar}_*.csv"):
        inst = f.replace("\\", "/").split("/")[-1].split(f"_{bar}_")[0]
        data[inst] = (load_csv(f), float(ticks[inst]))
    return data


def run():
    data = load_cache()
    # no caps -> full signal census; cfg_stop also disables TPs/time-stop
    census = dict(MAX_DD_HALT=BIG, DAILY_LOSS_LIMIT=BIG, MAX_CONCURRENT=BIG,
                  MAX_AGGREGATE_RISK=BIG, LOSS_STREAK_TRIGGER=BIG)
    cfg_stop = replace(Config(), TP1_R=BIG, TP2_R=BIG, TP3_R=BIG, TIME_STOP_BARS=BIG, **census)
    cfg_base = replace(Config(), **census)

    A = PortfolioBacktester(data, cfg_stop).run()["trade_log"]   # stop-only -> true MFE
    B = PortfolioBacktester(data, cfg_base).run()["trade_log"]   # current TP geometry

    print(f"signal census: {len(A)} trades (stop-only) | current-TP census: {len(B)}")
    print(f"\nMFE before structural stop:  mean={A.mfe_R.mean():.2f}R median={A.mfe_R.median():.2f}R"
          f"  >=1R:{(A.mfe_R>=1).mean():.0%} >=2R:{(A.mfe_R>=2).mean():.0%} >=3R:{(A.mfe_R>=3).mean():.0%}")
    print(f"MAE: mean={A.mae_R.mean():.2f}R median={A.mae_R.median():.2f}R")
    print(f"\nexpectancy/trade:  current-TP={B.R_multiple.mean():+.3f}R  "
          f"stop-only={A.R_multiple.mean():+.3f}R")

    def E(d, N):
        return np.where(d.mfe_R >= N, N, d.R_multiple).mean()

    cut = A.entry_ts.median()
    IS, OOS = A[A.entry_ts <= cut], A[A.entry_ts > cut]
    print("\noracle single-target expectancy E(N)  (fixed +N target + structural stop):")
    print(f"   {'N':>4} {'full':>8} {'in-samp':>8} {'out-samp':>8}")
    for N in (0.5, 1, 1.5, 2, 2.5, 3, 4, 5):
        print(f"   {N:>4} {E(A,N):>+8.3f} {E(IS,N):>+8.3f} {E(OOS,N):>+8.3f}")
    print("\nNote: oracle exits ignore fees (optimistic). If all E(N)<0 in & out of "
          "sample, the signal — not the exits — is the limiting factor.")


if __name__ == "__main__":
    run()
