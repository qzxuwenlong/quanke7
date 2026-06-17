"""
Random-entry baseline (Monte Carlo permutation test) for the LSOF signal.

Question: do the S1-S6 entry conditions actually select bars with better
favorable excursion than chance? Or is the signal noise?

Null model: identical entry/stop MECHANICS (buy/sell-stop trigger within
ENTRY_VALID_BARS, structural stop = swing wick +/- buffer, held to stop), but
the entry BAR is chosen at random and the SIDE is a coin flip. Everything that
differs from the real strategy is exactly the signal logic.

We match the real per-symbol signal counts, draw many random books, and build
the null distribution of:
  * mean MFE (R)            * P(MFE >= 1R), P(MFE >= 2R)
  * oracle expectancy E(N) = P(MFE>=N)*N - P(MFE<N)*1   (pure function of MFE)
Then we locate the REAL statistics in that null -> one-sided empirical p-value.

If the real stats sit inside the null (high p), the signal adds no edge and no
amount of exit/parameter tuning will help — fix the input (per-instrument flow).

Usage:  py random_baseline.py            # uses ./cache + universe.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import replace
from strategy import Config, compute_features, Backtester
from portfolio import PortfolioBacktester
from mfe_study import load_cache, BIG

WALK_CAP = 300          # max bars to hold a random entry while measuring MFE
B = 2000                # Monte Carlo iterations
SEED = 42


def real_signal_mfe(data, bt_class=Backtester):
    """Run the actual signal census (no caps/TPs) -> per-trade MFE + counts."""
    cfg = replace(Config(), TP1_R=BIG, TP2_R=BIG, TP3_R=BIG, TIME_STOP_BARS=BIG,
                  MAX_DD_HALT=BIG, DAILY_LOSS_LIMIT=BIG, MAX_CONCURRENT=BIG,
                  MAX_AGGREGATE_RISK=BIG, LOSS_STREAK_TRIGGER=BIG)
    return PortfolioBacktester(data, cfg, bt_class=bt_class).run()["trade_log"]


def random_entry_mfe(h, l, rng_arr, ur, tick, t, side, cfg):
    """MFE of a single entry at bar t/side, held to the structural stop.
    Returns (mfe_in_R, mfe_in_volunits) or None if the stop never triggers."""
    n = len(h)
    buf = max(cfg.SL_TICKS * tick, cfg.SL_RANGE_FRAC * rng_arr[t])
    if side == 1:
        entry, sl = h[t] + tick, l[t] - buf
    else:
        entry, sl = l[t] - tick, h[t] + buf
    R = abs(entry - sl)
    if R <= 0:
        return None
    fill = None
    for k in range(t + 1, min(t + 1 + cfg.ENTRY_VALID_BARS, n)):
        if (side == 1 and h[k] >= entry) or (side == -1 and l[k] <= entry):
            fill = k
            break
    if fill is None:
        return None
    hi = 0.0
    for j in range(fill, min(fill + WALK_CAP, n)):
        if side == 1:
            hi = max(hi, h[j] - entry)
            if l[j] <= sl:
                break
        else:
            hi = max(hi, entry - l[j])
            if h[j] >= sl:
                break
    u = ur[fill]
    mfe_vol = hi / u if (u and not np.isnan(u)) else np.nan
    return hi / R, mfe_vol


def oracle_E(mfe, N):
    mfe = np.asarray(mfe)
    return float(np.where(mfe >= N, N, -1.0).mean())


def monte_carlo_compare(data, real, cfg=None, B=B, seed=SEED):
    if cfg is None:
        cfg = Config()
    counts = real.groupby("symbol").size().to_dict()
    n_real = int(sum(counts.values()))

    # numpy arrays per symbol (incl. unit_range for vol-normalized MFE)
    arr = {}
    warm = cfg.N_VOL
    for sym, (df, tick) in data.items():
        f = compute_features(df, cfg)
        h = f["high"].to_numpy(float); l = f["low"].to_numpy(float)
        arr[sym] = (h, l, (h - l), f["unit_range"].to_numpy(float), float(tick), len(h))

    rng = np.random.default_rng(seed)
    null = {"meanMFE_R": [], "E1": [], "E2": [], "meanMFE_vol": []}
    for _ in range(B):
        pool_R, pool_vol = [], []
        for sym, n_s in counts.items():
            h, l, r, ur, tick, n = arr[sym]
            got = tries = 0
            while got < n_s and tries < n_s * 30:
                tries += 1
                t = int(rng.integers(warm, n - 2))
                side = 1 if rng.random() < 0.5 else -1
                m = random_entry_mfe(h, l, r, ur, tick, t, side, cfg)
                if m is not None:
                    pool_R.append(m[0]); pool_vol.append(m[1]); got += 1
        pool_R = np.array(pool_R); pool_vol = np.array(pool_vol)
        null["meanMFE_R"].append(pool_R.mean())
        null["E1"].append(oracle_E(pool_R, 1))
        null["E2"].append(oracle_E(pool_R, 2))
        null["meanMFE_vol"].append(np.nanmean(pool_vol))

    rmfe = real.mfe_R.to_numpy(); rvol = real.mfe_vol.to_numpy()
    real_stat = {"meanMFE_R": rmfe.mean(), "E1": oracle_E(rmfe, 1),
                 "E2": oracle_E(rmfe, 2), "meanMFE_vol": np.nanmean(rvol)}

    print(f"real signals: {n_real} trades across {len(counts)} symbols")
    print(f"random books: {B} x {n_real} entries (matched per-symbol counts, coin-flip side)\n")
    labels = {"meanMFE_R": "mean MFE (R)", "E1": "oracle E(1R)", "E2": "oracle E(2R)",
              "meanMFE_vol": "mean MFE (vol-units)"}
    print("(A) strategy R-framework + (B) vol-normalized selection test")
    print(f"{'stat':>20} {'REAL':>9} {'null mean':>10} {'null 5%':>9} {'null 95%':>9} {'p(real>rand)':>13}")
    for k in ("meanMFE_R", "E1", "E2", "meanMFE_vol"):
        nd = np.array(null[k]); rv = real_stat[k]
        p = float((nd >= rv).mean())
        print(f"{labels[k]:>20} {rv:>9.3f} {nd.mean():>10.3f} {np.percentile(nd,5):>9.3f} "
              f"{np.percentile(nd,95):>9.3f} {p:>13.3f}")
    print("\nReading: p ~ 0.5 => real indistinguishable from random (no edge).")
    print("         p < 0.05 => real beats random.   p > 0.95 => real WORSE than random.")
    print("mean MFE (vol-units) isolates bar/direction selection from stop width.")


def main():
    data = load_cache()
    real = real_signal_mfe(data)
    monte_carlo_compare(data, real)


if __name__ == "__main__":
    main()
