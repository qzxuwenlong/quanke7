"""
Wyckoff Spring / Upthrust detector (Phase C) on naked price action + order flow.

A WyckoffBacktester that replaces the bare liquidity-sweep entry with a
context-conditioned one. The failed signal was, in Wyckoff terms, a context-free
spring; here a spring only counts when it occurs at the support of an ESTABLISHED
trading range AND order flow shows absorption (effort-vs-result divergence).

All rules are objective and non-repainting (confirmed swings, closed bars):

Trading range (TR) at bar t, from confirmed swings within the last RANGE_LOOKBACK:
  * >= MIN_TOUCHES swing highs cluster near a top, >= MIN_TOUCHES swing lows near
    a bottom (each within EDGE_TOL*width of the boundary),
  * width/price within [MIN_W, MAX_W] (a range, not a trend or a tick),
  * spans >= MIN_RANGE_BARS bars (cause has been built).

Spring (long):   low < support, close back inside (< top), rejection wick, AND
                 cvd[t] >= cvd[support-test bar]  (price made a lower low, CVD
                 did not -> demand absorbing supply).  Upthrust = mirror.

Stop: beyond the spring extreme.  Targets are STRUCTURAL (range mid, opposite
edge, and the measured move = edge + range width, i.e. Wyckoff cause->effect) —
not blind R-multiples — directly addressing the payoff-geometry finding.

This module also gates the signal: run it to compare the Wyckoff signal's MFE
profile against the random-entry baseline on whatever cached data exists.

    py wyckoff.py            # dry-run + random-baseline gate on ./cache
"""
from __future__ import annotations

import numpy as np

from strategy import Backtester, Config
from mfe_study import load_cache
from random_baseline import real_signal_mfe, monte_carlo_compare

CENSUS = 1e8   # if cfg.TP1_R exceeds this we're in stop-only census mode


class WyckoffBacktester(Backtester):
    # trading-range parameters (objective; not optimized — defaults for the gate)
    RANGE_LOOKBACK = 60
    MIN_TOUCHES = 2
    EDGE_TOL = 0.15        # within 15% of width counts as touching a boundary
    MIN_W = 0.01           # min range width as fraction of price (1%)
    MAX_W = 0.30           # max (above this it's trending, not ranging)
    MIN_RANGE_BARS = 20    # range must span at least this many bars

    def _current_range(self, t: int):
        """Confirmed-swing trading range as of bar t, or None."""
        lo = t - self.RANGE_LOOKBACK
        shs = [(b, p) for (b, p) in self.swing_highs if lo <= b <= t]
        sls = [(b, p) for (b, p) in self.swing_lows if lo <= b <= t]
        if len(shs) < self.MIN_TOUCHES or len(sls) < self.MIN_TOUCHES:
            return None
        top = max(p for _, p in shs)
        bot = min(p for _, p in sls)
        width = top - bot
        if width <= 0 or not (self.MIN_W <= width / bot <= self.MAX_W):
            return None
        tol = self.EDGE_TOL * width
        top_tests = [b for b, p in shs if p >= top - tol]
        bot_tests = [b for b, p in sls if p <= bot + tol]
        if len(top_tests) < self.MIN_TOUCHES or len(bot_tests) < self.MIN_TOUCHES:
            return None
        if t - min(b for b, _ in shs + sls) < self.MIN_RANGE_BARS:
            return None
        return {"top": top, "bot": bot, "width": width,
                "sup_bar": max(bot_tests), "res_bar": max(top_tests)}

    def _evaluate_signal(self, t: int):
        cfg = self.cfg
        if self.pos is not None or self.pending is not None:
            return
        if self.pf.halted or self.pf.daily_blocked:
            return
        if np.isnan(self.unit_range[t]) or self.rng[t] <= 0:
            return
        tr = self._current_range(t)
        if tr is None:
            return
        bot, top = tr["bot"], tr["top"]
        rej = cfg.REJ_FRAC * self.rng[t]

        # SPRING (long): shakeout below support, reclaim inside, CVD absorption
        if not self.block_long_until_new_SL:
            if (self.l[t] < bot and self.c[t] > bot and self.c[t] < top
                    and (self.c[t] - self.l[t]) >= rej
                    and self.cvd[t] >= self.cvd[tr["sup_bar"]]):
                self._cur_tr = tr
                self._place_entry(t, side=+1)
                return

        # UPTHRUST (short): poke above resistance, reclaim inside, CVD divergence
        if not self.block_short_until_new_SH:
            if (self.h[t] > top and self.c[t] < top and self.c[t] > bot
                    and (self.h[t] - self.c[t]) >= rej
                    and self.cvd[t] <= self.cvd[tr["res_bar"]]):
                self._cur_tr = tr
                self._place_entry(t, side=-1)
                return

    def _place_entry(self, t: int, side: int):
        cfg = self.cfg
        buf = self._buffer(t)
        tr = self._cur_tr
        census = cfg.TP1_R >= CENSUS
        if side == +1:
            entry_px, sl = self.h[t] + cfg.tick, self.l[t] - buf
        else:
            entry_px, sl = self.l[t] - cfg.tick, self.h[t] + buf
        if abs(entry_px - sl) <= 0:
            return
        if census:                              # stop-only: targets unreachable
            far = 1e12
            tps = [entry_px + far] * 3 if side == +1 else [entry_px - far] * 3
        elif side == +1:
            if tr["top"] <= entry_px + 4 * cfg.tick:    # no room to the edge
                return
            tps = [entry_px + 0.5 * (tr["top"] - entry_px), tr["top"], tr["top"] + tr["width"]]
        else:
            if tr["bot"] >= entry_px - 4 * cfg.tick:
                return
            tps = [entry_px - 0.5 * (entry_px - tr["bot"]), tr["bot"], tr["bot"] - tr["width"]]
        self.pending = {"side": side, "entry_px": entry_px, "sl": sl, "tp": tps,
                        "signal_bar": t, "earliest_fill_bar": t + 1,
                        "valid_until": t + cfg.ENTRY_VALID_BARS}


def main():
    data = load_cache()
    real = real_signal_mfe(data, bt_class=WyckoffBacktester)
    print(f"Wyckoff signal census: {len(real)} trades across "
          f"{real.symbol.nunique() if len(real) else 0} symbols")
    if len(real) < 10:
        print("Too few Wyckoff signals on this data to gate meaningfully "
              "(range/spring conditions are strict on a 30-day 1H sample).")
        if len(real):
            print(real[["symbol", "side", "mfe_R", "mae_R"]].to_string(index=False))
        return
    print(f"MFE: mean={real.mfe_R.mean():.2f}R median={real.mfe_R.median():.2f}R  "
          f">=1R:{(real.mfe_R>=1).mean():.0%} >=2R:{(real.mfe_R>=2).mean():.0%}\n")
    monte_carlo_compare(data, real)


if __name__ == "__main__":
    main()
