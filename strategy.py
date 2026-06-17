"""
LSOF — Liquidity-Sweep / Order-Flow Reversal strategy.
Reference event-driven backtest engine. See STRATEGY.md for the full spec.

Design guarantees (enforced in code, not just documented):
  * Closed-bar evaluation only; every decision at bar t uses data <= t.
  * Trailing statistics exclude the current bar (.shift(1)).
  * Swing pivots are confirmed P bars late and acted on only at confirmation.
  * Signals on bar t place resting orders fillable no earlier than bar t+1.
  * Intrabar ambiguity resolved conservatively (adverse stop assumed first).
  * Deterministic: same data + params => identical trade list.

No moving averages / RSI / MACD / Bollinger / any smoothed indicator is used.
The only volatility statistic (unit_range) is a raw trailing median of candle
ranges, used solely for buffers/filters — never as a directional signal.

Run a self-contained demo on synthetic data:
    python strategy.py --demo

Run on your own data (CSV with columns:
    timestamp, open, high, low, close, volume, buy_vol, sell_vol [, oi]):
    python strategy.py --csv path/to/data.csv --tick 0.1
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Configuration — the single source of truth for every parameter (STRATEGY §8) #
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # instrument
    tick: float = 0.1
    initial_equity: float = 10_000.0

    # structure / signal
    P: int = 2                    # pivot fractal half-width
    N_VOL: int = 50               # trailing window for unit_range / flow_unit
    REJ_FRAC: float = 0.5         # min rejection-wick fraction of bar range
    # Order-flow confirmation (S4), calibrated for currency-level flow:
    OPP_K: float = 0.5            # sweep-bar delta may not oppose by > OPP_K*flow_unit
    USE_CVD_DIV: bool = True      # also require CVD absorption vs the swept pivot
    MIN_RANGE_FRAC: float = 0.75  # signal-bar range vs unit_range
    LEVEL_MAX_AGE: int = 60       # swept level freshness (bars)
    ENTRY_VALID_BARS: int = 3     # entry-stop validity window

    # stop / targets
    SL_TICKS: int = 2
    SL_RANGE_FRAC: float = 0.10
    TP1_R: float = 1.0
    TP2_R: float = 2.0
    TP3_R: float = 3.0
    TP_SPLIT: tuple = (0.40, 0.30, 0.20, 0.10)  # TP1, TP2, TP3, runner

    # risk management
    RISK_PER_TRADE: float = 0.005
    MAX_LEVERAGE: float = 5.0
    MAX_CONCURRENT: int = 3        # max simultaneous open positions (portfolio)
    MAX_AGGREGATE_RISK: float = 0.02  # cap on summed open initial-risk (portfolio)
    DAILY_LOSS_LIMIT: float = 0.02
    LOSS_STREAK_TRIGGER: int = 3
    MAX_DD_HALT: float = 0.10
    TIME_STOP_BARS: int = 12

    # costs
    FEE_TAKER: float = 0.0005
    FEE_MAKER: float = 0.0002
    SLIPPAGE_TICKS: int = 1


# --------------------------------------------------------------------------- #
# Feature engineering — order flow + raw volatility unit (all trailing)        #
# --------------------------------------------------------------------------- #
def compute_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    df["delta"] = df["buy_vol"] - df["sell_vol"]
    df["cvd"] = df["delta"].cumsum()
    df["range"] = df["high"] - df["low"]
    df["body"] = (df["close"] - df["open"]).abs()
    # Trailing, current-bar-EXCLUDING statistics (shift(1) => no lookahead).
    df["unit_range"] = df["range"].rolling(cfg.N_VOL).median().shift(1)
    df["flow_unit"] = df["delta"].abs().rolling(cfg.N_VOL).median().shift(1)
    return df


# --------------------------------------------------------------------------- #
# Position state                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class Position:
    side: int                 # +1 long, -1 short
    entry_bar: int
    entry_px: float
    qty_full: float           # size at fill (units)
    qty_rem: float            # remaining open units
    sl: float                 # current stop (only moves toward profit)
    init_risk_px: float       # |entry - initial_sl| per unit
    tp: list                  # [tp1, tp2, tp3] price levels
    tp_filled: list = field(default_factory=lambda: [False, False, False])
    phase: str = "OPEN"       # OPEN -> RUNNER
    realized: float = 0.0     # realized $ pnl across partials (net of fees)
    pending_market_exit: bool = False
    sl_moved: bool = False    # True once SL is advanced (BE/lock/trail)
    hi_fav: float = 0.0       # max favorable excursion (price), for MFE
    hi_adv: float = 0.0       # max adverse excursion (price), for MAE


def to_epoch_ms(ts) -> int:
    """Coerce a bar timestamp (epoch-ms number or pandas/ISO) to epoch-ms int."""
    if isinstance(ts, (int, float, np.integer, np.floating)):
        return int(ts)
    return int(pd.Timestamp(ts).value // 1_000_000)


def day_of(ts) -> str:
    """UTC calendar day of a bar timestamp (epoch-ms int or parseable)."""
    if isinstance(ts, (int, float, np.integer, np.floating)):
        return datetime.fromtimestamp(float(ts) / 1000, tz=timezone.utc).date().isoformat()
    p = pd.Timestamp(ts)
    return (p.tz_localize(None) if p.tzinfo is None else p).date().isoformat()


# --------------------------------------------------------------------------- #
# Portfolio — shared equity + portfolio-level risk caps (STRATEGY §6).         #
# One Portfolio is shared by all per-symbol Backtesters so that concurrency,   #
# aggregate risk, daily loss, drawdown halt, and loss-streak de-risking apply  #
# across the whole book, not per symbol.                                       #
# --------------------------------------------------------------------------- #
class Portfolio:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.equity = cfg.initial_equity
        self.peak = cfg.initial_equity
        self.realized_total = 0.0
        self.loss_streak = 0
        self.halted = False
        self.daily_blocked = False
        self.current_day = None
        self.day_start_equity = cfg.initial_equity
        self.open_risk: dict[int, float] = {}   # token -> open initial-risk $
        self._tok = 0
        self.equity_curve: list[tuple] = []

    def eff_risk(self) -> float:
        mult = 0.5 if self.loss_streak >= self.cfg.LOSS_STREAK_TRIGGER else 1.0
        return self.cfg.RISK_PER_TRADE * mult

    def on_time(self, ts):
        """Roll the UTC trading day (resets daily-loss block). Idempotent."""
        day = day_of(ts)
        if day != self.current_day:
            self.current_day = day
            self.day_start_equity = self.equity
            self.daily_blocked = False

    def has_slot(self) -> bool:
        return (not self.halted and not self.daily_blocked
                and len(self.open_risk) < self.cfg.MAX_CONCURRENT)

    def can_add_risk(self, new_risk: float) -> bool:
        return (sum(self.open_risk.values()) + new_risk
                <= self.cfg.MAX_AGGREGATE_RISK * self.equity)

    def register(self, risk: float) -> int:
        self._tok += 1
        self.open_risk[self._tok] = risk
        return self._tok

    def book(self, token: int, net: float, ts):
        self.open_risk.pop(token, None)
        self.realized_total += net
        self.equity = self.cfg.initial_equity + self.realized_total
        self.peak = max(self.peak, self.equity)
        if (self.peak - self.equity) / self.peak >= self.cfg.MAX_DD_HALT:
            self.halted = True
        self.loss_streak = self.loss_streak + 1 if net < 0 else 0
        if (self.equity - self.day_start_equity) <= -self.cfg.DAILY_LOSS_LIMIT * self.day_start_equity:
            self.daily_blocked = True
        self.equity_curve.append((ts, self.equity))


# --------------------------------------------------------------------------- #
# Backtester (single symbol). Shares a Portfolio; PortfolioBacktester drives   #
# many of these on a merged timeline — see portfolio.py and STRATEGY §6.       #
# --------------------------------------------------------------------------- #
class Backtester:
    def __init__(self, df: pd.DataFrame, cfg: Config,
                 portfolio: "Portfolio | None" = None, symbol: str = "SYM"):
        self.cfg = cfg
        self.symbol = symbol
        self.pf = portfolio if portfolio is not None else Portfolio(cfg)
        self.df = compute_features(df, cfg)
        self.n = len(self.df)

        # numpy views for clarity / speed inside the bar loop
        self.t = self.df["timestamp"].to_numpy()
        self.o = self.df["open"].to_numpy(float)
        self.h = self.df["high"].to_numpy(float)
        self.l = self.df["low"].to_numpy(float)
        self.c = self.df["close"].to_numpy(float)
        self.delta = self.df["delta"].to_numpy(float)
        self.cvd = self.df["cvd"].to_numpy(float)
        self.rng = self.df["range"].to_numpy(float)
        self.unit_range = self.df["unit_range"].to_numpy(float)
        self.flow_unit = self.df["flow_unit"].to_numpy(float)

        # structure state
        self.swing_highs: list[tuple[int, float]] = []
        self.swing_lows: list[tuple[int, float]] = []
        self.last_SH = self.last_SL = None
        self.last_SH_bar = self.last_SL_bar = -1
        self.trend = 0

        # order / position state
        self.pos: Position | None = None
        self.pos_token: int | None = None    # portfolio open-risk handle
        self.pending = None  # dict for resting entry order

        # cooldown: need a fresh swing in trade direction after an SL/time exit
        self.block_long_until_new_SL = False
        self.block_short_until_new_SH = False
        self.long_exit_bar = -1
        self.short_exit_bar = -1

        self.trades: list[dict] = []

    # --- helpers ---------------------------------------------------------- #
    def _buffer(self, bar: int) -> float:
        return max(self.cfg.SL_TICKS * self.cfg.tick,
                   self.cfg.SL_RANGE_FRAC * self.rng[bar])

    # --- structure update (uses data <= t only) -------------------------- #
    def _update_structure(self, t: int):
        cfg = self.cfg
        P = cfg.P
        i = t - P  # candidate pivot; its right side closes exactly at bar t
        if i - P >= 0:
            lo, hi = i - P, i + P  # inclusive window, hi == t (all known)
            seg_h = self.h[lo:hi + 1]
            seg_l = self.l[lo:hi + 1]
            if self.h[i] == seg_h.max() and self.h[i] > self.h[i - 1]:
                self.swing_highs.append((i, self.h[i]))
                self.last_SH, self.last_SH_bar = self.h[i], i
                if self.block_short_until_new_SH and i > self.short_exit_bar:
                    self.block_short_until_new_SH = False
                self._maybe_trail(side=-1, swing_bar=i, swing_px=self.h[i])
            if self.l[i] == seg_l.min() and self.l[i] < self.l[i - 1]:
                self.swing_lows.append((i, self.l[i]))
                self.last_SL, self.last_SL_bar = self.l[i], i
                if self.block_long_until_new_SL and i > self.long_exit_bar:
                    self.block_long_until_new_SL = False
                self._maybe_trail(side=+1, swing_bar=i, swing_px=self.l[i])

        # BOS / CHoCH on close, level-based (never wick-based)
        if self.last_SH is not None and self.c[t] > self.last_SH and self.trend <= 0:
            self.trend = +1
        if self.last_SL is not None and self.c[t] < self.last_SL and self.trend >= 0:
            self.trend = -1

    def _maybe_trail(self, side: int, swing_bar: int, swing_px: float):
        """Runner trailing on confirmed structure (STRATEGY §5.1)."""
        p = self.pos
        if p is None or p.phase != "RUNNER" or p.side != side:
            return
        buf = self._buffer(swing_bar)
        if side == +1:  # long trails up under higher swing lows
            new_sl = swing_px - buf
            if new_sl > p.sl:
                p.sl = new_sl
                p.sl_moved = True
        else:           # short trails down over lower swing highs
            new_sl = swing_px + buf
            if new_sl < p.sl:
                p.sl = new_sl
                p.sl_moved = True

    # --- fills / accounting ---------------------------------------------- #
    def _close_units(self, exit_px: float, qty: float, maker: bool):
        p = self.pos
        fee_rate = self.cfg.FEE_MAKER if maker else self.cfg.FEE_TAKER
        gross = (exit_px - p.entry_px) * qty * p.side
        fee = exit_px * qty * fee_rate
        p.realized += gross - fee
        p.qty_rem -= qty

    def _finalize_trade(self, t: int, reason: str):
        p = self.pos
        net = p.realized
        # book PnL to the shared portfolio (updates equity/peak/DD/streak/daily)
        self.pf.book(self.pos_token, net, self.t[t])
        init_risk_dollars = p.qty_full * p.init_risk_px
        self.trades.append({
            "symbol": self.symbol, "side": p.side, "entry_bar": p.entry_bar,
            "exit_bar": t, "entry_ts": to_epoch_ms(self.t[p.entry_bar]),
            "exit_ts": to_epoch_ms(self.t[t]), "entry_px": p.entry_px,
            "qty": p.qty_full, "net_pnl": net,
            "R_multiple": net / init_risk_dollars if init_risk_dollars else 0.0,
            "mfe_R": p.hi_fav / p.init_risk_px if p.init_risk_px else 0.0,
            "mae_R": p.hi_adv / p.init_risk_px if p.init_risk_px else 0.0,
            "mfe_vol": (p.hi_fav / self.unit_range[p.entry_bar]
                        if self.unit_range[p.entry_bar] and not np.isnan(self.unit_range[p.entry_bar])
                        else float("nan")),
            "reason": reason, "equity": self.pf.equity,
        })
        self.pos = None
        self.pos_token = None

    # --- entry order management ------------------------------------------ #
    def _try_fill_entry(self, t: int):
        cfg = self.cfg
        o = self.pending
        if t < o["earliest_fill_bar"]:
            return
        if t > o["valid_until"]:
            self.pending = None  # cancelled, never triggered
            return
        side = o["side"]
        slip = cfg.SLIPPAGE_TICKS * cfg.tick
        if side == +1 and self.h[t] >= o["entry_px"]:
            fill = o["entry_px"] + slip
        elif side == -1 and self.l[t] <= o["entry_px"]:
            fill = o["entry_px"] - slip
        else:
            return  # not triggered this bar

        # portfolio gate: open-position slot available? (concurrency / halt / daily)
        if not self.pf.has_slot():
            self.pending = None
            return

        # size on shared equity (STRATEGY §6.1)
        R = abs(fill - o["sl"])
        if R <= 0:
            self.pending = None
            return
        risk_budget = self.pf.equity * self.pf.eff_risk()
        qty = risk_budget / R
        qty = min(qty, self.pf.equity * cfg.MAX_LEVERAGE / fill)
        new_risk = qty * R
        # portfolio gate: would this breach summed open-risk cap?
        if qty <= 0 or not self.pf.can_add_risk(new_risk):
            self.pending = None
            return

        # entry taker fee
        entry_fee = fill * qty * cfg.FEE_TAKER
        self.pos = Position(
            side=side, entry_bar=t, entry_px=fill, qty_full=qty, qty_rem=qty,
            sl=o["sl"], init_risk_px=R, tp=o["tp"],
        )
        self.pos.realized -= entry_fee
        self.pos_token = self.pf.register(new_risk)
        self.pending = None
        # an entry-stop and its SL can both sit inside the same bar:
        # invariant says assume entry THEN SL (worst case) -> manage now.
        self._manage_position(t, just_entered=True)

    # --- open position management ---------------------------------------- #
    def _manage_position(self, t: int, just_entered: bool = False):
        cfg = self.cfg
        p = self.pos
        if p is None:
            return

        # track max favorable / adverse excursion (MFE/MAE), every open bar
        if p.side == +1:
            p.hi_fav = max(p.hi_fav, self.h[t] - p.entry_px)
            p.hi_adv = max(p.hi_adv, p.entry_px - self.l[t])
        else:
            p.hi_fav = max(p.hi_fav, p.entry_px - self.l[t])
            p.hi_adv = max(p.hi_adv, self.h[t] - p.entry_px)

        # pending market exit (time stop) executes at this bar's open
        if p.pending_market_exit:
            slip = cfg.SLIPPAGE_TICKS * cfg.tick
            px = self.o[t] - slip * p.side
            self._close_units(px, p.qty_rem, maker=False)
            self._register_stop_exit(t, "time_stop")
            return

        sl_at_start = p.sl
        slip = cfg.SLIPPAGE_TICKS * cfg.tick

        if p.side == +1:
            sl_touched = self.l[t] <= sl_at_start
            tp_touchable = any(not p.tp_filled[k] and self.h[t] >= p.tp[k] for k in range(3))
        else:
            sl_touched = self.h[t] >= sl_at_start
            tp_touchable = any(not p.tp_filled[k] and self.l[t] <= p.tp[k] for k in range(3))

        # conservative: if SL and any TP both reachable this bar -> SL first
        if sl_touched and (tp_touchable or True is False):
            pass  # fall through to unified handling below
        if sl_touched:
            exit_px = sl_at_start - slip * p.side  # stop slips against us
            self._close_units(exit_px, p.qty_rem, maker=False)
            self._register_stop_exit(t, "trail_stop" if p.sl_moved else "stop_loss")
            return

        # SL not touched -> process take-profit ladder in order
        split = cfg.TP_SPLIT
        if p.side == +1:
            for k in range(3):
                if not p.tp_filled[k] and self.h[t] >= p.tp[k]:
                    self._close_units(p.tp[k], p.qty_full * split[k], maker=True)
                    p.tp_filled[k] = True
                    self._on_tp_fill(k)
        else:
            for k in range(3):
                if not p.tp_filled[k] and self.l[t] <= p.tp[k]:
                    self._close_units(p.tp[k], p.qty_full * split[k], maker=True)
                    p.tp_filled[k] = True
                    self._on_tp_fill(k)

        # a TP fill may have raised the stop into this same bar's range
        if p.qty_rem > 1e-12:
            if (p.side == +1 and self.l[t] <= p.sl) or (p.side == -1 and self.h[t] >= p.sl):
                exit_px = p.sl - slip * p.side
                self._close_units(exit_px, p.qty_rem, maker=False)
                self._register_stop_exit(t, "trail_stop")
                return

        # runner exit on opposing BOS (close-based)
        if p.phase == "RUNNER" and p.qty_rem > 1e-12:
            if (p.side == +1 and self.last_SL is not None and self.c[t] < self.last_SL) or \
               (p.side == -1 and self.last_SH is not None and self.c[t] > self.last_SH):
                self._close_units(self.c[t], p.qty_rem, maker=False)
                self._register_stop_exit(t, "bos_exit")
                return

        # time stop: not at TP1 within window -> exit at NEXT bar open
        if p.qty_rem > 1e-12 and not p.tp_filled[0]:
            if t >= p.entry_bar + cfg.TIME_STOP_BARS:
                p.pending_market_exit = True

        # fully closed by TPs (e.g., runner is 0% in a custom split)
        if p.qty_rem <= 1e-12:
            self._finalize_trade(t, "take_profit")

    def _on_tp_fill(self, k: int):
        p = self.pos
        if k == 0:                    # TP1 -> stop to breakeven
            p.sl = max(p.sl, p.entry_px) if p.side == +1 else min(p.sl, p.entry_px)
        elif k == 1:                  # TP2 -> stop to TP1 price (lock +1R)
            p.sl = p.tp[0]
        elif k == 2:                  # TP3 -> stop to TP2 price, runner phase
            p.sl = p.tp[1]
            p.phase = "RUNNER"
        p.sl_moved = True

    def _register_stop_exit(self, t: int, reason: str):
        p = self.pos
        # set direction cooldown before finalizing (which nulls self.pos)
        if reason in ("stop_loss", "trail_stop", "time_stop", "bos_exit"):
            if p.side == +1:
                self.block_long_until_new_SL = True
                self.long_exit_bar = t
            else:
                self.block_short_until_new_SH = True
                self.short_exit_bar = t
        self._finalize_trade(t, reason)

    # --- signal generation (evaluated on close[t]) ----------------------- #
    def _evaluate_signal(self, t: int):
        cfg = self.cfg
        if self.pos is not None or self.pending is not None:
            return
        if self.pf.halted or self.pf.daily_blocked:
            return
        if np.isnan(self.unit_range[t]) or np.isnan(self.flow_unit[t]):
            return
        if self.rng[t] <= 0 or self.flow_unit[t] <= 0 or self.unit_range[t] <= 0:
            return

        rng = self.rng[t]
        vol_ok = rng >= cfg.MIN_RANGE_FRAC * self.unit_range[t]
        if not vol_ok:
            return

        # ---- LONG sweep of swing low ----
        if self.last_SL is not None and not self.block_long_until_new_SL:
            S1 = self.l[t] < self.last_SL
            S2 = self.c[t] > self.last_SL
            S3 = (self.c[t] - self.l[t]) >= cfg.REJ_FRAC * rng
            # S4: sweep-bar flow not strongly opposing + CVD absorption (price
            # made a lower low than the swept pivot, CVD did not).
            opp = self.delta[t] >= -cfg.OPP_K * self.flow_unit[t]
            div = (not cfg.USE_CVD_DIV) or (self.cvd[t] >= self.cvd[self.last_SL_bar])
            S4 = opp and div
            S6 = (t - self.last_SL_bar) <= cfg.LEVEL_MAX_AGE
            if S1 and S2 and S3 and S4 and S6:
                self._place_entry(t, side=+1)
                return

        # ---- SHORT sweep of swing high ----
        if self.last_SH is not None and not self.block_short_until_new_SH:
            S1 = self.h[t] > self.last_SH
            S2 = self.c[t] < self.last_SH
            S3 = (self.h[t] - self.c[t]) >= cfg.REJ_FRAC * rng
            # S4 mirror: flow not strongly opposing + bearish CVD divergence
            opp = self.delta[t] <= cfg.OPP_K * self.flow_unit[t]
            div = (not cfg.USE_CVD_DIV) or (self.cvd[t] <= self.cvd[self.last_SH_bar])
            S4 = opp and div
            S6 = (t - self.last_SH_bar) <= cfg.LEVEL_MAX_AGE
            if S1 and S2 and S3 and S4 and S6:
                self._place_entry(t, side=-1)
                return

    def _place_entry(self, t: int, side: int):
        cfg = self.cfg
        buf = self._buffer(t)
        if side == +1:
            entry_px = self.h[t] + cfg.tick
            sl = self.l[t] - buf
            R = entry_px - sl
            tp1 = entry_px + cfg.TP1_R * R
            tp2 = entry_px + cfg.TP2_R * R
            tp3 = entry_px + cfg.TP3_R * R
            # snap TP3 to nearest confirmed swing high inside (TP2, TP3] band
            band = [px for (_, px) in self.swing_highs if tp2 < px <= tp3]
            if band:
                tp3 = min(band)
        else:
            entry_px = self.l[t] - cfg.tick
            sl = self.h[t] + buf
            R = sl - entry_px
            tp1 = entry_px - cfg.TP1_R * R
            tp2 = entry_px - cfg.TP2_R * R
            tp3 = entry_px - cfg.TP3_R * R
            band = [px for (_, px) in self.swing_lows if tp3 <= px < tp2]
            if band:
                tp3 = max(band)

        self.pending = {
            "side": side, "entry_px": entry_px, "sl": sl,
            "tp": [tp1, tp2, tp3], "signal_bar": t,
            "earliest_fill_bar": t + 1,
            "valid_until": t + cfg.ENTRY_VALID_BARS,
        }

    # --- per-bar step (shared by single-symbol run and PortfolioBacktester) - #
    def step(self, t: int):
        # 1) structure (data <= t), may also trail an open runner
        self._update_structure(t)
        # 2) manage resting entry orders against bar t (>= earliest_fill_bar)
        if self.pending is not None:
            self._try_fill_entry(t)
        # 3) manage an open position against bar t
        if self.pos is not None:
            self._manage_position(t)
        # 4) evaluate a NEW signal on close[t] -> order fillable at t+1
        self._evaluate_signal(t)

    def close_open_at_end(self):
        if self.pos is not None:
            self._close_units(self.c[self.n - 1], self.pos.qty_rem, maker=False)
            self._finalize_trade(self.n - 1, "eod_close")

    # --- main loop (single symbol) --------------------------------------- #
    def run(self) -> dict:
        for t in range(self.n):
            self.pf.on_time(self.t[t])   # roll UTC day for the daily loss limit
            self.step(t)
        self.close_open_at_end()
        return self.report()

    # --- metrics ---------------------------------------------------------- #
    def report(self) -> dict:
        tr = pd.DataFrame(self.trades)
        if tr.empty:
            return {"trades": 0, "summary": "no trades"}
        wins = tr[tr.net_pnl > 0]
        losses = tr[tr.net_pnl <= 0]
        gross_win = wins.net_pnl.sum()
        gross_loss = -losses.net_pnl.sum()
        eq = pd.Series([self.cfg.initial_equity] + tr.equity.tolist())
        peak = eq.cummax()
        max_dd = ((peak - eq) / peak).max()
        return {
            "trades": len(tr),
            "win_rate": round(len(wins) / len(tr), 4),
            "total_return_pct": round((self.pf.equity / self.cfg.initial_equity - 1) * 100, 3),
            "final_equity": round(self.pf.equity, 2),
            "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else float("inf"),
            "avg_R": round(tr.R_multiple.mean(), 3),
            "expectancy_R": round(tr.R_multiple.mean(), 3),
            "max_drawdown_pct": round(max_dd * 100, 3),
            "exit_reasons": tr.reason.value_counts().to_dict(),
            "trade_log": tr,
        }


# --------------------------------------------------------------------------- #
# Data utilities                                                               #
# --------------------------------------------------------------------------- #
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume", "buy_vol", "sell_vol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


def generate_synthetic(n: int = 4000, seed: int = 7) -> pd.DataFrame:
    """Synthetic OHLCV + aggressor flow with deliberate sweep-and-reverse
    structure so the engine has something to trade. For demonstration only —
    NOT a substitute for real exchange data."""
    rng = np.random.default_rng(seed)
    price = 30_000.0
    rows = []
    ts0 = pd.Timestamp("2024-01-01", tz="UTC")
    trend = 0.0
    for i in range(n):
        trend = 0.97 * trend + rng.normal(0, 0.6)
        drift = trend * 4.0
        o = price
        body = drift + rng.normal(0, 18)
        c = o + body
        wick = abs(rng.normal(0, 22))
        h = max(o, c) + wick
        l = min(o, c) - wick
        # occasional liquidity sweep: long lower wick that reclaims
        if rng.random() < 0.04:
            l -= abs(rng.normal(0, 45))
            c = max(o, c) - rng.random() * (max(o, c) - min(o, c)) * 0.2
        vol = abs(rng.normal(120, 40)) + 10
        # aggressor split correlated with candle direction + absorption noise
        bias = 0.5 + np.clip(body / 120, -0.35, 0.35) + rng.normal(0, 0.08)
        bias = float(np.clip(bias, 0.05, 0.95))
        buy = vol * bias
        sell = vol * (1 - bias)
        rows.append((ts0 + pd.Timedelta(minutes=15 * i), o, h, l, c, vol, buy, sell))
        price = c
    return pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "buy_vol", "sell_vol"])


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="LSOF reference backtest")
    ap.add_argument("--csv", help="path to OHLCV+flow CSV")
    ap.add_argument("--demo", action="store_true", help="run on synthetic data")
    ap.add_argument("--tick", type=float, default=0.1)
    ap.add_argument("--equity", type=float, default=10_000.0)
    args = ap.parse_args()

    if args.csv:
        df = load_csv(args.csv)
    else:
        if not args.demo:
            print("No --csv given; running --demo on synthetic data.\n")
        df = generate_synthetic()

    cfg = Config(tick=args.tick, initial_equity=args.equity)
    bt = Backtester(df, cfg)
    res = bt.run()

    print("=" * 60)
    print("LSOF backtest result")
    print("=" * 60)
    for k, v in res.items():
        if k == "trade_log":
            continue
        print(f"{k:>20}: {v}")
    if "trade_log" in res:
        print("\nLast 10 trades:")
        cols = ["side", "entry_bar", "exit_bar", "net_pnl", "R_multiple", "reason"]
        print(res["trade_log"][cols].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
