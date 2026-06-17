# Liquidity-Sweep / Order-Flow Reversal Strategy (LSOF)

A fully mechanical crypto-futures strategy built **only** from naked candlestick
price action and order-flow (aggressor / delta) data. No moving averages, RSI,
MACD, Bollinger, stochastics, or any smoothed/lagging indicator is used anywhere
in signal generation. Every rule below is a deterministic function of data that
is **fully closed and known at decision time**, so backtests cannot repaint.

---

## 0. Design rules (non-negotiable invariants)

1. **Closed-bar only.** Every condition is evaluated on the *close* of a
   completed candle `t`, using data from candles `<= t` only.
2. **No centered windows.** All rolling statistics are trailing. The only
   forward-looking construct is a swing pivot, which is *confirmed* `R` bars
   after it forms and is **acted on only at the confirmation bar** (never
   back-dated to the pivot bar). See §2.
3. **Resting orders, realistic fills.** Entries and stops are resting
   stop/limit orders. A level is "hit" only if a *later* candle's high/low
   crosses it. Intrabar ambiguity is resolved conservatively (§7).
4. **No indicator may gate a trade.** The only volatility statistic used
   (`unit_range`, §1) is a raw trailing median of candle ranges, used solely
   for buffers/filters — never as a directional signal.
5. **Symmetric.** Long and short logic are exact mirror images.
6. **No discretion.** There is no "if it looks like..." anywhere. Every term
   below is a number or a boolean derived from numbers.

---

## 1. Data inputs & primitive definitions

Per symbol, fixed timeframe (default **15m** for entries; works on any TF).
Each candle `t` provides:

| Field | Meaning |
|---|---|
| `open[t] high[t] low[t] close[t]` | OHLC |
| `vol[t]` | total traded base volume |
| `buy_vol[t]` | taker-buy (aggressor-buy) volume |
| `sell_vol[t]` | taker-sell (aggressor-sell) volume |
| `oi[t]` | open interest at close (optional) |

**Order-flow primitives** (exchange-objective; from OKX `history-trades`, each
trade's `side` is the taker/aggressor side: `side=="buy"` ⇒ taker-buy volume,
`side=="sell"` ⇒ taker-sell volume — see `okx_fetch.py`):

```
delta[t]   = buy_vol[t] - sell_vol[t]          # per-candle net aggressor flow
cvd[t]     = cvd[t-1] + delta[t]               # cumulative volume delta
range[t]   = high[t] - low[t]
body[t]    = abs(close[t] - open[t])
```

**Volatility unit** (raw statistic, not an indicator):

```
unit_range = median( range[t-N_VOL .. t-1] )    # trailing, excludes current bar
```

`N_VOL = 50`. Used only for buffers and the minimum-volatility filter.

**Relative flow strength** (trailing, normalizes delta across regimes):

```
flow_unit  = median( abs(delta[t-N_VOL .. t-1]) )
```

---

## 2. Market structure (objective, non-repainting)

**Swing pivot (fractal of strength `P`, default `P = 2`):**

- Candle `i` is a **swing high** if `high[i] = max(high[i-P .. i+P])`
  *and* `high[i] > high[i-1]` (strict on the immediate left).
- Candle `i` is a **swing low** if `low[i] = min(low[i-P .. i+P])`
  *and* `low[i] < low[i-1]`.

A pivot at `i` becomes **confirmed at bar `i+P`** (that is the first bar at
which all `P` right-side candles have closed). The engine only ever references
*confirmed* pivots, and only from bar `i+P` onward. This is the single place a
look-forward exists, and it is handled correctly: in live trading you simply
cannot know a pivot until `P` bars later, and the backtest enforces the same.

State carried forward:

```
last_SH        = price of most recent confirmed swing high
last_SH_bar    = its bar index
last_SL        = price of most recent confirmed swing low
last_SL_bar    = its bar index
trend          = +1 / -1 / 0   (see CHoCH below)
```

**Break of structure / CHoCH** (close-based, never wick-based):

```
bull_BOS = close[t] > last_SH
bear_BOS = close[t] < last_SL
```

`trend` flips to `+1` on the first `bull_BOS` after a `-1` regime (bullish
CHoCH) and to `-1` on the first `bear_BOS` after `+1`.

---

## 3. Entry signal

The setup is a **liquidity sweep that is rejected and confirmed by order flow**,
i.e. price runs resting stops beyond a confirmed swing, fails to hold, and
aggressor flow shows absorption in the opposite direction.

**S4 flow note.** S4 is calibrated for the flow source. With *per-instrument*
delta, "strong confirming aggression" is appropriate. With *currency-level* flow
(OKX Rubik `taker-volume`, the only multi-day free source — see `okx_fetch.py`),
per-candle sign is loosely coupled to one instrument, so S4 instead requires the
sweep bar's flow to be **non-opposing** plus a **CVD-absorption divergence**.
Both forms are objective and non-repainting; switch via `USE_CVD_DIV` / `OPP_K`.

### 3.1 Long setup (mirror for short)

All conditions evaluated at the close of candle `t`:

```
S1  sweep:        low[t]  < last_SL          (wick pierces swept liquidity)
S2  reclaim:      close[t] > last_SL          (closes back above it)
S3  rejection:    (close[t] - low[t]) >= REJ_FRAC * range[t]      # lower wick
S4  flow:         delta[t] >= -OPP_K * flow_unit                  # not opposing
      AND (if USE_CVD_DIV)  cvd[t] >= cvd[last_SL_bar]            # CVD absorption
S5  volatility:   range[t] >= MIN_RANGE_FRAC * unit_range
S6  freshness:    (t - last_SL_bar) <= LEVEL_MAX_AGE             # level not stale
S7  not in trade on this symbol AND cooldown elapsed (§6)
```

S4 rationale: a long after a swept low is invalidated if aggressors are still
strongly selling the breakdown (`delta` very negative). The CVD term encodes
classic absorption — price printed a *lower low* than the swept pivot, but
cumulative delta did **not** make a lower low, so sellers failed to follow
through. `cvd[last_SL_bar]` references a confirmed past pivot → no repaint.

If `S1..S7` all true, candle `t` is the **signal bar**. Place a resting
**buy-stop** order:

```
entry_px   = high[t] + TICK
entry_valid_until = t + ENTRY_VALID_BARS
```

Entry **fills** when a later candle's `high >= entry_px` within the validity
window; otherwise the order is cancelled (no entry). Requiring the break of the
signal-bar high filters sweeps that never resume — it is a real, objective
trigger, not a guess.

### 3.2 Short setup (exact mirror)

```
S1  high[t] > last_SH
S2  close[t] < last_SH
S3  (high[t] - close[t]) >= REJ_FRAC * range[t]        # upper wick
S4  delta[t] <= OPP_K * flow_unit                      # not opposing
      AND (if USE_CVD_DIV)  cvd[t] <= cvd[last_SH_bar]  # bearish CVD divergence
S5  range[t] >= MIN_RANGE_FRAC * unit_range
S6  (t - last_SH_bar) <= LEVEL_MAX_AGE
S7  flat + cooldown elapsed
entry_px = low[t] - TICK     # resting sell-stop
```

---

## 4. Stop loss (structural, fixed at entry)

The invalidation is "price reclaimed the swept liquidity in the wrong
direction" — a clean, objective level.

**Long:**
```
sweep_low = low[signal_bar]
SL = sweep_low - SL_BUFFER
SL_BUFFER = max( SL_TICKS * TICK , SL_RANGE_FRAC * range[signal_bar] )
```

**Short:**
```
sweep_high = high[signal_bar]
SL = sweep_high + SL_BUFFER
```

`SL` is set **once** at entry and is only ever moved *toward profit* by the
take-profit ladder (§5) — never widened. Initial risk:

```
R = abs(entry_px - SL)     # per-unit risk distance
```

---

## 5. Multi-level take profit (R-multiples + structural target)

Risk distance `R` from §4 anchors the ladder. Default allocation of position
size `Q`:

| Level | Trigger price (long / short)         | Size closed | Action on fill |
|------:|--------------------------------------|:-----------:|----------------|
| TP1   | entry ± `TP1_R`·R  (1.0R)            | 40% of `Q`  | move SL → entry (breakeven) |
| TP2   | entry ± `TP2_R`·R  (2.0R)            | 30% of `Q`  | move SL → TP1 price (lock +1R) |
| TP3   | nearest untapped opposing swing, capped at entry ± `TP3_R`·R (3.0R) | 20% of `Q` | move SL → TP2 price |
| Runner| — (10% of `Q`)                       | trail (§5.1) | exit on opposing BOS |

- **TP3 structural target (long):** the lowest confirmed swing high strictly
  above `entry_px` that has *not* yet been traded through ("untapped"). If none
  exists within `entry_px + TP3_R·R`, use `entry_px + TP3_R·R`. (Mirror for
  short: highest untapped swing low below entry.) This is fully objective — it
  reads the confirmed-pivot list only.
- All TP levels are resting limit orders placed at entry; partial fills follow
  the same "later candle crosses level" rule.

### 5.1 Runner trailing (structure-based, no indicator)

After TP3, the 10% runner trails on **confirmed swing structure**:

- **Long:** each time a new *higher* confirmed swing low (`HL`) prints, move SL
  to `HL - SL_BUFFER` (only upward). Exit the runner on `bear_BOS`
  (`close < last_SL`) or if the trailing SL is hit.
- **Short:** mirror with lower confirmed swing highs and `bull_BOS`.

---

## 6. Risk management

### 6.1 Position sizing (risk-based, fixed-fractional)
```
risk_budget   = equity * RISK_PER_TRADE          # default 0.5%
Q (base units)= risk_budget / R
notional      = Q * entry_px
Q             = min(Q, (equity * MAX_LEVERAGE) / entry_px)   # leverage cap
```
Risk per trade is the loss incurred if the *full* position hits the initial SL,
before any partial is taken. Sizing never depends on conviction.

### 6.2 Portfolio caps
- `MAX_CONCURRENT` open positions (default 3).
- `MAX_AGGREGATE_RISK` = 2.0% of equity summed across all open initial risks;
  a new signal is skipped if it would breach this.
- **Correlation guard:** no two simultaneous positions in the same direction on
  symbols in the same `CORR_GROUP` (e.g. BTC & ETH) — count as one risk slot.

### 6.3 Drawdown / loss controls (kill switches)
- **Daily loss limit:** if realized PnL for the UTC day `<= -DAILY_LOSS_LIMIT`
  (default 2% of day-start equity), close nothing already open by force but
  **take no new entries** until the next UTC day.
- **Consecutive-loss de-risking:** after `LOSS_STREAK_TRIGGER` (default 3)
  consecutive losing trades, halve `RISK_PER_TRADE` until one winner, then
  restore.
- **Max drawdown halt:** if equity drawdown from peak `>= MAX_DD_HALT`
  (default 10%), stop all new entries and flag for manual review.
- **No averaging down. No adding to losers. No moving SL away from price. Ever.**

### 6.4 Time stop (objective decay control)
If a trade has not reached TP1 within `TIME_STOP_BARS` (default 12 bars) of
fill, close the remaining position at the next candle's open. Prevents capital
sitting in a stalled setup whose edge has expired.

### 6.5 Cooldown
After any SL/time-stop exit on a symbol, take no new entry on that symbol until
a **new** confirmed swing (in the trade direction) forms — prevents re-entering
the same dead level.

---

## 7. No-repaint & backtest integrity

1. **Bar-close evaluation only**; signals computed with data `<= t`.
2. **Pivot confirmation lag** `P` enforced (§2): a pivot is invisible until
   `bar = pivot_bar + P`.
3. **Intrabar fill priority (conservative):** if within the same candle both an
   adverse stop and a favorable TP are touchable, assume the **stop fills
   first**. If both an entry-stop and its SL are touchable in the same candle,
   assume entry then SL (worst case). Rationale: never overstate fills.
4. **Fees & slippage:** apply `FEE_TAKER` on entries and stop exits (taker),
   `FEE_MAKER` on limit TP fills, plus `SLIPPAGE_TICKS` on stop orders.
5. **Funding:** for perpetuals, apply funding at each funding timestamp to open
   notional.
6. **No parameter peeking:** parameters fixed before the test window. Use
   walk-forward: optimize on in-sample, report only out-of-sample.
7. **Single source of truth for price:** all crossings use candle
   `high/low/close`; no external mark price.
8. **Deterministic:** given the same data + parameters, the engine produces the
   identical trade list every run.

---

## 8. Parameters (single config block)

| Param | Default | Notes |
|---|---|---|
| `TIMEFRAME` | `15m` | entry timeframe |
| `P` (pivot strength) | `2` | swing fractal half-width |
| `N_VOL` | `50` | trailing window for `unit_range`, `flow_unit` |
| `REJ_FRAC` | `0.5` | min rejection-wick fraction of bar range |
| `OPP_K` | `0.5` | S4: sweep-bar delta may oppose by at most `OPP_K × flow_unit` |
| `USE_CVD_DIV` | `true` | S4: also require CVD absorption vs the swept pivot |
| `MIN_RANGE_FRAC` | `0.75` | signal bar range vs `unit_range` |
| `LEVEL_MAX_AGE` | `60` | bars; swept level must be fresher than this |
| `ENTRY_VALID_BARS` | `3` | entry-stop validity |
| `SL_TICKS` | `2` | min stop buffer in ticks |
| `SL_RANGE_FRAC` | `0.10` | stop buffer as fraction of signal-bar range |
| `TP1_R / TP2_R / TP3_R` | `1.0 / 2.0 / 3.0` | R-multiples |
| TP size split | `40/30/20/10` | %, sums to 100 |
| `RISK_PER_TRADE` | `0.005` | 0.5% equity |
| `MAX_LEVERAGE` | `5` | notional cap |
| `MAX_CONCURRENT` | `3` | open positions |
| `MAX_AGGREGATE_RISK` | `0.02` | 2% summed open risk |
| `DAILY_LOSS_LIMIT` | `0.02` | 2% day-start equity |
| `LOSS_STREAK_TRIGGER` | `3` | halve risk after N losers |
| `MAX_DD_HALT` | `0.10` | 10% peak-to-trough halt |
| `TIME_STOP_BARS` | `12` | bars to reach TP1 |
| `FEE_TAKER / FEE_MAKER` | `0.0005 / 0.0002` | per side |
| `SLIPPAGE_TICKS` | `1` | on stop fills |

---

## 9. Trade lifecycle (state machine)

```
FLAT
  │  S1..S7 true on bar t  → place buy/sell-stop, store SL/TP plan
  ▼
PENDING  ── entry not filled within ENTRY_VALID_BARS ──► FLAT (cancel)
  │  entry_px crossed
  ▼
OPEN_FULL
  │  TP1 hit → close 40%, SL→BE          ┐
  │  TP2 hit → close 30%, SL→TP1         │ each transition objective,
  │  TP3 hit → close 20%, SL→TP2         │ driven by later-bar crossings
  │  SL hit  → close all → FLAT(cooldown)│
  │  time-stop → close all → FLAT        ┘
  ▼
RUNNER (10%)
  │  new confirmed HL/LH → trail SL
  │  opposing BOS OR trailing SL hit → close → FLAT(cooldown)
  ▼
FLAT
```

See `strategy.py` for the executable reference implementation of this exact
machine.
