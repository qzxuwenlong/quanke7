# LSOF — Liquidity-Sweep / Order-Flow Reversal

A fully mechanical crypto strategy using **only** naked candlestick price action
and order-flow (aggressor/delta) data. No moving averages, RSI, MACD, Bollinger,
stochastics, or any lagging/smoothed indicator. Every rule is objective and the
backtest cannot repaint.

- **`STRATEGY.md`** — the complete, programmable specification (entry, stop,
  multi-level TP, risk management, no-repaint rules, parameter table).
- **`strategy.py`** — event-driven reference backtest engine implementing that
  spec exactly, with no lookahead.
- **`okx_fetch.py`** — pulls OKX OHLC + order flow and writes a ready-to-use CSV.

## Quickstart

```bash
# 1) fetch OKX data (OHLC + per-candle taker buy/sell volume)
py okx_fetch.py --inst BTC-USDT-SWAP --bar 15m \
    --start 2024-01-01 --end 2024-03-01 --out data.csv

# 2) backtest on it (use the instrument's real tick size)
py strategy.py --csv data.csv --tick 0.1 --equity 10000

# no data / offline? self-contained demo on synthetic data:
py strategy.py --demo          # Windows ('python3 ...' elsewhere)
```

Requires `numpy` and `pandas` (`pip install numpy pandas`). `okx_fetch.py` uses
only the standard library for HTTP — no `requests` needed.

## Input data schema

One row per **closed** candle, ascending by time:

| column | meaning |
|---|---|
| `timestamp` | ISO string or epoch-ms |
| `open high low close` | OHLC |
| `volume` | total base volume |
| `buy_vol` | taker-buy (aggressor-buy) volume |
| `sell_vol` | taker-sell (aggressor-sell) volume |
| `oi` | open interest (optional, unused by default) |

### Sourcing data from OKX (`okx_fetch.py`)

OKX V5 public endpoints (no API key) provide everything the strategy needs:

- **OHLC** — `GET /api/v5/market/history-candles`. The fetcher keeps only bars
  with `confirm == "1"` (closed), so the still-forming candle never enters the
  dataset — no-repaint starts at the data layer.
- **Order flow** — two modes via `--flow`:
  - **`rubik`** (default) — `GET /api/v5/rubik/stat/taker-volume` returns
    `[ts, sellVol, buyVol]`, **currency-level** taker volume, resampled up into
    your bar. Lightweight and spans history, so it's the practical backtest
    source. The engine derives `delta = buy_vol - sell_vol` and `cvd = cumsum`.
  - **`tape`** — aggregates per-trade `history-trades` (each trade's `side` is
    the taker side). Exact and per-instrument, but ~1M+ records per 15 min on a
    liquid perp, so only feasible for very short, recent windows.
  - `none` (or `--no-flow`) — OHLC only, for a quick smoke test.

Practical notes / limits:

- **Region & proxy:** OKX blocks some IPs (e.g. mainland China). Use
  `--proxy http://127.0.0.1:7890` (or your port), or `--base https://aws.okx.com`.
- **Flow retention (rubik):** 5m ≈ 48h, **1H ≈ 30 days**, 1D longer. So a
  multi-day/-week backtest should use `--bar 1H` (or coarser). Sub-5m bars need
  `--flow tape`. Requests beyond retention degrade to partial coverage + warning.
- **Currency-level S4:** because rubik flow is per-currency (not per-instrument),
  the strategy's S4 gate is the **non-opposing + CVD-divergence** form (`OPP_K`,
  `USE_CVD_DIV`), not strong same-bar confirmation — see `STRATEGY.md` §3.
- **Instrument id:** perps `BTC-USDT-SWAP`, spot `BTC-USDT`; pass the matching
  **tick size** to `strategy.py --tick` (BTC-USDT-SWAP = `0.1`).
- **Sample size:** ~30 days of 1H ≈ a handful of trades — far too few to judge
  edge. Aggregate multiple symbols / accumulate history before drawing
  conclusions, and walk-forward per `STRATEGY.md` §7.

## What the demo proves (and doesn't)

The synthetic demo is a **random walk**, which by construction has no edge, so a
negative result there is correct behavior — it shows the engine runs end-to-end,
is deterministic, books partial take-profits, and that the **risk controls fire**
(the run halts at ≈ −10%, the configured `MAX_DD_HALT`). It does **not** claim
profitability. Validate edge only on real exchange data with walk-forward
out-of-sample testing per `STRATEGY.md` §7.

## No-repaint guarantees (enforced in code)

- Every decision at bar `t` uses data `<= t`; trailing stats use `.shift(1)`.
- Swing pivots are confirmed `P` bars late and acted on only at confirmation.
- A signal on bar `t` places a resting order fillable no earlier than `t+1`.
- Same-bar stop/target ambiguity resolves to the adverse side (never overstates
  fills). Fees, slippage, and (for perps) funding are modeled.
