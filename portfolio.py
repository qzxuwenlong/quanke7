"""
Multi-symbol portfolio backtest for the LSOF strategy.

Runs one per-symbol Backtester for every matched instrument, all sharing a
single Portfolio so the portfolio-level risk caps apply across the whole book:
  * MAX_CONCURRENT open positions (across all symbols)
  * MAX_AGGREGATE_RISK summed open initial-risk
  * DAILY_LOSS_LIMIT, MAX_DD_HALT, loss-streak de-risking — all portfolio-wide

The symbols are stepped on a single MERGED timeline (sorted by timestamp, ties
broken by symbol name) so causality and the shared caps are correct: equity at
any bar reflects every close up to that instant, never the future.

Reads the screener output (universe.csv, match==True) for the instrument list
and tick sizes, fetches 1H + Rubik flow per symbol (cached to ./cache), and
prints pooled stats + a per-symbol breakdown + an in-sample/out-of-sample split.

Usage:
  py portfolio.py --proxy http://127.0.0.1:7890 --start 2026-05-17 --end 2026-06-16
"""
from __future__ import annotations

import argparse
import os
from dataclasses import replace

import pandas as pd

import okx_fetch
from strategy import Backtester, Config, Portfolio, load_csv


class PortfolioBacktester:
    def __init__(self, data: dict[str, tuple[pd.DataFrame, float]], cfg: Config,
                 bt_class=Backtester):
        self.cfg = cfg
        self.pf = Portfolio(cfg)
        self.bts: dict[str, Backtester] = {}
        for sym, (df, tick) in data.items():
            self.bts[sym] = bt_class(df, replace(cfg, tick=tick),
                                     portfolio=self.pf, symbol=sym)

    def run(self) -> dict:
        # merged timeline: (ts, symbol, local_bar_index), sorted causally
        events = []
        for sym, bt in self.bts.items():
            for i in range(bt.n):
                events.append((int(bt.t[i]), sym, i))
        events.sort(key=lambda e: (e[0], e[1]))

        last_ts = None
        for ts, sym, i in events:
            if ts != last_ts:
                self.pf.on_time(ts)      # roll UTC day once per timestamp
                last_ts = ts
            self.bts[sym].step(i)

        for bt in self.bts.values():     # mark-to-close anything still open
            bt.close_open_at_end()
        return self.report()

    def report(self) -> dict:
        frames = [pd.DataFrame(bt.trades) for bt in self.bts.values() if bt.trades]
        if not frames:
            return {"trades": 0, "summary": "no trades"}
        tr = pd.concat(frames, ignore_index=True).sort_values("exit_ts").reset_index(drop=True)

        def stats(d: pd.DataFrame) -> dict:
            if d.empty:
                return dict(trades=0, win_rate=0.0, pf=0.0, expR=0.0, pnl=0.0)
            gl = float(-d.net_pnl[d.net_pnl <= 0].sum())
            return dict(
                trades=int(len(d)),
                win_rate=round(float((d.net_pnl > 0).mean()), 3),
                pf=round(float(d.net_pnl[d.net_pnl > 0].sum()) / gl, 3) if gl > 0 else float("inf"),
                expR=round(float(d.R_multiple.mean()), 3),
                pnl=round(float(d.net_pnl.sum()), 2),
            )

        # portfolio equity curve -> max drawdown
        eqc = pd.DataFrame(self.pf.equity_curve, columns=["ts", "equity"]).sort_values("ts")
        eq = pd.Series([self.cfg.initial_equity] + eqc.equity.tolist())
        max_dd = ((eq.cummax() - eq) / eq.cummax()).max()

        # in-sample / out-of-sample split at the median exit time
        cut = tr.exit_ts.median()
        is_, oos = tr[tr.exit_ts <= cut], tr[tr.exit_ts > cut]

        rows = []
        for sym, d in tr.groupby("symbol"):
            s = stats(d)
            s["symbol"] = sym
            rows.append(s)
        per_sym = pd.DataFrame(rows).set_index("symbol").sort_values("pnl", ascending=False)

        return {
            "symbols": len(self.bts),
            "overall": stats(tr),
            "total_return_pct": round((self.pf.equity / self.cfg.initial_equity - 1) * 100, 3),
            "final_equity": round(self.pf.equity, 2),
            "max_drawdown_pct": round(max_dd * 100, 3),
            "halted": self.pf.halted,
            "exit_reasons": tr.reason.value_counts().to_dict(),
            "in_sample": stats(is_),
            "out_of_sample": stats(oos),
            "per_symbol": per_sym,
            "trade_log": tr,
        }


def fetch_cached(inst, bar, start_ms, end_ms, proxy, base, cache_dir) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{inst}_{bar}_{start_ms}_{end_ms}.csv")
    if os.path.exists(path):
        return load_csv(path)
    df = okx_fetch.build(inst, bar, start_ms, end_ms, base, "rubik", proxy)
    df.to_csv(path, index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description="LSOF multi-symbol portfolio backtest")
    ap.add_argument("--universe", default="universe.csv", help="screener output")
    ap.add_argument("--bar", default="1H")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--base", default="https://www.okx.com")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--equity", type=float, default=10_000.0)
    ap.add_argument("--top", type=int, default=0, help="limit to top-N matched by turnover")
    args = ap.parse_args()

    uni = pd.read_csv(args.universe)
    uni = uni[uni.match].sort_values("turnover_usd", ascending=False)
    if args.top:
        uni = uni.head(args.top)
    print(f"matched universe: {len(uni)} symbols -> {list(uni.instId)}")

    start_ms, end_ms = okx_fetch.to_ms(args.start), okx_fetch.to_ms(args.end)
    data = {}
    for _, row in uni.iterrows():
        inst, tick = row.instId, float(row.tickSz)
        try:
            df = fetch_cached(inst, args.bar, start_ms, end_ms, args.proxy, args.base, args.cache)
            if len(df) >= 60:                      # need warmup + signal room
                data[inst] = (df, tick)
            else:
                print(f"  skip {inst}: only {len(df)} bars")
        except Exception as e:
            print(f"  skip {inst}: {e}")

    cfg = Config(initial_equity=args.equity)
    res = PortfolioBacktester(data, cfg).run()

    print("\n" + "=" * 64)
    print(f"PORTFOLIO BACKTEST  ({res.get('symbols', 0)} symbols, {args.bar})")
    print("=" * 64)
    for k in ("overall", "total_return_pct", "final_equity", "max_drawdown_pct",
              "halted", "exit_reasons", "in_sample", "out_of_sample"):
        print(f"{k:>18}: {res.get(k)}")
    if "per_symbol" in res:
        print("\nper-symbol:")
        print(res["per_symbol"].to_string())
        res["trade_log"].to_csv("portfolio_trades.csv", index=False)
        print("\npooled trade log -> portfolio_trades.csv")


if __name__ == "__main__":
    main()
