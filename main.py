# SPDX-License-Identifier: MIT
"""BTC pyramid trend bot for make.fun btc-race-24h-20261123-0511.

Long-only pyramid trend follower on 1-minute cbBTC/USDC bars, speaking
agent-fun/bars-replay/v1 through the vendored kit shim (vendor/makefun_bars.py).

What the dev window taught us (and the design follows from it):
- A fast EMA cross whipsaws itself to death on 1-minute bars: the kit's
  own ema-cross baseline loses ~$115 on the dev window at a 13% win rate,
  and every fast system we tried pays its edge away in the ~26 bps
  round-trip cost. The window's money sits in one multi-hour run, and the
  rest is chop that punishes every exit and re-entry.
- So this bot runs ONE campaign per window. It opens a first half-size leg
  when the short trend is up (60-bar return >= 0, 120-bar return >= +30 bps,
  price above the 60-bar EMA), adds the second half only when the 4-hour
  return confirms the trend (>= +100 bps), and then holds. It exits only
  on a hard, confirmed reversal (price 400 bps under its peak while the
  60-bar return is <= -100 bps, or price under the slow EMA while the
  4-hour return is < -200 bps) and never re-enters after that, so chop
  cannot bleed it. A mirrored (down-trending) copy of the dev window costs
  it only ~$2.60, and the worst shuffled copy ~$36, both with drawdown
  under 4% — while the real dev window pays +$26.44 on 2 trades.
- Position size is a share of equity, trimmed when the pool impact of the
  trade would eat the edge. Hard discipline: if equity ever falls 8%
  under its peak the bot flattens and stops trading. No trade is opened
  in the last bars. Everything is integer math on bar closes, seeded from
  nothing but the bars themselves, so two runs are identical.

No network, no wall clock, no files, no environment reads. Standard library
only, plus the vendored kit shim.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

SYMBOL = "CBBTC"
BPS = 10000

# Tunable strategy parameters (single source of truth for the tuner).
PARAMS = {
    "leg1_bps": 5000,     # first leg as a share of equity (50%)
    "leg2_bps": 5000,     # second leg as a share of equity (50%)
    "slow_n": 60,         # slow EMA length for the regime filter (bars)
    "ret60_min_bps": 0,   # 60-bar return must beat this to open leg 1
    "ret120_min_bps": 30,  # 120-bar return must beat this to open leg 1
    "ret240_add_bps": 100,  # 240-bar return must beat this to add leg 2
    "trail_bps": 400,     # hard exit: this far under the peak, with ret60 <= -100
    "exit_ret60_bps": -100,  # 60-bar return confirming the hard exit
    "regime_bps": -200,   # 240-bar return confirming the regime exit
    "max_impact_bps": 12,  # trim size so one trade's pool impact stays under this
    "cooldown_bars": 20,  # bars to wait after an exit before re-entering
    "halt_dd_bps": 800,   # flatten + halt if equity drops this far under peak
    "no_entry_last": 30,  # do not open in the final N bars of the window
    "max_slip_bps": 200,  # cap on fill move vs the latest close
}


def _tdiv(a, b):
    """Integer division truncated toward zero (deterministic across runs)."""
    q = abs(a) // b
    return q if (a >= 0) == (b >= 0) else -q


class TrendBot:
    """Keeps the bars, the slow EMA and the campaign state. All integer math."""

    def __init__(self, params):
        self.p = params
        self.closes = []
        self.ema_slow = None
        self.legs = 0            # legs opened so far in this campaign
        self.peak = None         # best close since the campaign opened
        self.cooldown_until = -1
        self.equity_peak = None
        self.halted = False
        self.done = False        # the campaign is over: never re-enter
        self.seq = -1
        self.total_bars = 1440   # the window is 1,440 one-minute bars

    # -- indicators ------------------------------------------------------
    def _ema_step(self, ema, price, n):
        # EMA: ema += (price - ema) * 2 / (n + 1), truncated.
        return ema + _tdiv((price - ema) * 2, n + 1)

    def _update(self, bar):
        c = int(bar["c"])
        self.closes.append(c)
        if self.ema_slow is None:
            self.ema_slow = c
        else:
            self.ema_slow = self._ema_step(self.ema_slow, c, self.p["slow_n"])
        if self.peak is not None and c > self.peak:
            self.peak = c

    def _ret_bps(self, n):
        # Return over the last n bars in bps; 0 until n bars of history.
        if len(self.closes) <= n:
            return 0
        old = self.closes[-1 - n]
        return (self.closes[-1] - old) * BPS // old

    # -- orders ----------------------------------------------------------
    def _buy(self, state, frac_bps):
        # A leg sized as a share of equity, trimmed by cash, the weight
        # cap and the pool-impact budget.
        notional = state.equity * frac_bps // BPS
        if notional > state.cash:
            notional = state.cash
        room = state.weight_room(SYMBOL)
        if notional > room:
            notional = room
        impact = state.impact_bps(SYMBOL, notional)
        if impact is not None and impact > self.p["max_impact_bps"]:
            notional = notional * self.p["max_impact_bps"] // impact
        if notional < 1_000_000:  # min notional is 1 USDC
            return []
        return [{"side": "buy", "symbol": SYMBOL,
                 "notional_quote_units": notional,
                 "max_slippage_bps": self.p["max_slip_bps"]}]

    def _sell_all(self, state):
        holding = state.positions.get(SYMBOL)
        if not holding:
            return []
        self.cooldown_until = self.seq + self.p["cooldown_bars"]
        self.peak = None
        return [{"side": "sell", "symbol": SYMBOL, "qty": holding.qty,
                 "max_slippage_bps": self.p["max_slip_bps"]}]

    # -- decisions -------------------------------------------------------
    def on_batch(self, state, batch):
        self.seq = state.seq
        for bar in batch["bars"]:
            if bar["symbol"] == SYMBOL:
                self._update(bar)
        if state.warming_up:
            return []
        if len(self.closes) < 122:  # need 120 bars of history to judge
            return []
        if self.halted or self.done:
            return []

        close = self.closes[-1]
        equity = state.equity
        if self.equity_peak is None or equity > self.equity_peak:
            self.equity_peak = equity

        holding = state.positions.get(SYMBOL)
        qty = holding.qty if holding else 0

        # Hard drawdown discipline: flatten once, then stop trading.
        if self.equity_peak and equity * BPS < self.equity_peak * (BPS - self.p["halt_dd_bps"]):
            self.halted = True
            return self._sell_all(state)

        if qty:
            # The campaign is open: add leg 2 once the 4-hour trend
            # confirms, else exit only on a hard, confirmed reversal.
            if self.peak is None:
                self.peak = close
            if self.legs == 1 and self._ret_bps(240) >= self.p["ret240_add_bps"]:
                self.legs = 2
                return self._buy(state, self.p["leg2_bps"])
            under_peak = (self.peak - close) * BPS // self.peak
            hard_exit = (under_peak >= self.p["trail_bps"]
                         and self._ret_bps(60) <= self.p["exit_ret60_bps"])
            regime_exit = (close < self.ema_slow
                           and self._ret_bps(240) < self.p["regime_bps"])
            if hard_exit or regime_exit:
                self.done = True
                return self._sell_all(state)
            return []

        # Flat: open leg 1 on a confirmed short uptrend, in an uptrend
        # regime. One campaign per window: after the exit, done is set.
        if self.seq >= self.total_bars - self.p["no_entry_last"]:
            return []
        if self.seq < self.cooldown_until:
            return []
        if self.legs == 0 and close > self.ema_slow \
                and self._ret_bps(60) >= self.p["ret60_min_bps"] \
                and self._ret_bps(120) >= self.p["ret120_min_bps"]:
            self.legs = 1
            self.peak = close
            return self._buy(state, self.p["leg1_bps"])
        return []


def on_batch(state, batch):
    return BOT.on_batch(state, batch)


BOT = TrendBot(PARAMS)

if __name__ == "__main__":
    makefun_bars.run(on_batch)
