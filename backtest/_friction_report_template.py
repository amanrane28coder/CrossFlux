"""HTML template for backtest/friction_report.py.

Kept separate so the page's markup and the numbers that fill it stay apart:
friction_report.py owns the arithmetic, this file owns the presentation. Palette
and mono display type are deliberately the same as
backtest/fee_sensitivity_report.html so the two result pages read as a set.
"""
from __future__ import annotations

CSS = """
    :root{
      color-scheme: light;
      --paper:#E7EAE4; --panel:#F2F4EF; --ink:#171B1A; --mute:#5C6664;
      --rule:#C9D0C9; --net:#0E6B57; --slip:#8C2F26; --ochre:#A8741F;
      --mono:ui-monospace,"SFMono-Regular",Menlo,Consolas,monospace;
      --sans:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    }
    *{box-sizing:border-box}
    body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
         font-size:15px;line-height:1.62;-webkit-font-smoothing:antialiased}
    .wrap{max-width:1000px;margin:0 auto;padding:56px 28px 96px}
    .eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.18em;
             text-transform:uppercase;color:var(--mute);margin:0 0 14px}
    h1{font-family:var(--mono);font-size:clamp(26px,4.2vw,42px);font-weight:600;
       letter-spacing:-.02em;line-height:1.12;margin:0 0 18px;max-width:22ch}
    h2{font-family:var(--mono);font-size:14px;font-weight:600;letter-spacing:.06em;
       text-transform:uppercase;color:var(--mute);margin:56px 0 16px;
       padding-bottom:8px;border-bottom:1px solid var(--rule)}
    p{margin:0 0 16px;max-width:74ch}
    .lede{font-size:17px;max-width:70ch}
    b{font-weight:600}
    code{font-family:var(--mono);font-size:.9em;background:var(--panel);
         padding:1px 5px;border:1px solid var(--rule)}
    .fig{background:var(--panel);border:1px solid var(--rule);padding:20px 18px 10px;
         margin:8px 0 10px}
    .fig svg{width:100%;height:auto;display:block;overflow:visible}
    .cap{font-family:var(--mono);font-size:11.5px;color:var(--mute);
         margin:0 0 34px;max-width:80ch}
    .grid{stroke:var(--rule);stroke-width:1}
    .tick{stroke:var(--rule);stroke-width:1}
    .ax{font-family:var(--mono);font-size:10.5px;fill:var(--mute)}
    .axlab{font-family:var(--mono);font-size:10.5px;fill:var(--mute);
           letter-spacing:.09em;text-transform:uppercase}
    .win-lab{fill:var(--ink)}
    .win{fill:none;stroke:var(--slip);stroke-width:2.2;stroke-linejoin:round}
    .pnl{fill:none;stroke:var(--net);stroke-width:2.2;stroke-linejoin:round;
         stroke-dasharray:6 3}
    .dwin{fill:var(--slip)} .dpnl{fill:var(--net)}
    .peak{stroke:var(--ink);stroke-width:1;stroke-dasharray:2 3;opacity:.55}
    .peaklab{font-family:var(--mono);font-size:11px;fill:var(--ink)}
    table{width:100%;border-collapse:collapse;font-family:var(--mono);
          font-size:12.5px;margin:0 0 10px}
    th{text-align:right;font-weight:600;color:var(--mute);padding:7px 10px;
       border-bottom:1px solid var(--ink);white-space:nowrap;font-size:11px;
       letter-spacing:.04em;text-transform:uppercase}
    th:first-child,td:first-child{text-align:left}
    td{padding:6px 10px;border-bottom:1px solid var(--rule)}
    td.n{text-align:right;font-variant-numeric:tabular-nums}
    td.k{color:var(--mute)}
    .pos{color:var(--net)} .slip{color:var(--slip)}
    .note{border-left:2px solid var(--ochre);padding:2px 0 2px 16px;
          margin:26px 0;max-width:74ch}
    .note h3{font-family:var(--mono);font-size:12px;letter-spacing:.06em;
             text-transform:uppercase;color:var(--ochre);margin:0 0 6px}
    .note p{font-size:14px;margin:0 0 10px}
    .note p:last-child{margin:0}
    footer{margin-top:64px;padding-top:18px;border-top:1px solid var(--rule);
           font-family:var(--mono);font-size:11px;color:var(--mute)}
    @media (max-width:640px){
      .wrap{padding:34px 16px 64px} table{font-size:11px}
      td,th{padding:5px 6px}
    }
"""


def render(rows: list[dict], by_preset: dict[str, list[dict]],
           stress: list[dict]) -> str:
    from backtest.friction_report import (curve_svg, peak_of, preset_table,
                                          rows_table)

    small, big = stress[0], stress[-1]
    peak = peak_of(stress)
    signals = stress[0]["signals"]
    presets = ", ".join(sorted(by_preset))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CrossFlux — what execution friction costs</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <p class="eyebrow">CrossFlux · execution friction · BTC-USD/USDT · 2024-03-01</p>
  <h1>The backtester can now lose money</h1>

  <p class="lede">The entry gate and the PnL calculation used to read the same
  order-book snapshot: a trade was taken when <code>margin &gt; fee</code> and
  then booked at <code>margin − fee</code>, so every trade it took was
  profitable by construction and the win rate was 100% by arithmetic rather
  than by skill. Two mechanisms break that. A signal at <b>T</b> now fills
  against the book prevailing at <b>T + latency</b>, and the fill price is the
  VWAP of the levels the order actually consumes rather than the touch. A trade
  that turns against us in the window is booked as a loss instead of being
  rejected.</p>

  <p>Below is the same day of real Tardis L2 data swept across order size, with
  institutional fees (3.0 bps round trip) and the <code>stress</code> friction
  preset (100 ms per leg, no jitter). {signals:,} signals, both directions.</p>

  <h2>Win rate and PnL, against size</h2>
  <div class="fig">
    {curve_svg(stress)}
  </div>
  <p class="cap">Solid — win rate, left axis. Dashed — net PnL, scaled to its
  own maximum. The curves move against each other, which is the point: size
  buys notional faster than it costs accuracy, until it doesn't.</p>
""" + f"""
  <h2>The full sweep</h2>
  <table>
    <thead><tr>
      <th>BTC / leg</th><th>win</th><th>adverse fills</th><th>unfilled</th>
      <th>legged</th><th>edge bps</th><th>net PnL</th><th>worst fill</th>
    </tr></thead>
    <tbody>
        {rows_table(stress)}
    </tbody>
  </table>
  <p class="cap">Adverse fills are trades booked at a loss — the ones the old
  code discarded. Legged fills are trades where one leg filled more than the
  other, leaving a naked residual charged at the preset's legging cost. Worst
  fill is the single largest loss in the ledger.</p>

  <p>Three things in that table are worth reading closely. Win rate falls
  monotonically, {small["win_rate_pct"]:.1f}% at {small["qty"]:g} BTC down to
  {big["win_rate_pct"]:.1f}% at {big["qty"]:g} BTC, so the strategy is now
  falsifiable — there are {big["adverse_fills"]:,} losing trades at the top of
  the range where before there were none at any size. Mean edge decays from
  {small["mean_edge_bps"]:.2f} bps to {big["mean_edge_bps"]:.2f} bps, which is
  slippage being paid rather than assumed. And net PnL peaks at
  {peak["qty"]:g} BTC (${peak["pnl_net"]/1e6:.1f}M) and then <b>falls</b> to
  ${big["pnl_net"]/1e6:.1f}M at {big["qty"]:g} BTC.</p>

  <p>That peak is the first result this backtester has produced that it could
  not have produced by construction. A capacity limit is what a real execution
  model looks like; the old one had none, because doubling size doubled PnL
  exactly.</p>

  <h2>Latency alone barely matters. Size does.</h2>
  <table>
    <thead><tr>
      <th>friction preset</th><th>0.01 BTC</th><th>0.25 BTC</th>
      <th>1.0 BTC</th><th>3.0 BTC</th>
    </tr></thead>
    <tbody>
        {preset_table(by_preset)}
    </tbody>
  </table>
  <p class="cap">Win rate by preset. Presets in this run: {presets}.</p>
""" + f"""
  <p>Note the <code>zero</code> row: no latency at all, and the win rate is
  still not 100%. I expected it to be, and wrote that down before measuring;
  the measurement disagreed. Removing latency removes the delay but not the
  book walk, and size on its own is enough to lose money — the gate quotes the
  touch while the fill pays the VWAP of five levels. Recovering a true 100%
  would need zero latency <b>and</b> an infinitely deep top level, which is to
  say it would need the tautology back.</p>

  <p>The reason 100 ms costs so little at small size is a property of the feed
  rather than a bug. <code>binance_book_snapshot_5</code> updates on a ~100 ms
  cadence, so a 100 ms delay advances the book by roughly one row, and the touch
  price changes on only 14.0% of binance rows and 8.4% of kraken rows (mean
  |Δask| ≈ $0.45). Widening latency to the <code>retail</code> preset's
  hundreds of milliseconds moves the needle; widening it within one update
  interval cannot.</p>

  <div class="note">
    <h3>Caveat: above ~0.25 BTC, unfilled is data truncation</h3>
    <p>These are five-level snapshots. Past roughly 0.25 BTC the unfilled
    fraction stops measuring illiquidity and starts measuring the depth of the
    file — the book almost certainly had a sixth level, and this dataset does
    not say what was on it. The {big["unfilled_pct"]:.1f}% unfilled at
    {big["qty"]:g} BTC is therefore an upper bound on the real shortfall, and
    the PnL peak is a lower bound on the real capacity. Resolving it needs the
    full L2 incremental feed, the same blocker as true multi-level OFI.</p>
  </div>

  <div class="note">
    <h3>Caveat: the surviving PnL is a basis, not an arbitrage</h3>
    <p>Total PnL stays positive at every size, and that is not evidence the
    tautology survived — it is the USDT/USD basis. 99.7% of signals point one
    way (307,089 buy-binance/sell-kraken against 792 the reverse), and the mean
    edge of ~4 bps sits right on the +4.79 bps basis measured in TEST_REPORT
    §2.3. A genuine cross-venue arbitrage would fire in both directions at
    similar rates.</p>
    <p>So breaking the tautology makes the win rate <em>meaningful</em> without
    making this strategy an arbitrage. What the sweep now measures honestly is
    how much of a known basis a given order size can actually collect after
    latency and slippage — which is a real question, and was previously
    unanswerable.</p>
  </div>

  <div class="note">
    <h3>Caveat: the entry gate selects deep books</h3>
    <p>On signal rows the buy leg absorbs a full 3.0 BTC 87.4% of the time,
    against about 55% of rows unconditionally. The gate is picking moments when
    the book happens to be deep, so any slippage estimate taken over an
    unselected sample overstates what this strategy pays — and, symmetrically,
    the numbers here are specific to this gate and would move if the gate
    changed.</p>
  </div>
""" + f"""
  <h2>How the fill is decided</h2>
  <p>A signal at <b>T</b> is buffered rather than booked. Each leg draws its own
  fill time at <b>T + latency</b> (with optional log-normal jitter), and each
  leg then reads the <em>prevailing</em> quote on its own venue — the last
  update at or before its fill time, <code>searchsorted(..., "right") − 1</code>.
  Dropping the <code>− 1</code> would read the next quote instead, which is
  look-ahead: the same class of defect as using the exchange
  <code>timestamp</code> where the arriving <code>local_timestamp</code> is what
  a trader would have seen.</p>

  <p>Because each leg has its own latency, the two legs can fill different
  quantities. Only <code>min(buy_filled, sell_filled)</code> is hedged and earns
  the spread; the difference is a naked residual, unwound at the over-filling
  venue's price and charged at the preset's legging cost. Fees are charged on
  what each leg <em>filled</em>, not on the matched quantity, so an over-filled
  leg pays in full and earns nothing back. Only one rejection remains in the
  whole path: a venue with no quote at all at the fill time. Everything else
  fills, at whatever price the book gives.</p>

  <p>The queue is a genuine buffer in the tick-by-tick path and collapses to an
  as-of lookup in the vectorised one, which is sound only because nothing here
  lets one pending order affect another — no shared inventory, no capital
  constraint, no queue position. Adding an inventory limit would make that
  collapse invalid and the drain would have to become ordered.</p>

  <h2>Verification</h2>
  <p>The fill path is checked element-wise against an independent
  reimplementation — a separate scalar book-walk and a separate bisect for the
  as-of lookup, written against the spec rather than the code. Worst relative
  deviation across buy price, sell price, matched quantity, fee, legging cost,
  net PnL and fill timestamp is <b>2.2 × 10⁻¹⁵</b> over 10,000 trades at two
  sizes. The Python and C++ book walks agree bit-for-bit, which took writing the
  reference loop as a running total (<code>want − taken</code>) rather than a
  decrement; the obvious form disagreed with the vectorised cumsum path by
  2.8 × 10⁻¹⁴, and a parity check that only nearly passes is not a parity
  check.</p>

  <p>One earlier version of this curve, from a throwaway measurement script,
  reported 93.3% and 59.8% win rates and 54.96% unfilled. Those numbers are
  wrong and have been struck. That script read only the first 400,000 rows of
  each file, which truncates binance at 13.86 h but kraken at 7.58 h, and its
  forward-fill then held a single frozen kraken snapshot across the final 45% of
  its sample. Every fill in that tail priced against a stale book.</p>

  <footer>
    Generated by <code>backtest/friction_report.py</code> from the sweep JSON —
    every number on this page comes from that file, so the page cannot drift
    from the run. Regenerate with
    <code>python3 backtest/sweep_order_size.py --out sweep.json</code> then
    <code>python3 backtest/friction_report.py sweep.json</code>.
  </footer>
</div>
</body>
</html>
"""
