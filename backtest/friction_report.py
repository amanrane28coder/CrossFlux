#!/usr/bin/env python3
"""Generate backtest/friction_report.html from the sweep JSON.

A generator rather than a hand-written page, so the numbers on it cannot drift
away from the run that produced them. Regenerate with:

    python3 backtest/sweep_order_size.py --friction stress \\
        --qty 0.01 0.05 0.1 0.25 0.5 1.0 2.0 3.0 5.0 --out /tmp/sweep.json
    python3 backtest/friction_report.py /tmp/sweep.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:      # so `python3 backtest/friction_report.py` works
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "backtest" / "friction_report.html"

W, H = 900, 420
PAD_L, PAD_R, PAD_T, PAD_B = 62, 62, 28, 52


def load(path: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    rows = json.loads(path.read_text())
    by_preset: dict[str, list[dict]] = {}
    for r in rows:
        by_preset.setdefault(r["friction"], []).append(r)
    for v in by_preset.values():
        v.sort(key=lambda r: r["qty"])
    return rows, by_preset


def log_x(qty: float, lo: float, hi: float) -> float:
    t = (math.log(qty) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return PAD_L + t * (W - PAD_L - PAD_R)


def peak_of(series: list[dict]) -> dict:
    """The capacity peak: the size with the highest net PnL.

    One function, because the chart's marker and the prose both make this claim
    and two separate ``max`` calls can disagree. A mutation that made the marker
    read the *minimum* was invisible while each consumer computed its own.
    """
    return max(series, key=lambda r: r["pnl_net"])


def curve_svg(series: list[dict]) -> str:
    """Win rate (left axis) against net PnL (right axis), over order size.

    The two curves cross, and where they cross is the finding: win rate falls
    monotonically with size while PnL rises to a peak and then turns over. One
    chart, one crossing, nothing else on it.
    """
    lo, hi = series[0]["qty"], series[-1]["qty"]
    wins = [r["win_rate_pct"] for r in series]
    pnls = [r["pnl_net"] for r in series]
    # Axis floor is derived, not the 60 that happens to suit today's numbers: a
    # rerun at larger size could take the win rate below any hardcoded floor and
    # the curve would silently leave the chart.
    w_lo = min(60.0, 10.0 * math.floor(min(wins) / 10.0) - 5.0)
    w_hi = 100.0
    ticks = [w_lo + i * (w_hi - w_lo) / 4 for i in range(5)]
    p_hi = max(pnls) * 1.08

    def wy(v: float) -> float:
        t = (v - w_lo) / (w_hi - w_lo)
        return H - PAD_B - t * (H - PAD_T - PAD_B)

    def py(v: float) -> float:
        return H - PAD_B - (v / p_hi) * (H - PAD_T - PAD_B)

    peak = peak_of(series)
    parts: list[str] = []

    # horizontal guides on the win-rate axis
    for v in ticks:
        y = wy(v)
        parts.append(f'<line class="grid" x1="{PAD_L}" y1="{y:.1f}" '
                     f'x2="{W - PAD_R}" y2="{y:.1f}"/>')
        parts.append(f'<text class="ax" x="{PAD_L - 10}" y="{y + 4:.1f}" '
                     f'text-anchor="end">{v:g}%</text>')

    # size ticks — the real sequence in this data, so it gets to be the axis
    for r in series:
        x = log_x(r["qty"], lo, hi)
        parts.append(f'<line class="tick" x1="{x:.1f}" y1="{H - PAD_B}" '
                     f'x2="{x:.1f}" y2="{H - PAD_B + 6}"/>')
        parts.append(f'<text class="ax" x="{x:.1f}" y="{H - PAD_B + 22}" '
                     f'text-anchor="middle">{r["qty"]:g}</text>')

    pnl_pts = " ".join(f"{log_x(r['qty'], lo, hi):.1f},{py(p):.1f}"
                       for r, p in zip(series, pnls))
    win_pts = " ".join(f"{log_x(r['qty'], lo, hi):.1f},{wy(w):.1f}"
                       for r, w in zip(series, wins))
    parts.append(f'<polyline class="pnl" points="{pnl_pts}"/>')
    parts.append(f'<polyline class="win" points="{win_pts}"/>')

    for r, v in zip(series, pnls):
        parts.append(f'<circle class="dpnl" cx="{log_x(r["qty"], lo, hi):.1f}" '
                     f'cy="{py(v):.1f}" r="3.5"/>')
    for r, v in zip(series, wins):
        parts.append(f'<circle class="dwin" cx="{log_x(r["qty"], lo, hi):.1f}" '
                     f'cy="{wy(v):.1f}" r="3.5"/>')

    px = log_x(peak["qty"], lo, hi)
    parts.append(f'<line class="peak" x1="{px:.1f}" y1="{PAD_T}" '
                 f'x2="{px:.1f}" y2="{H - PAD_B}"/>')
    # The peak sits near the right edge, so the label flips to the inside rather
    # than running off the viewBox. Anchor follows the side it lands on.
    right = px > W * 0.55
    parts.append(f'<text class="peaklab" x="{px + (-8 if right else 8):.1f}" '
                 f'y="{PAD_T + 14}" '
                 f'text-anchor="{"end" if right else "start"}">'
                 f'PnL peaks at {peak["qty"]:g} BTC — '
                 f'${peak["pnl_net"] / 1e6:.1f}M</text>')

    parts.append(f'<text class="axlab" x="{PAD_L}" y="{H - 6}">'
                 f'order size, BTC per leg (log)</text>')
    parts.append(f'<text class="axlab win-lab" x="{W - PAD_R}" y="{H - 6}" '
                 f'text-anchor="end">win rate — net PnL</text>')
    body = "\n      ".join(parts)
    return (f'<svg viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="Win rate and net PnL against order size">\n      '
            f'{body}\n    </svg>')


def rows_table(series: list[dict]) -> str:
    out = []
    for r in series:
        out.append(
            "<tr>"
            f'<td class="n">{r["qty"]:g}</td>'
            f'<td class="n">{r["win_rate_pct"]:.1f}%</td>'
            f'<td class="n slip">{r["adverse_fills"]:,}</td>'
            f'<td class="n">{r["unfilled_pct"]:.2f}%</td>'
            f'<td class="n">{r["legged_fills"]:,}</td>'
            f'<td class="n">{r["mean_edge_bps"]:.2f}</td>'
            f'<td class="n pos">${r["pnl_net"]/1e6:,.2f}M</td>'
            f'<td class="n">${r["worst_fill"]:,.2f}</td>'
            "</tr>"
        )
    return "\n        ".join(out)


def preset_table(by_preset: dict[str, list[dict]]) -> str:
    order = ["zero", "colocated", "stress", "retail"]
    sizes = [0.01, 0.25, 1.0, 3.0]
    out = []
    for name in order:
        series = by_preset.get(name)
        if not series:
            continue
        cells = []
        for q in sizes:
            hit = next((r for r in series if abs(r["qty"] - q) < 1e-12), None)
            cells.append(f'<td class="n">{hit["win_rate_pct"]:.1f}%</td>'
                         if hit else '<td class="n">—</td>')
        out.append(f'<tr><td class="k">{name}</td>' + "".join(cells) + "</tr>")
    return "\n        ".join(out)


def main() -> int:
    # Imported here, not at module scope: the template calls back into this
    # module for curve_svg/rows_table, so a top-level import would be circular.
    from backtest._friction_report_template import render

    src = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sweep.json")
    rows, by_preset = load(src)
    if "stress" not in by_preset:
        raise SystemExit(f"{src} has no 'stress' rows; the page charts that "
                         f"preset. Found: {sorted(by_preset)}")
    OUT.write_text(render(rows, by_preset, by_preset["stress"]))
    print("wrote", OUT, f"({len(rows)} sweep rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
