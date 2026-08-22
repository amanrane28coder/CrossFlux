import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

st.set_page_config(
    page_title="CrossFlux | Microstructure Execution Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# One <style> block, not two. A second injected block would not "override" this
# one -- both land in the same cascade, and equal-specificity rules are decided by
# document order, so the winner would depend on which st.markdown ran last. Every
# rule lives here.
#
# No blank lines inside the payload. A <style> tag opens a CommonMark HTML block
# of type 1, which runs to the closing tag regardless of blank lines, so this
# particular block is not exposed to the bug that printed the header's tags -- but
# the rule is kept anyway, because the exemption is easy to forget and the failure
# is silent.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
    * { font-family: 'Inter', -apple-system, sans-serif; }
    /* True black for the page, with panels one step above it. Panel surfaces stay
       #0f1217 because the Plotly figures set paper_bgcolor to that value in Python
       where CSS cannot reach them -- moving the panel colour here alone would leave
       every chart floating on a mismatched rectangle. */
    html, body, .main, .stApp { background: #0a0a0a; overflow-anchor: auto; }
    [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #0a0a0a; }
    [data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent; }
    .block-container { padding: 0.25rem 0.75rem; max-width: 1560px; }
    .stApp header, #MainMenu, footer, .stDeployButton { display: none !important; }
    hr { margin: 0.2rem 0; opacity: 0.06; border: 0; border-top: 1px solid #2b3139; }
    /* Reserve minimum heights on chart and metric containers so the layout
       skeleton does not collapse to 0px during the teardown-rebuild gap. */
    .stPlotlyChart { border: 1px solid #2b3139; border-radius: 4px; background: #0f1217; padding: 0.15rem; min-height: 80px; }
    .stDataFrame, [data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid #2b3139; border-radius: 4px; font-size: 0.6rem; background: #0f1217; }
    /* Tables hold the fast-moving numbers, so they take the mono face and tabular
       figures. Caveat worth knowing before believing this worked: st.dataframe
       renders through glide-data-grid, which paints text into a <canvas> from its
       own theme object and does not inherit CSS. The custom property below is the
       hook it reads; if grid text still looks proportional, the real fix is
       font = "monospace" under [theme] in .streamlit/config.toml. */
    [data-testid="stDataFrame"], [data-testid="stDataFrame"] *, [data-testid="stTable"], [data-testid="stTable"] * { font-family: 'JetBrains Mono', 'Courier New', monospace !important; font-variant-numeric: tabular-nums; }
    [data-testid="stDataFrame"] { --gdg-font-family: 'JetBrains Mono', 'Courier New', monospace; }
    /* The asset selector is a radio group, not a select. Styled as a row of chips,
       but the radio circle is deliberately left visible: if the :has() rules below
       stop matching after a Streamlit DOM change, a hidden circle would leave no
       selection indicator at all, whereas an unstyled one still tells the truth. */
    div[role="radiogroup"] { gap: 0.15rem; flex-wrap: wrap; align-items: center; }
    div[role="radiogroup"] label { background: #0f1217; border: 1px solid #2b3139; border-radius: 3px; padding: 0.1rem 0.45rem; margin: 0; cursor: pointer; }
    div[role="radiogroup"] label p, div[role="radiogroup"] label div { font-family: 'JetBrains Mono', 'Courier New', monospace !important; font-size: 0.6rem !important; letter-spacing: 0.04em; margin-bottom: 0; }
    div[role="radiogroup"] label p { color: #6b7280 !important; }
    div[role="radiogroup"] label:hover { border-color: #3b444f; }
    div[role="radiogroup"] label:has(input:checked) { border-color: #6b5410; background: #14120a; }
    div[role="radiogroup"] label:has(input:checked) p { color: #eab308 !important; }
    div[role="radiogroup"] label:has(input:focus-visible) { outline: 1px solid #6fa8bb; outline-offset: 1px; }
    div[role="tablist"] { background: transparent; border: none; gap: 1rem; margin-bottom: 0.2rem; }
    button[role="tab"] { border-radius: 0 !important; font-size: 0.65rem !important; font-weight: 500 !important; color: #6b7280 !important; border-bottom: 1px solid transparent !important; padding: 0.2rem 0.5rem !important; }
    button[role="tab"][aria-selected="true"] { background: transparent !important; color: #eaecef !important; border-bottom: 1px solid #eab308 !important; }
    /* Metric styling. Target both the old "metric-container" and the current
       "stMetric" test-id. No centering overrides -- Streamlit's native layout
       uses visually-hidden / clipped spans for accessibility, and overriding
       display/flex/width on those containers un-clips the SR copy, causing
       every label and value to render twice (the "ghost text" bug). */
    div[data-testid="stMetric"], div[data-testid="metric-container"] { background: transparent; border: none; padding: 0.15rem 0.1rem; min-height: 3rem; }
    div[data-testid="stMetric"] label, div[data-testid="metric-container"] label, [data-testid="stMetricLabel"] { color: #6fa8bb !important; font-weight: 500 !important; font-size: 0.55rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
    [data-testid="stMetricLabel"] > div, [data-testid="stMetricLabel"] p { font-family: 'JetBrains Mono', 'Courier New', monospace !important; }
    /* tabular-nums pins every digit to one advance width, so a value ticking
       9 -> 0 does not reflow the row. */
    [data-testid="stMetricValue"] { color: #eaecef !important; font-weight: 500 !important; font-size: 0.85rem !important; font-family: 'JetBrains Mono', 'Courier New', monospace !important; line-height: 1.4; font-variant-numeric: tabular-nums; font-feature-settings: "tnum" 1; white-space: nowrap; }
    [data-testid="stMetricDelta"] { font-size: 0.6rem !important; font-weight: 400 !important; color: #6b7280 !important; font-family: 'JetBrains Mono', 'Courier New', monospace !important; font-variant-numeric: tabular-nums; }
    /* Hide the delta direction indicator in both forms Streamlit has shipped it:
       an inline <svg> (current) and a ::before glyph (older). */
    [data-testid="stMetricDelta"] svg { display: none; }
    [data-testid="stMetricDelta"]::before { content: none !important; }
    .stTabs [data-baseweb="tab-list"] { background: transparent; border: none; gap: 1rem; margin-bottom: 0.2rem; }
    .stTabs [data-baseweb="tab"] { border-radius: 0 !important; font-size: 0.65rem !important; font-weight: 500 !important; color: #6b7280 !important; border-bottom: 1px solid transparent !important; padding: 0.2rem 0.5rem !important; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background: transparent !important; color: #eaecef !important; border-bottom: 1px solid #eab308 !important; }
    [data-testid="stCaptionContainer"] { color: #6b7280; font-size: 0.55rem; }
    [data-testid="stInfoText"] { color: #6b7280; font-size: 0.65rem; }
    .st-bb, .st-at { background-color: transparent !important; }
    .element-container:has(> .stPlotlyChart) { margin: 0; contain: layout style; }
    .st-emotion-cache-1y4p8pa { max-width: 100%; padding: 0; }
    /* Prevent the main content block from reflowing during rerenders. */
    .block-container > div { contain: layout style; }
    /* Uniform column padding, no first/last exceptions. The exceptions bought
       0.15rem of flush edge alignment at the price of shifting the first and last
       metric's centre relative to the rest of the row -- the wrong trade once the
       values are centred. */
    div[data-testid="column"] { padding: 0 0.15rem; }
    /* Hairline between adjacent metric columns, so a row of readings reads as one
       instrument cluster. Scoped with :has() to metric columns on both sides, so it
       never draws beside a chart or a table, and simply does not render on a browser
       without :has() rather than misfiring. */
    div[data-testid="column"]:has([data-testid="stMetric"]) + div[data-testid="column"]:has([data-testid="stMetric"]) { border-left: 1px solid #15191e; }
    .stAlert { background: #0f1217; border: 1px solid #2b3139; border-radius: 4px; color: #6b7280; font-size: 0.7rem; }
    .stAlert > div { gap: 0.3rem; }
</style>
""", unsafe_allow_html=True)

SIGNALS_PATH = Path("/tmp/live_signals.csv")
SIGNALS_PY_PATH = Path("/tmp/live_signals_py.csv")
STATUS_PATH = Path("/tmp/live_status.csv")
PRICES_PATH = Path("/tmp/live_prices.csv")
DEPTH_PATH = Path("/tmp/live_depth.csv")
ORDERS_PATH = Path("/tmp/live_orders.csv")
ORDERS_PY_PATH = Path("/tmp/live_orders_py.csv")

def read_csv(path):
    if not path.exists(): return None
    try:
        df = pd.read_csv(path)
        return df if len(df) > 0 else None
    except Exception:
        return None

def calc_obi(bid_v, ask_v):
    tot = bid_v + ask_v
    return (bid_v - ask_v) / tot if tot > 0 else 0.0

# --- Weighted OBI ---------------------------------------------------------
# The engine writes two deltas per signal: obi_delta (equal weight on every
# book level) and weighted_obi_delta (per-level weights from the profile named
# in the obi_profile column). Both are shown; neither replaces the other. Which
# one actually drove the entry decision is a C++-side setting, so the dashboard
# reports both and labels the profile rather than guessing.
#
# weighted_obi_delta is empty -> NaN whenever the engine did not compute it,
# which is the default flat configuration. That is distinct from 0.0, which
# means the weighted reading was computed and came out balanced.

def weighted_obi_state(df):
    """Return (present, values, profile) for the weighted column of `df`.

    present : bool          -- column exists and holds at least one real number
    values  : Series | None -- the numeric column, NaNs intact
    profile : str           -- profile name from the CSV, or "" if unlabelled
    """
    if df is None or "weighted_obi_delta" not in df.columns:
        return False, None, ""
    vals = pd.to_numeric(df["weighted_obi_delta"], errors="coerce")
    profile = ""
    if "obi_profile" in df.columns:
        named = df["obi_profile"].dropna()
        if len(named):
            # Last wins: a profile change mid-file means the tail is current.
            profile = str(named.iloc[-1])
    return bool(vals.notna().any()), vals, profile

# --- Order column names ---------------------------------------------------
# This page was written against a column called pnl_realized. No writer in the
# repo ever produced that name: cpp_engine/src/execution_manager.cpp emits
# gross_pnl/net_pnl and src/live_ingestion.py emits net_pnl. The result was that
# every PnL, fill-count and win-rate readout below was permanently dead on real
# engine output -- the guards all fall through to "--" rather than erroring, so
# it looked like an idle engine instead of a broken read.
#
# Aliasing on read, rather than renaming the ~13 downstream references, keeps
# both writers working and leaves one place to look when a third one appears.

ORDER_PNL_ALIASES = ("net_pnl", "pnl_net", "realized_pnl")
ORDER_FILL_FLAGS = ("filled", "approved")

# All three producers spell the adverse-fill flag the same way and write it as
# 0/1 -- execution_manager.cpp:354, live_ingestion.py's ORDERS_HEADER,
# seed_demo_feed.py's SCHEMAS -- so there is no alias list to keep here. What
# does vary is whether the column is there at all: any order log written before
# it was added has 23 or 17 columns and none of them is this one.
#
# Hence two names. `adverse` is whatever the writer left in the file, and its
# presence is the only thing that distinguishes "this fill was not adverse" from
# "this file cannot say". `adverse_fill` is derived, always present, always
# boolean, and is what the panels below read.
ORDER_ADVERSE_COL = "adverse"

def _flag(col):
    """A 0/1 (or true/false) flag column as booleans, blanks counting as False.

    Every writer in this repo emits 0/1, but pd.to_numeric turns the string
    "True" into NaN and then into False -- which for the adverse flag would
    report a booked loss as a clean fill, the exact reading this column exists
    to prevent. Accepting both spellings is cheaper than relying on the next
    writer to match the current three.
    """
    num = pd.to_numeric(col, errors="coerce").fillna(0) != 0
    words = col.astype(str).str.strip().str.lower().isin(("true", "t", "yes", "y"))
    return num | words

def normalize_orders(df):
    """Derive the two columns the panels read by name: pnl_realized, adverse_fill.

    The derivations are guarded one at a time rather than behind a single early
    return, because they were added at different times: a file can carry a PnL
    column and no adverse flag, and one that has already been normalized must
    not skip the newer half.
    """
    if df is None or len(df) == 0:
        return df
    if "pnl_realized" not in getattr(df, "columns", []):
        for src in ORDER_PNL_ALIASES:
            if src not in df.columns:
                continue
            pnl = pd.to_numeric(df[src], errors="coerce")
            # A rejected attempt is logged with net_pnl = 0.0, which is not the
            # same statement as a fill that happened to break even. Blank those
            # rows so the fill and win-rate counters below count attempts, not
            # rows.
            for flag in ORDER_FILL_FLAGS:
                if flag in df.columns:
                    pnl = pnl.where(_flag(df[flag]))
                    break
            df = df.copy()
            df["pnl_realized"] = pnl
            break
    if "adverse_fill" not in getattr(df, "columns", []):
        df = df.copy()
        if ORDER_ADVERSE_COL in df.columns:
            adverse = _flag(df[ORDER_ADVERSE_COL])
            # A rejection carries adverse = 0 from every writer, so this is
            # belt-and-braces -- but an adverse fill is by definition a fill,
            # and the panels treat the two as disjoint.
            for flag in ORDER_FILL_FLAGS:
                if flag in df.columns:
                    adverse = adverse & _flag(df[flag])
                    break
        else:
            adverse = pd.Series(False, index=df.index)
        df["adverse_fill"] = adverse.astype(bool)
    return df

def dot(ok):
    return f'<span style="display:inline-block;width:5px;height:5px;border-radius:50%;background:{"#22c55e" if ok else "#ef4444"};vertical-align:middle;"></span>'

def section(title):
    st.markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:0.5rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.15rem;">{title}</div>', unsafe_allow_html=True)

# --- Feed freshness -------------------------------------------------------
# A feed counts as live only while its last heartbeat is recent. Without this
# ceiling an age counter keeps incrementing after the producer dies, which reads
# as a running clock rather than a dead feed.
STALE_AFTER_S = 10.0

def fmt_age(age_s):
    """Feed age, or '--' once it is past the staleness ceiling."""
    if age_s is None or age_s < 0 or age_s > STALE_AFTER_S:
        return "--"
    return f"{age_s:.1f}s"

def fmt_clock(ts_ms):
    if not ts_ms or ts_ms <= 0:
        return "--"
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%H:%M:%S UTC")

def _num(row, key, default=0.0):
    try:
        v = pd.to_numeric(row.get(key, default), errors="coerce")
        return default if pd.isna(v) else float(v)
    except Exception:
        return default

# --- Initial data read (outside fragment) for symbol discovery & radio ---
# Only the reads needed to build the radio options live here. Everything else
# is re-read inside the fragment on each tick.
status = read_csv(STATUS_PATH)
if status is None:
    st.warning("Crypto engine offline -- start `live_trading_binance_kraken` to connect. Equity feeds are still shown below if `src/live_ingestion.py` is running.")

signals_py_all = read_csv(SIGNALS_PY_PATH)


def pretty_symbol(sym, asset_class):
    """btcusdt -> BTC/USDT, aapl -> AAPL, BTC/USD -> BTC/USD.

    Two separate double-slash traps here, both producing labels like "BTC//USD":

    1. Chained .replace("USDT","/USDT").replace("USD","/USD") double-inserts
       because "USD" is a prefix of "USDT". Fixed by matching the longest quote
       asset first and stopping.
    2. The symbol may arrive already separated. src/live_ingestion.py writes
       "BTC/USD" literally, so appending a slash to a string that has one is the
       common case, not the edge case. Fixed by returning such symbols unchanged.
    """
    u = str(sym).upper()
    # Already separated by any conventional delimiter -- normalise case only.
    for sep in ("/", "-", "_", ":"):
        if sep in u:
            return u.replace(sep, "/")
    for quote in ("USDT", "USDC", "USD"):
        if u.endswith(quote) and len(u) > len(quote):
            return f"{u[:-len(quote)]}/{quote}"
    if asset_class == "crypto":
        return f"{u}/USDT"
    return u

symbol_options = [{"symbol": "btcusdt", "asset_class": "crypto", "label": "BTC/USDT"}]
if signals_py_all is not None and "symbol" in signals_py_all.columns:
    sym_groups = signals_py_all.groupby(["symbol", "asset_class"]).size().reset_index()
    opts = []
    for _, row in sym_groups.iterrows():
        opts.append({
            "symbol": row["symbol"],
            "asset_class": row["asset_class"],
            "label": pretty_symbol(row["symbol"], row["asset_class"]),
        })
    if opts:
        symbol_options = opts

sym_labels = [o["label"] for o in symbol_options]

# ---------------------------------------------------------------------------
# Everything data-dependent lives inside this fragment. On each tick only the
# fragment re-executes; the static page skeleton (CSS, helper functions) stays
# intact in the DOM, eliminating the full-page teardown jitter that
# st_autorefresh caused. The radio widget is inside the fragment intentionally:
# a widget click triggers only a fragment rerun, which is all that's needed.
# ---------------------------------------------------------------------------
@st.fragment(run_every=timedelta(seconds=2.5))
def _live_dashboard():
    # Sanitise stored radio selection before the widget reads it.
    if st.session_state.get("asset_sel") not in sym_labels:
        st.session_state["asset_sel"] = sym_labels[0]

    sel_label = st.radio("Asset", sym_labels, horizontal=True,
                         label_visibility="collapsed", key="asset_sel")
    selected_sym_info = next(o for o in symbol_options if o["label"] == sel_label)
    selected_symbol = selected_sym_info["symbol"]
    is_crypto = selected_sym_info["asset_class"] == "crypto"

    # Re-read CSVs fresh each tick so the numbers update.
    status = read_csv(STATUS_PATH)
    engine_row = None
    if status is not None and len(status) > 0:
        if "timestamp_ms" in status.columns:
            ts_col = pd.to_numeric(status["timestamp_ms"], errors="coerce")
            if ts_col.notna().any():
                engine_row = status.loc[ts_col.idxmax()]
        if engine_row is None:
            engine_row = status.iloc[-1]

    now_ms = time.time() * 1000

    if engine_row is not None:
        engine_ts = _num(engine_row, "timestamp_ms", 0.0)
        engine_age_s = (now_ms - engine_ts) / 1000 if engine_ts > 0 else None
        engine_status = str(engine_row.get("status", ""))
        engine_demo = engine_status == "demo"
        engine_live = (
            engine_status in ("running", "demo")
            and engine_age_s is not None
            and engine_age_s <= STALE_AFTER_S
        )
        realized_pnl = _num(engine_row, "realized_pnl", 0.0)
        current_pos = _num(engine_row, "position", 0.0)
        total_orders_stat = int(_num(engine_row, "total_orders", 0.0))
        filled_orders_stat = int(_num(engine_row, "filled_orders", 0.0))
    else:
        engine_ts, engine_age_s, engine_live = 0.0, None, False
        engine_demo = False
        realized_pnl = current_pos = 0.0
        total_orders_stat = filled_orders_stat = 0

    prices_all = read_csv(PRICES_PATH)
    signals_all = read_csv(SIGNALS_PATH)
    if signals_all is None:
        signals_all = read_csv(SIGNALS_PY_PATH)
    orders_all = normalize_orders(read_csv(ORDERS_PATH))
    if orders_all is None:
        orders_all = normalize_orders(read_csv(ORDERS_PY_PATH))
    signals_py_all = read_csv(SIGNALS_PY_PATH)

    feed_df = signals_py_all[signals_py_all["symbol"] == selected_symbol].copy() if signals_py_all is not None and "symbol" in signals_py_all.columns else pd.DataFrame()
    prices_filt = prices_all if prices_all is not None else pd.DataFrame()
    if prices_all is not None and "symbol" in prices_all.columns:
        prices_filt = prices_all[prices_all["symbol"] == selected_symbol]
    orders_filt = orders_all if orders_all is not None else pd.DataFrame()
    if orders_all is not None and "symbol" in orders_all.columns:
        orders_filt = orders_all[orders_all["symbol"] == selected_symbol]
    signals_filt = signals_all if signals_all is not None else pd.DataFrame()
    if signals_all is not None and "symbol" in signals_all.columns:
        signals_filt = signals_all[signals_all["symbol"] == selected_symbol]

    total_signals = len(signals_filt) if signals_filt is not None else 0
    avg_obi_delta = 0.0
    if signals_filt is not None and len(signals_filt) > 0 and "obi_delta" in signals_filt.columns:
        avg_obi_delta = abs(signals_filt["obi_delta"]).mean()

    sig_rate = 0
    if signals_filt is not None and len(signals_filt) > 2 and "timestamp_ms" in signals_filt.columns:
        span_s = (signals_filt["timestamp_ms"].iloc[-1] - signals_filt["timestamp_ms"].iloc[0]) / 1000
        sig_rate = total_signals / span_s if span_s > 0 else 0

    orders_has_pnl = orders_filt is not None and "pnl_realized" in orders_filt.columns
    # Whether the file can speak about adverse fills at all, as opposed to reporting
    # none. Older logs have no such column and must read "--" rather than "0".
    orders_has_adverse = (orders_filt is not None
                          and ORDER_ADVERSE_COL in getattr(orders_filt, "columns", []))
    total_attempts_all = len(orders_filt) if orders_filt is not None else 0
    win_rate = sharpe = mean_pnl = 0.0
    filled_orders_count = 0
    wins_count = 0
    adverse_count = 0
    adverse_mismatch = 0
    if orders_has_pnl:
        if "timestamp_ms" in orders_filt.columns:
            orders_filt["time"] = pd.to_datetime(orders_filt["timestamp_ms"], unit="ms", utc=True)
        filled_df = orders_filt[orders_filt["pnl_realized"].notna() & (orders_filt["pnl_realized"] != 0)]
        filled_orders_count = len(filled_df)
        if filled_orders_count > 0:
            wins_count = len(filled_df[filled_df["pnl_realized"] > 0])
            win_rate = wins_count / filled_orders_count * 100
            mean_pnl = filled_df["pnl_realized"].mean()
            std_pnl = filled_df["pnl_realized"].std()
            if std_pnl > 0:
                sharpe = mean_pnl / std_pnl * (252 * 6.5 * 3600 / 3) ** 0.5
            if orders_has_adverse:
                adverse_count = int(filled_df["adverse_fill"].sum())
                # The flag is the writer's own verdict; the sign of the PnL is this
                # page re-deriving it. Every producer sets adverse = net_pnl < 0.0,
                # so the two agree by construction and a disagreement is news: either
                # a writer's definition moved, or a column did. This project has
                # already had one silent column shift (tests/test_demo_feed_schema.py),
                # and the symptom of the next one is exactly this count going
                # non-zero, so it is counted rather than assumed away.
                adverse_mismatch = int((filled_df["adverse_fill"]
                                        != (filled_df["pnl_realized"] < 0)).sum())

    signals_chart = signals_filt.tail(500).copy() if signals_filt is not None and len(signals_filt) > 0 else pd.DataFrame()
    orders_chart = orders_filt.tail(500).copy() if orders_filt is not None and len(orders_filt) > 0 else pd.DataFrame()
    prices_chart = prices_filt.tail(3000) if prices_filt is not None and len(prices_filt) > 3000 else prices_filt

    if prices_chart is not None and len(prices_chart) > 0 and "timestamp_ms" in prices_chart.columns:
        if "time" not in prices_chart.columns:
            prices_chart["time"] = pd.to_datetime(prices_chart["timestamp_ms"], unit="ms", utc=True)
        b_ex = prices_chart[prices_chart["exchange"] == "binance"] if "exchange" in prices_chart.columns else pd.DataFrame()
        k_ex = prices_chart[prices_chart["exchange"] == "kraken"] if "exchange" in prices_chart.columns else pd.DataFrame()
        # fmt_age caps these: a dead venue reads "--" instead of counting up forever.
        gap_b = fmt_age((pd.Timestamp.now(tz="UTC") - b_ex["time"].max()).total_seconds()) if len(b_ex) > 0 else "--"
        gap_k = fmt_age((pd.Timestamp.now(tz="UTC") - k_ex["time"].max()).total_seconds()) if len(k_ex) > 0 else "--"
    else:
        gap_b = gap_k = "--"

    # --- Active-feed status ---------------------------------------------------
    # Crypto is served by the C++ engine (/tmp/live_status.csv); equities are served
    # by src/live_ingestion.py (/tmp/live_signals_py.csv). Report whichever actually
    # backs the current selection, so switching asset class switches the clock too.
    if is_crypto:
        feed_ts_ms, feed_age_s, feed_live = engine_ts, engine_age_s, engine_live
        feed_src = "ENGINE"
        feed_demo = engine_demo
    else:
        feed_ts_ms = (
            float(pd.to_numeric(feed_df["timestamp_ms"], errors="coerce").max())
            if feed_df is not None and len(feed_df) > 0 and "timestamp_ms" in feed_df.columns
            else 0.0
        )
        feed_age_s = (now_ms - feed_ts_ms) / 1000 if feed_ts_ms > 0 else None
        feed_live = feed_age_s is not None and 0 <= feed_age_s <= STALE_AFTER_S
        feed_src = "FEED"
        feed_demo = False

    # Every downstream `running` check means "is the feed behind the current
    # selection live", not "did the crypto engine ever write a row".
    running = feed_live

    # A synthetic feed says so, in the one place nobody can miss. This link is
    # shareable, so an unlabelled demo would be someone else's screenshot of a
    # trading system that made money.
    #
    # Provenance and liveness are reported separately and deliberately. Folding them
    # into one word forces a choice between two wrong answers when the demo producer
    # dies: "Demo data" implies something is still writing, and "Stale" alone drops
    # the fact that every number on screen is synthetic. So the status word stays
    # purely about liveness, and the DEMO chip stays up for as long as the rows it
    # describes are still the ones being rendered.
    feed_word = "Live" if feed_live else "Stale"

    # Written as one unindented line per element, not as an indented block.
    #
    # This markup used to be a pretty-printed triple-quoted f-string, and it printed
    # its own tags on screen whenever feed_demo was False. unsafe_allow_html was set
    # the whole time; the flag was never the problem. Streamlit renders markdown
    # (CommonMark, via markdown-it-py) before it allows the HTML through, and an
    # interpolation that is the only thing on its line leaves a whitespace-only line
    # behind when it evaluates to "" -- which closes the HTML block. Every following
    # line was indented eight spaces to look tidy, so once the block had closed those
    # lines became an indented code block, and the status indicator's own <span>s were
    # escaped and printed as text.
    #
    # Two rules keep that shut, and both matter: nothing here is indented, and no
    # element sits alone on a line where it can vanish and leave the line blank.
    demo_chip = ('<span style="font-size:0.45rem;color:#eab308;padding:0.05rem 0.3rem;'
                 'border:1px solid #6b5410;border-radius:2px;letter-spacing:0.05em;">'
                 'DEMO</span>') if feed_demo else ""
    
    col_title, col_status = st.columns([4, 1])
    with col_title:
        st.markdown(
            "### CROSSFLUX <span style='font-size: 0.75rem; color: #6fa8bb;'>v0.1 — Microstructure Execution Engine</span>", 
            unsafe_allow_html=True
        )
    with col_status:
        st.markdown(
            '<div style="display:flex;flex-direction:column;align-items:flex-end;gap:0.15rem;'
            'font-family:JetBrains Mono,monospace;font-size:0.55rem;color:#6b7280;margin-top:0.3rem;">'
            f'<div style="display:flex;align-items:center;gap:0.3rem;">{demo_chip} {dot(feed_live)} {feed_word} <span style="color:#4b5563;">{sel_label}</span></div>'
            f'<div>UPD {fmt_clock(feed_ts_ms) if feed_live else "--"} | {feed_src} {fmt_age(feed_age_s)}</div>'
            '</div>',
            unsafe_allow_html=True)
    st.markdown('<hr style="margin: 0 0 0.35rem 0; opacity: 0.06; border: 0; border-top: 1px solid #2b3139;">', unsafe_allow_html=True)

    if feed_demo:
        st.markdown(
            "<div style='font-size:0.6rem;color:#eab308;border:1px solid #3a2f0b;"
            "background:#1a1608;border-radius:3px;padding:0.25rem 0.45rem;"
            "margin-bottom:0.4rem;'>Synthetic feed from "
            "<code>dashboard/seed_demo_feed.py</code> &mdash; a random walk, not "
            "recorded market data. Every price, signal and PnL figure below is "
            "generated. Start the C++ engine for real numbers.</div>",
            unsafe_allow_html=True)

    section("System Status")

    if is_crypto:
        h1, h2, h3, h4, h5 = st.columns(5)
        h1.metric("Circuit Breaker", "Open" if running else "Tripped")
        h2.metric("Binance", gap_b)
        h3.metric("Kraken", gap_k)
        h4.metric("Signals/s", f"{sig_rate:.1f}" if sig_rate else "0.0")
        h5.metric("Position", f"{current_pos:.4f}")
    else:
        # Position/PnL in /tmp/live_status.csv belong to the crypto arb engine, so
        # they are deliberately not surfaced here -- there is no equities position.
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Feed Status", feed_word)
        h2.metric("Feed Delay", fmt_age(feed_age_s))
        h3.metric("Updates/s", f"{sig_rate:.1f}" if sig_rate else "0.0")
        h4.metric("Source", "Alpaca IEX")

    section("Market")

    col_a, col_b = st.columns([1.3, 1])

    with col_a:
        if is_crypto:
            ov1, ov2, ov3, ov4 = st.columns(4)
            ov1.metric("Signals", f"{total_signals:,}" if total_signals else "--", f"{avg_obi_delta:.4f} avg obi" if avg_obi_delta else None)
            ov2.metric("PnL", f"${realized_pnl:.2f}" if running else "--", f"{filled_orders_count:,} fills" if running else None, delta_color="normal")
            ov3.metric("Win Rate", f"{win_rate:.1f}%" if win_rate else "--", f"{mean_pnl:.4f} avg" if mean_pnl else None)
            ov4.metric("Cooldown", "5s", f"S {sharpe:.2f}" if sharpe else None)
        else:
            feed_ticks = len(feed_df) if feed_df is not None else 0
            feed_rate = 0
            if feed_df is not None and len(feed_df) > 2 and "timestamp_ms" in feed_df.columns:
                span_s = (feed_df["timestamp_ms"].iloc[-1] - feed_df["timestamp_ms"].iloc[0]) / 1000
                feed_rate = feed_ticks / span_s if span_s > 0 else 0
            feed_latest = feed_df.iloc[-1] if feed_df is not None and len(feed_df) > 0 else None
            spread_val = spread_bps = 0
            if feed_latest is not None and "bid_price" in feed_latest.index and "ask_price" in feed_latest.index:
                bp, ap = float(feed_latest["bid_price"]), float(feed_latest["ask_price"])
                if bp > 0 and ap > 0:
                    spread_val = ap - bp
                    spread_bps = spread_val / bp * 10000
            ov1, ov2, ov3, ov4 = st.columns(4)
            ov1.metric("Updates", f"{feed_ticks:,}" if feed_ticks else "--", f"{feed_rate:.1f}/s" if feed_rate else None)
            ov2.metric("Spread", f"${spread_val:.4f}" if spread_val else "--", f"{spread_bps:.2f} bps" if spread_bps else None, delta_color="normal")
            mid_val = (float(feed_latest["bid_price"]) + float(feed_latest["ask_price"])) / 2 if feed_latest is not None and "bid_price" in feed_latest.index and "ask_price" in feed_latest.index else None
            ov3.metric("Mid", f"${mid_val:.2f}" if mid_val else "--")
            ov4.metric("Symbol", selected_symbol.upper())

    with col_b:
        if is_crypto and prices_chart is not None and len(prices_chart) > 0:
            px = prices_chart.tail(200).copy()
            fig = go.Figure()
            for label, key, clr in [("Binance", "binance", "#eab308"), ("Kraken", "kraken", "#3b82f6")]:
                d = px[px["exchange"] == key] if "exchange" in px.columns else px
                if len(d) > 1:
                    mid = (d["bid_price"] + d["ask_price"]) / 2
                    fig.add_trace(go.Scatter(x=d["time"], y=mid, mode="lines", name=label, line=dict(color=clr, width=0.8)))
            fig.update_layout(template="plotly_dark", height=120, margin=dict(l=0, r=0, t=0, b=0), showlegend=True, legend=dict(orientation="h", y=1.02, x=1, xanchor="right", font=dict(size=8, color="#6b7280")), hovermode="x unified", paper_bgcolor="#0f1217", plot_bgcolor="#0f1217")
            fig.update_xaxes(showgrid=False, visible=False)
            fig.update_yaxes(showgrid=True, gridcolor="#1e2329", griddash="dot", zeroline=False)
            st.plotly_chart(fig, width='stretch')
        elif not is_crypto and feed_df is not None and len(feed_df) > 20:
            fd = feed_df.tail(200).copy()
            fd["mid_price"] = (fd["bid_price"] + fd["ask_price"]) / 2
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pd.to_datetime(fd["timestamp_ms"], unit="ms", utc=True), y=fd["mid_price"], mode="lines", line=dict(color="#3b82f6", width=0.8)))
            fig.update_layout(template="plotly_dark", height=120, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, hovermode="x unified", paper_bgcolor="#0f1217", plot_bgcolor="#0f1217")
            fig.update_xaxes(showgrid=False, visible=False)
            fig.update_yaxes(showgrid=True, gridcolor="#1e2329", griddash="dot", zeroline=False)
            st.plotly_chart(fig, width='stretch')
        else:
            st.markdown("<div style='font-size:0.6rem;color:#6b7280;height:120px;display:flex;align-items:center;justify-content:center;'>No data</div>", unsafe_allow_html=True)

    section("Order Book")

    if is_crypto:
        book_cols = st.columns(2)
        for i, (label, key, color) in enumerate([("Binance  BTC/USDT", "binance", "#eab308"), ("Kraken  XBT/USD", "kraken", "#3b82f6")]):
            with book_cols[i]:
                q = None
                if prices_chart is not None and len(prices_chart) > 0 and "exchange" in prices_chart.columns:
                    row = prices_chart[prices_chart["exchange"] == key]
                    q = row.iloc[-1] if len(row) > 0 else None
                st.markdown(f"<div style='font-family:JetBrains Mono,monospace;font-size:0.55rem;color:#6b7280;border-bottom:1px solid #1e2329;padding-bottom:0.15rem;margin-bottom:0.2rem;'>{label}</div>", unsafe_allow_html=True)
                if q is not None:
                    bp, bv = q["bid_price"], q["bid_vol"]
                    ap, av = q["ask_price"], q["ask_vol"]
                    spread = ap - bp
                    obi = calc_obi(bv, av)
                    mid = (bp + ap) / 2 if bp > 0 and ap > 0 else bp or ap
                    r1, r2 = st.columns(2)
                    r1.metric("Bid", f"${bp:,.2f}" if bp > 0 else "--", f"{bv:.4f}" if bv > 0 else None)
                    r2.metric("Ask", f"${ap:,.2f}" if ap > 0 else "--", f"{av:.4f}" if av > 0 else None)
                    r3, r4, r5 = st.columns(3)
                    r3.metric("Mid", f"${mid:,.2f}" if mid > 0 else "--")
                    r4.metric("Spread", f"${spread:.2f}" if spread > 0 else "--", f"{spread / bp * 10000:.1f} bps" if spread > 0 and bp > 0 else None)
                    r5.metric("OBI", f"{obi:+.4f}")
                    depth_df = read_csv(DEPTH_PATH)
                    if depth_df is not None and len(depth_df) > 0:
                        ex_depth = depth_df[depth_df["exchange"] == key]
                        if len(ex_depth) > 0:
                            lt = ex_depth["timestamp_ms"].max()
                            ex_latest = ex_depth[ex_depth["timestamp_ms"] == lt]
                            bids_d = ex_latest[ex_latest["side"] == 0].sort_values("level")
                            asks_d = ex_latest[ex_latest["side"] == 1].sort_values("level")
                            bp_arr, bv_arr = bids_d["price"].tolist(), bids_d["volume"].tolist()
                            ap_arr, av_arr = asks_d["price"].tolist(), asks_d["volume"].tolist()
                            max_vol = max(bv_arr + av_arr) if bv_arr or av_arr else 1
                            fig = go.Figure()
                            fig.add_trace(go.Bar(x=bp_arr[::-1] + ap_arr, y=bv_arr[::-1] + av_arr,
                                marker_color=[color] * len(bp_arr + ap_arr),
                                marker_opacity=[0.15 + 0.4 * (v / max_vol) for v in bv_arr[::-1] + av_arr], width=0.4))
                            fig.update_layout(template="plotly_dark", height=80, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, paper_bgcolor="#0f1217", plot_bgcolor="#0f1217")
                            fig.update_xaxes(visible=False, showgrid=False)
                            fig.update_yaxes(visible=False, showgrid=False)
                            st.plotly_chart(fig, width='stretch')
                            ld = []
                            for j in range(max(len(bp_arr), len(ap_arr))):
                                ld.append({"": str(j + 1),
                                    "Bid": f"${bp_arr[j]:,.2f}" if j < len(bp_arr) else "--",
                                    "Bid Vol": f"{bv_arr[j]:.4f}" if j < len(bv_arr) else "--",
                                    "Ask": f"${ap_arr[j]:,.2f}" if j < len(ap_arr) else "--",
                                    "Ask Vol": f"{av_arr[j]:.4f}" if j < len(av_arr) else "--"})
                            st.dataframe(pd.DataFrame(ld), width='stretch', hide_index=True, column_config={"": st.column_config.NumberColumn(width=20)})
                    else:
                        st.caption("Loading depth...")
                else:
                    st.markdown("<div style='font-size:0.6rem;color:#6b7280;'>Waiting</div>", unsafe_allow_html=True)
    else:
        col_q, col_p = st.columns([1, 1.3])
        with col_q:
            st.markdown(f"<div style='font-family:JetBrains Mono,monospace;font-size:0.55rem;color:#6b7280;border-bottom:1px solid #1e2329;padding-bottom:0.15rem;margin-bottom:0.2rem;'>Alpaca IEX  {selected_symbol.upper()}</div>", unsafe_allow_html=True)
            if feed_df is not None and len(feed_df) > 0:
                q = feed_df.iloc[-1]
                bp = float(q["bid_price"]) if "bid_price" in q.index else 0
                ap = float(q["ask_price"]) if "ask_price" in q.index else 0
                bv = float(q["bid_qty"]) if "bid_qty" in q.index else 0
                av = float(q["ask_qty"]) if "ask_qty" in q.index else 0
                if bp > 0 or ap > 0:
                    spread = ap - bp
                    obi = calc_obi(bv, av)
                    mid = (bp + ap) / 2 if bp > 0 and ap > 0 else bp or ap
                    r1, r2 = st.columns(2)
                    r1.metric("Bid", f"${bp:,.2f}" if bp > 0 else "--", f"{bv:.4f}" if bv > 0 else None)
                    r2.metric("Ask", f"${ap:,.2f}" if ap > 0 else "--", f"{av:.4f}" if av > 0 else None)
                    r3, r4, r5 = st.columns(3)
                    r3.metric("Mid", f"${mid:,.2f}" if mid > 0 else "--")
                    r4.metric("Spread", f"${spread:.4f}" if spread > 0 else "--", f"{spread / bp * 10000:.1f} bps" if spread > 0 and bp > 0 else None)
                    r5.metric("OBI", f"{obi:+.4f}")
                else:
                    st.markdown("<div style='font-size:0.6rem;color:#6b7280;'>No quote</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size:0.6rem;color:#6b7280;'>No feed</div>", unsafe_allow_html=True)
        with col_p:
            st.markdown(f"<div style='font-family:JetBrains Mono,monospace;font-size:0.55rem;color:#6b7280;border-bottom:1px solid #1e2329;padding-bottom:0.15rem;margin-bottom:0.2rem;'>Price  {selected_symbol.upper()}</div>", unsafe_allow_html=True)
            if feed_df is not None and len(feed_df) > 20:
                fd = feed_df.tail(200).copy()
                fd["mid_price"] = (fd["bid_price"] + fd["ask_price"]) / 2
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=pd.to_datetime(fd["timestamp_ms"], unit="ms", utc=True), y=fd["mid_price"], mode="lines", line=dict(color="#3b82f6", width=0.8)))
                fig.update_layout(template="plotly_dark", height=160, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, hovermode="x unified", paper_bgcolor="#0f1217", plot_bgcolor="#0f1217")
                fig.update_xaxes(showgrid=False, visible=False)
                fig.update_yaxes(showgrid=True, gridcolor="#1e2329", griddash="dot", zeroline=False)
                st.plotly_chart(fig, width='stretch')
                fd["spread_bps"] = np.where((fd["bid_price"] > 0) & (fd["ask_price"] > 0), (fd["ask_price"] - fd["bid_price"]) / fd["bid_price"] * 10000, np.nan)
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=pd.to_datetime(fd["timestamp_ms"], unit="ms", utc=True), y=fd["spread_bps"], mode="lines", line=dict(color="#eab308", width=0.6)))
                fig2.add_hline(y=fd["spread_bps"].mean(), line_dash="dot", line_color="#2b3139")
                fig2.update_layout(template="plotly_dark", height=100, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, hovermode="x unified", paper_bgcolor="#0f1217", plot_bgcolor="#0f1217")
                fig2.update_xaxes(showgrid=False, visible=False)
                fig2.update_yaxes(showgrid=True, gridcolor="#1e2329", griddash="dot", zeroline=False)
                st.plotly_chart(fig2, width='stretch')
            else:
                st.markdown("<div style='font-size:0.6rem;color:#6b7280;height:160px;display:flex;align-items:center;justify-content:center;'>Collecting</div>", unsafe_allow_html=True)

    if is_crypto and prices_chart is not None and len(prices_chart) > 0 and "bid_vol" in prices_chart.columns:
        section("Order Book Imbalance")
        pv = prices_chart.tail(2000).copy()
        pv["obi_val"] = np.where((pv["bid_vol"] + pv["ask_vol"]) > 0, (pv["bid_vol"] - pv["ask_vol"]) / (pv["bid_vol"] + pv["ask_vol"]), np.nan)
        fig = go.Figure()
        for label, key, clr in [("Binance", "binance", "#eab308"), ("Kraken", "kraken", "#3b82f6")]:
            d = pv[pv["exchange"] == key].dropna(subset=["obi_val"]).tail(200)
            if len(d) > 1:
                fig.add_trace(go.Scatter(x=d["time"], y=d["obi_val"], mode="lines", name=label, line=dict(color=clr, width=0.8)))
        fig.add_hline(y=0, line_dash="dot", line_color="#2b3139")
        fig.update_layout(template="plotly_dark", height=80, margin=dict(l=0, r=0, t=0, b=0), showlegend=True, legend=dict(orientation="h", y=1.02, x=1, xanchor="right", font=dict(size=7, color="#6b7280")), hovermode="x unified", paper_bgcolor="#0f1217", plot_bgcolor="#0f1217")
        fig.update_xaxes(showgrid=False, visible=False)
        fig.update_yaxes(showgrid=True, gridcolor="#1e2329", griddash="dot", zeroline=False)
        st.plotly_chart(fig, width='stretch')

    tab_trade, tab_market, tab_raw = st.tabs(["Orders", "Market", "Data"])

    with tab_trade:
        if is_crypto and orders_chart is not None and len(orders_chart) > 0:
            has_fill = "pnl_realized" in orders_chart.columns
            if has_fill:
                fc = filled_df.tail(500) if filled_orders_count > 0 else orders_filt[orders_filt["pnl_realized"].notna() & (orders_filt["pnl_realized"] != 0)].tail(500)
                tr1, tr2, tr3, tr4 = st.columns(4)
                tr1.metric("Total Orders", f"{total_attempts_all:,}", f"{filled_orders_count:,} filled" if filled_orders_count > 0 else None)
                tr2.metric("Win / Loss", f"{wins_count:,} / {filled_orders_count - wins_count:,}" if filled_orders_count > 0 else "--", f"{win_rate:.1f}%" if win_rate else None)
                tr3.metric("Avg PnL", f"${mean_pnl:.4f}" if mean_pnl else "--")
                # Adverse fills are losses the execution model booked rather than
                # declined, which is the behaviour the friction rework exists to make
                # visible (TEST_REPORT.md 2.1). "--" where the log predates the flag:
                # an old file reporting "0 adverse" would be the more misleading of
                # the two readings.
                tr4.metric("Adverse", f"{adverse_count:,}" if orders_has_adverse else "--",
                           f"{adverse_count / filled_orders_count * 100:.1f}% of fills"
                           if orders_has_adverse and filled_orders_count > 0 else None)
                if len(fc) > 1:
                    t_sub1, t_sub2 = st.tabs(["Trades", "PnL"])
                    with t_sub1:
                        display = orders_chart.tail(100).iloc[::-1].reset_index(drop=True)
                        cols = ["time", "action", "vwap_buy", "vwap_sell", "fill_qty", "pnl_realized", "slippage_bps"]
                        keep = [c for c in cols if c in display.columns]
                        disp = display[keep].copy()
                        if "pnl_realized" in disp.columns:
                            # A rejection has no PnL, and pnl_realized is NaN there
                            # by design. Formatting that with "${:+.4f}" prints
                            # "$+nan" on every rejected row, which in a panel whose
                            # job is to make booked losses legible is noise sitting
                            # next to the signal.
                            disp["pnl_realized"] = ["--" if pd.isna(v) else f"${v:+.4f}"
                                                    for v in disp["pnl_realized"]]
                        if "slippage_bps" in disp.columns:
                            disp["slippage_bps"] = disp["slippage_bps"].map("{:.1f} bps".format)
                        if "vwap_buy" in disp.columns:
                            disp["vwap_buy"] = disp["vwap_buy"].map("${:.2f}".format)
                            disp["vwap_sell"] = disp["vwap_sell"].map("${:.2f}".format)
                        labels = {"time": "Time", "action": "Dir", "vwap_buy": "Buy", "vwap_sell": "Sell", "fill_qty": "Qty", "pnl_realized": "PnL", "slippage_bps": "Slip"}
                        disp.columns = [labels.get(c, c) for c in disp.columns]
                        # The marker goes in a column of its own rather than into the
                        # PnL cell: st.dataframe renders no HTML, so a red PnL is not
                        # available here, and a glyph in the leftmost column is what
                        # makes a booked loss findable while scrolling. Only added
                        # when the file carries the flag -- an all-blank column would
                        # read as "no adverse fills" on a log that cannot say.
                        if orders_has_adverse and "adverse_fill" in display.columns:
                            disp.insert(0, "Adv", ["🔻" if a else "" for a in display["adverse_fill"]])
                        st.dataframe(disp, width='stretch', hide_index=True)
                        if adverse_mismatch:
                            st.markdown(
                                f"<div style='font-size:0.6rem;color:#eab308;'>"
                                f"{adverse_mismatch} filled order(s) where the adverse flag "
                                f"and the sign of the net PnL disagree -- check the order log's "
                                f"column alignment.</div>", unsafe_allow_html=True)
                    with t_sub2:
                        col_p, col_h = st.columns([1.2, 1])
                        with col_p:
                            fs = fc.sort_values("time") if "time" in fc.columns else fc
                            fs["cum_pnl"] = fs["pnl_realized"].cumsum()
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=fs["time"] if "time" in fs.columns else range(len(fs)), y=fs["cum_pnl"], mode="lines", line=dict(color="#eab308", width=1), fill="tozeroy", fillcolor="rgba(234,179,8,0.04)"))
                            fig.add_hline(y=0, line_dash="dot", line_color="#2b3139")
                            fig.update_layout(template="plotly_dark", height=180, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#0f1217", plot_bgcolor="#0f1217")
                            fig.update_xaxes(showgrid=True, gridcolor="#1e2329", griddash="dot")
                            fig.update_yaxes(showgrid=True, gridcolor="#1e2329", griddash="dot")
                            st.plotly_chart(fig, width='stretch')
                        with col_h:
                            fig = go.Figure()
                            fig.add_trace(go.Histogram(x=fc["pnl_realized"], nbinsx=25, marker_color="#eab308", marker_line_width=0))
                            fig.update_layout(template="plotly_dark", height=180, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#0f1217", plot_bgcolor="#0f1217")
                            fig.update_xaxes(showgrid=False)
                            fig.update_yaxes(showgrid=True, gridcolor="#1e2329", griddash="dot")
                            st.plotly_chart(fig, width='stretch')
            else:
                st.dataframe(orders_chart.tail(100).iloc[::-1], width='stretch', hide_index=True)
        elif is_crypto:
            st.markdown("<div style='font-size:0.65rem;color:#6b7280;'>Awaiting execution data.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:0.65rem;color:#6b7280;'>Trade execution available for crypto pairs.</div>", unsafe_allow_html=True)

    with tab_market:
        if is_crypto and signals_chart is not None and len(signals_chart) > 0 and "timestamp_ms" in signals_chart.columns:
            signals_chart["time"] = pd.to_datetime(signals_chart["timestamp_ms"], unit="ms", utc=True)
            ms1, ms2, ms3, ms4, ms5 = st.columns(5)
            buy_a = len(signals_chart[signals_chart["action"] == 0]) if "action" in signals_chart.columns else 0
            buy_b = len(signals_chart[signals_chart["action"] == 1]) if "action" in signals_chart.columns else 0
            pos_delta = len(signals_chart[signals_chart["obi_delta"] > 0]) if "obi_delta" in signals_chart.columns else 0
            neg_delta = len(signals_chart[signals_chart["obi_delta"] < 0]) if "obi_delta" in signals_chart.columns else 0
            ms1.metric("Total", f"{total_signals:,}")
            ms2.metric("B->K", f"{buy_a:,}")
            ms3.metric("K->B", f"{buy_b:,}")
            ms4.metric("+/-", f"{pos_delta}/{neg_delta}")
            ms5.metric("Avg|OBI|", f"{avg_obi_delta:.4f}")

            # Which weight profile produced these signals. Worth a line of its own:
            # two runs with identical metrics above can be different strategies, and
            # the only way to tell them apart after the fact is this label.
            #
            # Suppressed entirely when obi_delta is missing, because that means
            # signals_all fell back to live_signals_py.csv -- a market-data file with
            # no signals in it. Calling those rows "unlabelled" would imply they came
            # from an older engine build, when in fact no engine wrote them.
            #
            # Branching on w_present rather than on the profile name: the name is a
            # free-text field from the CSV, and any producer may extend it (the demo
            # feed appends "-demo"). Whether a weighted reading actually exists is a
            # property of the data, so read it from the data.
            w_present, w_vals, w_profile = weighted_obi_state(signals_chart)
            w_count = int(w_vals.notna().sum()) if w_vals is not None else 0
            if "obi_delta" not in signals_chart.columns:
                note = None
            elif not w_profile:
                note = ("Weight profile unlabelled &mdash; these rows predate the "
                        "obi_profile column")
            elif w_present:
                note = (f"Weight profile <b style='color:#eab308;'>{w_profile}</b> "
                        f"&mdash; per-level weights applied to "
                        f"{w_count:,} of {len(signals_chart):,} signals")
            else:
                note = (f"Weight profile <b>{w_profile}</b> &mdash; every book level "
                        f"counted equally, so no separate weighted delta is recorded")
            if note:
                st.markdown(
                    f"<div style='font-size:0.6rem;color:#6b7280;margin:-0.2rem 0 0.4rem 0;'>{note}</div>",
                    unsafe_allow_html=True)
            mt1, mt2, mt3 = st.tabs(["Cumulative", "OBI Scatter", "OBI Series"])
            with mt1:
                sw = signals_chart.tail(500).copy()
                sw["cum"] = range(1, len(sw) + 1)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sw["time"], y=sw["cum"], mode="lines", line=dict(color="#eab308", width=1)))
                fig.update_layout(template="plotly_dark", height=140, margin=dict(l=0, r=0, t=0, b=0), hovermode="x unified", paper_bgcolor="#0f1217", plot_bgcolor="#0f1217")
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=True, gridcolor="#1e2329", griddash="dot")
                st.plotly_chart(fig, width='stretch')
                if len(sw) >= 10:
                    sw["bin"] = sw["time"].dt.floor("10s")
                    hist = sw.groupby("bin").size().reset_index(name="count")
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=hist["bin"], y=hist["count"], marker_color="#eab308", marker_line_width=0))
                    fig.update_layout(template="plotly_dark", height=80, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#0f1217", plot_bgcolor="#0f1217")
                    fig.update_xaxes(showgrid=False)
                    fig.update_yaxes(showgrid=True, gridcolor="#1e2329", griddash="dot")
                    st.plotly_chart(fig, width='stretch')
            with mt2:
                d = signals_chart.tail(500)
                # signals_all falls back to live_signals_py.csv when the C++ engine's
                # live_signals.csv is absent, and that file has no obi_delta/p_execute
                # columns. Reading them unguarded here crashed the whole page whenever
                # the Python ingester ran without the C++ engine.
                if "obi_delta" not in d.columns:
                    st.markdown("<div style='font-size:0.65rem;color:#6b7280;'>No OBI signals -- start the C++ engine to populate /tmp/live_signals.csv.</div>", unsafe_allow_html=True)
                else:
                    colors = ["#eab308" if a == 0 else "#3b82f6" for a in d["action"]] if "action" in d.columns else ["#eab308"] * len(d)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=d["time"], y=d["obi_delta"], mode="markers", name="Flat", marker=dict(color=colors, size=2, opacity=0.3)))

                    # Weighted delta overlaid on the same axes, not on a second
                    # chart: the question these two answer together is "how far
                    # apart are they", and that is only readable when they share a
                    # scale. Drawn as a line over the scatter so the flat cloud
                    # stays visible underneath. Absent (all-NaN) under the default
                    # flat profile, in which case no trace is added at all.
                    d_w, d_w_vals, _ = weighted_obi_state(d)
                    if d_w:
                        fig.add_trace(go.Scatter(
                            x=d["time"], y=d_w_vals, mode="lines", name="Weighted",
                            line=dict(color="#a855f7", width=0.9),
                            connectgaps=False))
                        fig.update_layout(showlegend=True, legend=dict(
                            orientation="h", y=1.02, x=1, xanchor="right",
                            font=dict(size=7, color="#6b7280")))

                    fig.add_hline(y=0, line_dash="dot", line_color="#2b3139")
                    fig.update_layout(template="plotly_dark", height=180, margin=dict(l=0, r=0, t=0, b=0), hovermode="x unified", paper_bgcolor="#0f1217", plot_bgcolor="#0f1217")
                    fig.update_xaxes(showgrid=False)
                    fig.update_yaxes(showgrid=True, gridcolor="#1e2329", griddash="dot")
                    st.plotly_chart(fig, width='stretch')
                    cc1, cc2, cc3, cc4 = st.columns(4)
                    cc1.metric("Avg|OBI|", f"{abs(d['obi_delta']).mean():.4f}")
                    cc2.metric("Max|OBI|", f"{abs(d['obi_delta']).max():.4f}")
                    cc3.metric("p(exec)", f"{d['p_execute'].iloc[-1]:.1%}" if "p_execute" in d.columns and len(d) else "--")
                    # Mean absolute gap between the two readings on rows where both
                    # exist. Near zero means the weights changed nothing on this
                    # book -- which happens whenever depth is roughly proportional
                    # across levels, and is a real finding rather than a bug.
                    if d_w:
                        gap = (d_w_vals - d["obi_delta"]).abs().mean()
                        cc4.metric("Weighted gap", f"{gap:.4f}")
                    else:
                        cc4.metric("Weighted gap", "--")
            with mt3:
                if prices_chart is not None and len(prices_chart) > 0 and "bid_vol" in prices_chart.columns:
                    pv = prices_chart.tail(2000).copy()
                    pv["obi_val"] = np.where((pv["bid_vol"] + pv["ask_vol"]) > 0, (pv["bid_vol"] - pv["ask_vol"]) / (pv["bid_vol"] + pv["ask_vol"]), np.nan)
                    fig = go.Figure()
                    for label, key, clr in [("Binance", "binance", "#eab308"), ("Kraken", "kraken", "#3b82f6")]:
                        d = pv[pv["exchange"] == key].dropna(subset=["obi_val"]).tail(200)
                        if len(d) > 1:
                            fig.add_trace(go.Scatter(x=d["time"], y=d["obi_val"], mode="lines", name=label, line=dict(color=clr, width=0.8)))
                    fig.add_hline(y=0, line_dash="dot", line_color="#2b3139")
                    fig.update_layout(template="plotly_dark", height=180, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.02, x=1, xanchor="right", font=dict(size=7, color="#6b7280")), hovermode="x unified", paper_bgcolor="#0f1217", plot_bgcolor="#0f1217")
                    fig.update_xaxes(showgrid=False)
                    fig.update_yaxes(showgrid=True, gridcolor="#1e2329", griddash="dot")
                    st.plotly_chart(fig, width='stretch')
        elif is_crypto:
            st.markdown("<div style='font-size:0.65rem;color:#6b7280;'>No signals yet.</div>", unsafe_allow_html=True)
        elif feed_df is not None and len(feed_df) > 5:
            fd = feed_df.tail(1000).copy()
            fd["time"] = pd.to_datetime(fd["timestamp_ms"], unit="ms", utc=True)
            ms1, ms2, ms3, ms4 = st.columns(4)
            ms1.metric("Records", f"{len(feed_df):,}")
            ms2.metric("Avg Spread", f"{(fd['ask_price'] - fd['bid_price']).div(fd['bid_price']).replace([np.inf, -np.inf], np.nan).dropna().mean() * 10000:.2f} bps")
            ms3.metric("Avg Mid", f"${((fd['bid_price'] + fd['ask_price']) / 2).mean():.2f}")
            ms4.metric("Last Qty", f"{float(feed_df.iloc[-1]['ask_qty']):.2f}")
            mt1, mt2 = st.tabs(["Mid Price", "Spread"])
            with mt1:
                fd["mid_price"] = (fd["bid_price"] + fd["ask_price"]) / 2
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fd["time"], y=fd["mid_price"], mode="lines", line=dict(color="#3b82f6", width=1), fill="tozeroy", fillcolor="rgba(59,130,246,0.04)"))
                fig.update_layout(template="plotly_dark", height=180, margin=dict(l=0, r=0, t=0, b=0), hovermode="x unified", paper_bgcolor="#0f1217", plot_bgcolor="#0f1217")
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=True, gridcolor="#1e2329", griddash="dot")
                st.plotly_chart(fig, width='stretch')
            with mt2:
                fd["spread_bps"] = (fd["ask_price"] - fd["bid_price"]) / fd["bid_price"] * 10000
                fd["spread_bps"] = fd["spread_bps"].replace([np.inf, -np.inf], np.nan)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fd["time"], y=fd["spread_bps"], mode="lines", line=dict(color="#eab308", width=0.6)))
                fig.add_hline(y=fd["spread_bps"].mean(), line_dash="dot", line_color="#2b3139")
                fig.update_layout(template="plotly_dark", height=140, margin=dict(l=0, r=0, t=0, b=0), hovermode="x unified", paper_bgcolor="#0f1217", plot_bgcolor="#0f1217")
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=True, gridcolor="#1e2329", griddash="dot")
                st.plotly_chart(fig, width='stretch')
        else:
            st.markdown("<div style='font-size:0.65rem;color:#6b7280;'>Collecting equity feed data.</div>", unsafe_allow_html=True)

    with tab_raw:
        r1, r2, r3 = st.tabs(["Signals", "Feed", "Depth"])
        with r1:
            if signals_chart is not None and len(signals_chart) > 0:
                st.dataframe(signals_chart.tail(100).iloc[::-1].reset_index(drop=True), width='stretch', hide_index=True)
            else:
                st.caption("No data.")
        with r2:
            if feed_df is not None and len(feed_df) > 0:
                st.dataframe(feed_df.tail(100).iloc[::-1].reset_index(drop=True), width='stretch', hide_index=True)
            else:
                st.caption("No data.")
        with r3:
            depth_raw = read_csv(DEPTH_PATH)
            if depth_raw is not None and len(depth_raw) > 0:
                st.dataframe(depth_raw.tail(100).iloc[::-1].reset_index(drop=True), width='stretch', hide_index=True)
            else:
                st.caption("No data.")

    st.caption(f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

_live_dashboard()
