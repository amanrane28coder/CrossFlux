#!/usr/bin/env bash
#
# run_live_dashboard.sh
# =====================
# Serve dashboard/app.py on a temporary public URL.
#
# What it does, in order:
#   1. Seeds /tmp with a synthetic feed so the page is never an empty shell.
#   2. Starts Streamlit bound to 127.0.0.1 (never 0.0.0.0 -- see SECURITY).
#   3. Opens a cloudflared quick tunnel and prints the public URL.
#   4. Tries to start the real producer; if it comes up, the demo feed is
#      stopped and the page upgrades itself to live data in place.
#
# Everything it starts, it stops. Ctrl-C is the intended way to end the session.
#
# SECURITY
# --------
# A quick tunnel is public and unauthenticated. Anyone with the URL reaches the
# dashboard, and the dashboard renders whatever is in /tmp -- including a real
# position and PnL if the live engine is running. Streamlit is bound to loopback
# so the tunnel is the *only* way in and closing it closes the door; binding
# 0.0.0.0 would additionally expose the app to everyone on your LAN, which
# outlives the tunnel. Treat the URL as a password and stop the script when done.
#
# Usage:
#   ./run_live_dashboard.sh              # demo feed, try to upgrade to live
#   ./run_live_dashboard.sh --demo-only  # never start the real producer
#   ./run_live_dashboard.sh --no-tunnel  # localhost only, no public URL
#   ./run_live_dashboard.sh --port 8600  # use a different local port

set -uo pipefail

PORT=8501
WANT_TUNNEL=1
DEMO_ONLY=0
TUNNEL_TIMEOUT=45

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)      PORT="${2:?--port needs a value}"; shift 2 ;;
        --no-tunnel) WANT_TUNNEL=0; shift ;;
        --demo-only) DEMO_ONLY=1; shift ;;
        -h|--help)   sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)           echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
    esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO" || exit 1

RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/crossflux-live.XXXXXX")"
STREAMLIT_LOG="$RUN_DIR/streamlit.log"
TUNNEL_LOG="$RUN_DIR/tunnel.log"
PRODUCER_LOG="$RUN_DIR/producer.log"

STREAMLIT_PID=""
TUNNEL_PID=""
DEMO_PID=""
PRODUCER_PID=""
# Declared before the trap is installed: `set -u` plus a cleanup path that reads
# PUBLIC_URL means an early failure would otherwise abort cleanup itself, which
# is exactly when cleanup matters most.
PUBLIC_URL=""

PY="${PYTHON:-python3}"

log()  { printf '\033[2m[run_live_dashboard]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[run_live_dashboard]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m[run_live_dashboard]\033[0m %s\n' "$*" >&2; exit 1; }

# --- Cleanup ---------------------------------------------------------------
# Runs on every exit path, including Ctrl-C and `die`. An orphaned tunnel is the
# failure that matters here: it would keep serving the dashboard after the
# terminal is closed, with nothing on screen to say so.
cleanup() {
    trap - EXIT INT TERM
    echo
    log "shutting down..."
    for pid_var in TUNNEL_PID PRODUCER_PID DEMO_PID STREAMLIT_PID; do
        pid="${!pid_var}"
        [[ -n "$pid" ]] || continue
        kill "$pid" 2>/dev/null || true
    done
    # Give them a moment to exit on their own before insisting.
    sleep 1
    for pid_var in TUNNEL_PID PRODUCER_PID DEMO_PID STREAMLIT_PID; do
        pid="${!pid_var}"
        [[ -n "$pid" ]] || continue
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    done

    # Don't leave an empty directory behind for a run that never logged anything
    # (a preflight failure), and don't announce logs that do not exist.
    if [[ -d "$RUN_DIR" ]] && [[ -z "$(ls -A "$RUN_DIR" 2>/dev/null)" ]]; then
        rmdir "$RUN_DIR" 2>/dev/null || true
    else
        log "logs kept in $RUN_DIR"
    fi

    # Only claim to have closed something that was actually opened. Saying "the
    # public URL is dead" after a preflight failure would imply a URL had been
    # published, which is the one thing a reader most needs to be accurate.
    if [[ -n "$PUBLIC_URL" ]]; then
        log "stopped. The public URL is dead."
    else
        log "stopped."
    fi
}
trap cleanup EXIT INT TERM

# --- Preflight -------------------------------------------------------------
command -v "$PY" >/dev/null 2>&1 || die "$PY not found"
"$PY" -c "import streamlit" 2>/dev/null || die \
    "streamlit is not installed. Run: pip install -r requirements.txt"

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    die "port $PORT is already in use. Stop that process or pass --port"
fi

if [[ $WANT_TUNNEL -eq 1 ]] && ! command -v cloudflared >/dev/null 2>&1; then
    warn "cloudflared not found -- serving on localhost only."
    warn "  macOS:  brew install cloudflared"
    warn "  Linux:  https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
    WANT_TUNNEL=0
fi

# --- 1. Seed the demo feed -------------------------------------------------
# app.py reads seven CSVs in /tmp. With nothing writing them every panel reads
# "--", so a freshly shared link looks broken rather than idle. The seeded rows
# are labelled status=demo and profile "<name>-demo", and the dashboard prints a
# banner saying the numbers are synthetic -- so this can never be mistaken for a
# real capture, by a viewer or by a later reader of the CSVs.
log "seeding demo feed into /tmp ..."
"$PY" dashboard/seed_demo_feed.py --backfill-seconds 180 \
    || warn "demo backfill failed; the page may start empty"

# Keep writing: app.py calls a feed stale after 10s, so a one-shot backfill
# renders as a dead engine to anyone who opens the link a minute later.
"$PY" dashboard/seed_demo_feed.py --follow >"$RUN_DIR/demo.log" 2>&1 &
DEMO_PID=$!
log "demo feed following (pid $DEMO_PID)"

# --- 2. Streamlit ----------------------------------------------------------
log "starting streamlit on 127.0.0.1:$PORT ..."
"$PY" -m streamlit run dashboard/app.py \
    --server.port "$PORT" \
    --server.address 127.0.0.1 \
    --server.headless true \
    --browser.gatherUsageStats false \
    >"$STREAMLIT_LOG" 2>&1 &
STREAMLIT_PID=$!

for _ in $(seq 1 40); do
    kill -0 "$STREAMLIT_PID" 2>/dev/null || break
    if curl -sf -o /dev/null "http://127.0.0.1:$PORT/_stcore/health" 2>/dev/null; then
        break
    fi
    sleep 0.5
done

if ! kill -0 "$STREAMLIT_PID" 2>/dev/null; then
    warn "streamlit exited during startup. Last lines:"
    tail -20 "$STREAMLIT_LOG" >&2
    die "could not start the dashboard"
fi
log "dashboard up at http://localhost:$PORT"

# --- 3. Tunnel -------------------------------------------------------------
if [[ $WANT_TUNNEL -eq 1 ]]; then
    log "opening cloudflared quick tunnel ..."
    cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" \
        >"$TUNNEL_LOG" 2>&1 &
    TUNNEL_PID=$!

    for _ in $(seq 1 $((TUNNEL_TIMEOUT * 2))); do
        kill -0 "$TUNNEL_PID" 2>/dev/null || break
        PUBLIC_URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' \
                      "$TUNNEL_LOG" 2>/dev/null | head -1)"
        [[ -n "$PUBLIC_URL" ]] && break
        sleep 0.5
    done

    if [[ -z "$PUBLIC_URL" ]]; then
        # Distinguish "died" from "timed out": they have different fixes (bad
        # install or blocked network vs. a slow handshake worth retrying), and a
        # message that says "after 45s" about a process that exited in under a
        # second sends the reader looking in the wrong place.
        if kill -0 "$TUNNEL_PID" 2>/dev/null; then
            warn "no tunnel URL after ${TUNNEL_TIMEOUT}s; cloudflared is still running. Last lines:"
        else
            warn "cloudflared exited before publishing a URL. Last lines:"
        fi
        tail -15 "$TUNNEL_LOG" >&2
        warn "continuing on localhost only."
        [[ -n "$TUNNEL_PID" ]] && kill "$TUNNEL_PID" 2>/dev/null
        TUNNEL_PID=""
    fi
fi

echo
echo "  ┌──────────────────────────────────────────────────────────────┐"
if [[ -n "$PUBLIC_URL" ]]; then
    printf '  │  PUBLIC URL   %-47s│\n' "$PUBLIC_URL"
    printf '  │  LOCAL        %-47s│\n' "http://localhost:$PORT"
    echo "  ├──────────────────────────────────────────────────────────────┤"
    echo "  │  This URL is public and has no password. Anyone who has it   │"
    echo "  │  sees the dashboard, including live position and PnL if the  │"
    echo "  │  real engine is running. Ctrl-C closes it.                   │"
else
    printf '  │  LOCAL        %-47s│\n' "http://localhost:$PORT"
    echo "  │  No public URL -- tunnel disabled or unavailable.            │"
fi
echo "  └──────────────────────────────────────────────────────────────┘"
echo

# --- 4. Upgrade to live data if the real producer will start ---------------
# Best-effort: the C++ engine needs Boost/OpenSSL and a built binary, and the
# Python producer needs websockets plus exchange reachability. Neither is a
# precondition for sharing the page -- the demo feed already covers it -- so a
# failure here is reported and shrugged off rather than fatal.
start_live_producer() {
    [[ $DEMO_ONLY -eq 1 ]] && return 1

    local engine=""
    for candidate in \
        "$REPO/live_trading_binance_kraken" \
        "$REPO/cpp_engine/build/live_trading" \
        "$REPO/live_trading/build/live_trading"
    do
        [[ -x "$candidate" && -f "$candidate" ]] && { engine="$candidate"; break; }
    done

    if [[ -n "$engine" ]]; then
        log "found engine: ${engine#"$REPO"/} -- starting"
        "$engine" >"$PRODUCER_LOG" 2>&1 &
        PRODUCER_PID=$!
    elif "$PY" -c "import websockets" 2>/dev/null; then
        log "no engine binary; starting the Python producer instead"
        "$PY" -m src.live_ingestion >"$PRODUCER_LOG" 2>&1 &
        PRODUCER_PID=$!
    else
        log "no live producer available (no engine binary, no websockets module)"
        return 1
    fi

    # A live producer proves itself by writing status=running. Anything less --
    # a missing shared library, a refused connection -- and we keep the demo.
    for _ in $(seq 1 20); do
        kill -0 "$PRODUCER_PID" 2>/dev/null || break
        if [[ -f /tmp/live_status.csv ]] \
           && grep -q ',running,' /tmp/live_status.csv 2>/dev/null; then
            return 0
        fi
        sleep 0.5
    done

    if ! kill -0 "$PRODUCER_PID" 2>/dev/null; then
        warn "live producer exited immediately. Last lines:"
        tail -8 "$PRODUCER_LOG" >&2
    else
        warn "live producer started but wrote no 'running' status in 10s"
        kill "$PRODUCER_PID" 2>/dev/null
    fi
    PRODUCER_PID=""
    return 1
}

if start_live_producer; then
    # Stop the demo only now that real rows are arriving. Killing it first would
    # blank the page for however long the engine takes to connect.
    log "live data confirmed -- stopping the demo feed"
    [[ -n "$DEMO_PID" ]] && kill "$DEMO_PID" 2>/dev/null
    DEMO_PID=""
    log "the page is now showing real engine output"
else
    log "staying on the demo feed (page is labelled DEMO)"
fi

echo
log "Ctrl-C to stop everything."

# Poll rather than `wait "$STREAMLIT_PID"`.
#
# Bash defers a trap until the current foreground command returns, and `wait` on
# a child that did not receive the signal never returns -- so `kill -INT <script>`
# would hang with the trap pending and leave the tunnel serving. Ctrl-C in a
# terminal happens to work, because that signals the whole foreground process
# group and the children die on their own, but anything that signals only this
# script -- a supervisor, systemd, another shell -- would leak a public URL.
# Sleeping in one-second slices caps trap latency at ~1s on every path.
#
# The loop also exits if Streamlit dies on its own, since the URL is then a 502.
while kill -0 "$STREAMLIT_PID" 2>/dev/null; do
    sleep 1
done

warn "streamlit exited (last lines):"
tail -5 "$STREAMLIT_LOG" >&2
