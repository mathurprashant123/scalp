"""
WEB DASHBOARD for the trend-scalp algo — replaces the raw Command Prompt
window with a browser-based page (like a website) showing live status,
with Start/Stop buttons.

Run this INSTEAD of trend_scalp_live.py directly:
    pip install flask
    python app.py

Then open your browser to: http://localhost:5000
Leave that browser tab open — as long as this Python process keeps
running (this Command Prompt window stays open in the background),
the algo keeps running, and the webpage auto-refreshes to show progress.

NOTE: This still runs on YOUR computer — closing this Command Prompt
window (or shutting down the PC) will stop it, same as before. For
TRUE 24/7 operation independent of your PC being on, this needs to run
on a cloud server (see the earlier cloud-server setup instructions) —
this web dashboard just makes it much easier to watch/control locally.
"""
import sys
import os
import threading
import queue
import time
import requests
from datetime import datetime

from flask import Flask, render_template_string, jsonify, request, Response

import csv as csv_module
import trend_scalp_live as algo

app = Flask(__name__)

# ============================================================
# PASSWORD PROTECTION (HTTP Basic Auth)
# ============================================================
# The Render URL is public on the internet by default — anyone who has
# (or guesses) the URL can open the dashboard, click Start/Stop, reset
# the circuit breaker, or clear trade logs. This adds a simple username +
# password prompt (the browser's built-in login popup) in front of EVERY
# page/route in this app, so only someone who knows the credentials can
# use it.
#
# Credentials are read from environment variables (NOT hardcoded here,
# same pattern as the API keys in config.py) so this file stays safe to
# commit to a public or private GitHub repo either way:
#   - Locally: set them in your .env file
#   - On Render: set them under Settings -> Environment as
#     DASHBOARD_USERNAME and DASHBOARD_PASSWORD
DASHBOARD_USERNAME = os.getenv("DASHBOARD_USERNAME", "")
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "")

if not DASHBOARD_USERNAME or not DASHBOARD_PASSWORD:
    print("[WARN] DASHBOARD_USERNAME / DASHBOARD_PASSWORD are not set — "
          "the dashboard will run WITHOUT password protection. Set them "
          "under Render -> Settings -> Environment to secure it.")


@app.before_request
def require_login():
    # ---- BUG FIX: exempt /keepalive from password-protection ----
    # Without this, the self-ping thread's request would get a 401 (since
    # it sends no credentials) — a 401 still likely resets Render's idle
    # timer since the request IS received, but it would print a confusing
    # "failed" line in the logs every 10 minutes for what's actually
    # working fine. /keepalive reveals nothing sensitive (just {"ok":true})
    # so there's no security reason to require login for it.
    if request.path == "/keepalive":
        return None
    # If credentials aren't configured, skip auth entirely (so local dev
    # without a .env still works) — but this means it's WIDE OPEN, so
    # always set these in production (Render).
    if not DASHBOARD_USERNAME or not DASHBOARD_PASSWORD:
        return None

    auth = request.authorization
    if not auth or auth.username != DASHBOARD_USERNAME or auth.password != DASHBOARD_PASSWORD:
        return Response(
            "Login required to view this dashboard.",
            401,
            {"WWW-Authenticate": 'Basic realm="Trend-Scalp Dashboard"'},
        )
    return None

log_queue = queue.Queue()
log_buffer = []  # keeps full history for the page (capped)
MAX_LOG_LINES = 500

state_info = {"running": False, "stop_event": None, "thread": None}


def find_trades_csv_path(strategy="A"):
    """Checks the primary location first, then the fallback temp location
    (in case the algo switched to it due to a permission issue)."""
    if strategy == "B":
        log_file = algo.TRADES_LOG_B
    elif strategy == "C":
        log_file = algo.TRADES_LOG_C
    else:
        log_file = algo.TRADES_LOG
    if os.path.exists(log_file):
        return log_file
    fallback_path = os.path.join(algo.FALLBACK_DIR, os.path.basename(log_file))
    if os.path.exists(fallback_path):
        return fallback_path
    return None


def reconstruct_trades(strategy="A"):
    """
    Reads the raw trades CSV (which logs OPEN / PARTIAL_CLOSE / CLOSE as
    separate rows) and reconstructs them into one readable row per trade,
    Excel-style: symbol, side, entry, exit, reason, P&L%, time.
    """
    path = find_trades_csv_path(strategy)
    if path is None:
        return [], None

    rows = []
    with open(path, newline="") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            rows.append(row)

    trades = []
    current = None
    for row in rows:
        action = row.get("action")
        sym = row.get("symbol")
        if action == "OPEN":
            current = {
                "symbol": sym, "side": row.get("side"),
                "entry_price": row.get("entry_price"),
                "entry_time": row.get("time"),
                "size": row.get("size"),
                "partials": [],
                "exit_price": None, "exit_time": None,
                "reason": None, "pnl_pct": None, "net_pnl_pct": None, "status": "OPEN",
            }
            trades.append(current)
        elif action == "PARTIAL_CLOSE" and current is not None and current["symbol"] == sym:
            current["partials"].append({
                "time": row.get("time"), "exit_price": row.get("exit_price"),
                "size": row.get("size"),
            })
        elif action == "CLOSE" and current is not None and current["symbol"] == sym:
            current["exit_price"] = row.get("exit_price")
            current["exit_time"] = row.get("time")
            current["reason"] = row.get("reason")
            current["pnl_pct"] = row.get("approx_gross_pnl_pct")
            current["net_pnl_pct"] = row.get("approx_net_pnl_pct_after_fees")
            current["gross_pnl_amount"] = row.get("gross_pnl_amount")
            current["fees_amount"] = row.get("fees_amount")
            current["net_pnl_amount"] = row.get("net_pnl_amount")
            current["status"] = "CLOSED"
            current = None  # ready for next OPEN

    return trades, path


TRADES_PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Trades Log</title>
    <meta charset="utf-8">
    <style>
        body { background:#1e1e1e; color:#ddd; font-family: 'Segoe UI', sans-serif; margin:0; padding:20px; }
        h1, h2 { text-align:center; }
        a.back { color:#4A90D9; text-decoration:none; display:block; text-align:center; margin-bottom:15px; }
        table { border-collapse: collapse; width:100%; margin-bottom: 25px; background:#0d0d0d; }
        th, td { border:1px solid #333; padding:8px 10px; font-size:13px; text-align:center; }
        th { background:#1a3c6e; color:white; }
        tr.open { background:#2a2a00; }
        tr.win { background:#0d2e0d; }
        tr.loss { background:#2e0d0d; }
        .summary-box { background:#0d0d0d; border-radius:8px; padding:15px 25px; margin: 0 auto 25px auto;
                       max-width: 500px; text-align:center; font-size:16px; }
        .pos { color:#4CAF50; font-weight:bold; }
        .neg { color:#E85D5D; font-weight:bold; }
        .day-header { background:#1a3c6e; color:white; font-weight:bold; }
    </style>
</head>
<body>
    <a class="back" href="/">&larr; Back to Dashboard</a>
    <h1>Trades Log — Logic {{ strategy }}</h1>
    <div style="text-align:center; margin-bottom:15px;">
        <a href="/trades?strategy=A" style="padding:8px 18px; margin:0 5px; border-radius:6px; text-decoration:none;
           background:{{ '#1a3c6e' if strategy=='A' else '#333' }}; color:white; font-size:13px;">Logic A (200EMA+VWAP+CVD)</a>
        <a href="/trades?strategy=B" style="padding:8px 18px; margin:0 5px; border-radius:6px; text-decoration:none;
           background:{{ '#1a3c6e' if strategy=='B' else '#333' }}; color:white; font-size:13px;">Logic B (EMA5/13+RSI+ATR)</a>
        <a href="/trades?strategy=C" style="padding:8px 18px; margin:0 5px; border-radius:6px; text-decoration:none;
           background:{{ '#1a3c6e' if strategy=='C' else '#333' }}; color:white; font-size:13px;">Logic C (9/20EMA multi-TF)</a>
    </div>
    <div style="text-align:center; margin-bottom:15px;">
        <button onclick="clearLog()" style="background:#8a1f1f; color:white; border:none;
                padding:10px 20px; border-radius:6px; cursor:pointer; font-size:13px;">
            &#128465; Clear Log {{ strategy }} (delete old/corrupted data, start fresh)
        </button>
    </div>
    <script>
        function clearLog() {
            if (!confirm('Ye trade history file delete kar dega (primary + fallback dono) Logic {{ strategy }} ke liye. Sure?')) return;
            fetch('/clear_trades_log?strategy={{ strategy }}', {method:'POST'}).then(r => r.json()).then(data => {
                if (data.ok) { alert('Cleared! Deleted: ' + JSON.stringify(data.deleted)); location.reload(); }
                else { alert('Error: ' + data.error); }
            });
        }
    </script>

    <div class="summary-box">
        <div>Total Closed Trades: <b>{{ total_trades }}</b></div>
        <div>Wins: <span class="pos">{{ wins }}</span> &nbsp;|&nbsp; Losses: <span class="neg">{{ losses }}</span></div>
        <div>Cumulative Gross P&amp;L: <span class="{{ 'pos' if cum_pnl >= 0 else 'neg' }}">{{ '%.3f'|format(cum_pnl) }}%</span></div>
        <div>Cumulative Net P&amp;L (after est. fees): <span class="{{ 'pos' if cum_net_pnl >= 0 else 'neg' }}">{{ '%.3f'|format(cum_net_pnl) }}%</span></div>
    </div>

    <h2>Day-by-Day Summary</h2>
    <table>
        <tr><th>Date</th><th>Trades</th><th>Wins</th><th>Losses</th><th>Day P&amp;L (sum %)</th></tr>
        {% for day, s in day_summary.items() %}
        <tr>
            <td>{{ day }}</td><td>{{ s.count }}</td><td>{{ s.wins }}</td><td>{{ s.losses }}</td>
            <td class="{{ 'pos' if s.pnl >= 0 else 'neg' }}">{{ '%.3f'|format(s.pnl) }}%</td>
        </tr>
        {% endfor %}
    </table>

    <h2>Trade-by-Trade Detail</h2>
    <table>
        <tr>
            <th>Entry Date & Time</th><th>Symbol</th><th>Side</th><th>Lot Size</th>
            <th>Entry Price</th><th>Exit Price</th><th>Exit Date & Time</th><th>Reason</th>
            <th>Gross P&amp;L In Amount</th><th>Fees Deduction</th>
            <th>Nett Amount P&amp;L</th><th>Nett Amount P&amp;L%</th><th>Status</th>
        </tr>
        {% for t in trades|reverse %}
        <tr class="{{ 'open' if t.status=='OPEN' else ('win' if (t.pnl_pct and t.pnl_pct|float >= 0) else 'loss') }}">
            <td>{{ t.entry_time }}</td>
            <td>{{ t.symbol }}</td>
            <td>{{ t.side }}</td>
            <td>{{ t.size if t.size else '—' }}</td>
            <td>{{ t.entry_price }}</td>
            <td>{{ t.exit_price if t.exit_price else '—' }}</td>
            <td>{{ t.exit_time if t.exit_time else '—' }}</td>
            <td>{{ t.reason if t.reason else '—' }}</td>
            <td>{{ '$%.4f'|format(t.gross_pnl_amount|float) if t.gross_pnl_amount else '—' }}</td>
            <td>{{ '$%.4f'|format(t.fees_amount|float) if t.fees_amount else '—' }}</td>
            <td>{{ '$%.4f'|format(t.net_pnl_amount|float) if t.net_pnl_amount else '—' }}</td>
            <td>{{ '%.3f'|format(t.net_pnl_pct|float) + '%' if t.net_pnl_pct else '—' }}</td>
            <td>{{ t.status }}</td>
        </tr>
        {% endfor %}
    </table>
    {% if not trades %}
    <p style="text-align:center;">Koi trade abhi tak nahi hui hai.</p>
    {% endif %}
    <p style="text-align:center; color:#888; font-size:11px;">Source file: {{ csv_path }}</p>
</body>
</html>
"""


@app.route("/reset_circuit_breaker", methods=["POST"])
def reset_circuit_breaker():
    """Manually clears today's daily-loss circuit-breaker tracking (useful
    if it tripped incorrectly, or you want to give it a fresh start).
    Re-baselines start_of_day_balance to whatever the balance is right now.

    ---- FIXED 2026-08-10: was writing directly to state.json, which the
    running trading-loop's stale in-memory state would silently
    overwrite (same root-cause bug found and fixed in force_clear_
    position — see that endpoint's docstring for the full explanation).
    Now queues the update instead, so it reliably applies on the bot's
    very next loop even while it keeps running. ----"""
    fields = {"daily_breaker_tripped": False}
    try:
        fields["start_of_day_balance"] = algo.get_wallet_total_balance()
    except Exception:
        pass  # if balance fetch fails, still clear the tripped flag
    ok = algo.queue_state_update({"action": "set_fields", "fields": fields})
    msg = ("Circuit breaker reset queued — takes effect on the bot's very next loop."
           if ok else "Couldn't queue the reset (failed to write the update-file) — try again.")
    return jsonify({"ok": ok, "message": msg})


@app.route("/reset_min_balance_breaker", methods=["POST"])
def reset_min_balance_breaker():
    """Manually clears the minimum-balance floor breaker (the second,
    independent safety net that blocks new trades if the account ever
    falls below MIN_BALANCE_FLOOR_PCT of its all-time starting balance).
    This does NOT re-baseline the all-time starting balance — that number
    stays fixed on purpose, so resetting this only un-blocks new entries,
    it doesn't move the floor itself. Use this deliberately, only after
    understanding why it tripped.

    ---- FIXED 2026-08-10: see reset_circuit_breaker's docstring — same
    queue-based fix for the same underlying race-condition bug. ----"""
    ok = algo.queue_state_update({"action": "set_fields",
                                   "fields": {"min_balance_breaker_tripped": False}})
    msg = ("Minimum-balance floor breaker reset queued — takes effect on the bot's very next loop."
           if ok else "Couldn't queue the reset — try again.")
    return jsonify({"ok": ok, "message": msg})


@app.route("/reset_abnormal_fill_breaker", methods=["POST"])
def reset_abnormal_fill_breaker():
    """Manually clears the abnormal-fill breaker — this trips when a
    close order fills at a price wildly far from the expected/reference
    price (e.g. a thin/broken order book), which real accounting can't
    catch in advance. Reset only after checking the exchange's own order
    history for what actually happened.

    ---- FIXED 2026-08-10: see reset_circuit_breaker's docstring — same
    queue-based fix for the same underlying race-condition bug. ----"""
    ok = algo.queue_state_update({"action": "set_fields", "fields": {
        "abnormal_fill_breaker_tripped": False, "abnormal_fill_detail": None,
    }})
    msg = ("Abnormal-fill breaker reset queued — takes effect on the bot's very next loop."
           if ok else "Couldn't queue the reset — try again.")
    return jsonify({"ok": ok, "message": msg})


@app.route("/recalibrate_balance_floor", methods=["POST"])
def recalibrate_balance_floor():
    """
    Deliberately DIFFERENT from the reset-buttons above — those only clear
    a tripped flag, they never move the baseline. This one re-baselines
    BOTH the daily-loss-breaker's start_of_day_balance AND the min-balance-
    floor's all_time_starting_balance to whatever the account's REAL
    balance is right now.

    Meant for exactly one situation: a deposit (or withdrawal) just
    happened, so the old baseline (recorded before the deposit) no longer
    reflects a meaningful floor — e.g. depositing $25 into an account that
    had $0.13 leaves the floor frozen at 50% of $0.13, which will never
    realistically trip again and gives no real protection for the new
    money. Re-run this any time you deposit/withdraw funds so the safety
    nets stay meaningful. Requires a live balance fetch, so it can fail
    if the exchange API is unreachable at that moment — safe to just
    retry in that case, nothing is changed until it succeeds.
    ---- FIXED 2026-08-10: see reset_circuit_breaker's docstring — same
    queue-based fix for the same underlying race-condition bug. ----
    """
    try:
        current_balance = algo.get_wallet_total_balance()
    except Exception as e:
        return jsonify({"ok": False, "message": f"Couldn't fetch current balance to "
                        f"recalibrate against ({e}) — try again in a moment."}), 503

    state = algo.load_state()  # read-only here, just for the old-value display in the message
    old_all_time = state.get("all_time_starting_balance")

    fields = {
        "all_time_starting_balance": current_balance,
        "start_of_day_balance": current_balance,
        "min_balance_breaker_tripped": False,
        "daily_breaker_tripped": False,
    }
    ok = algo.queue_state_update({"action": "set_fields", "fields": fields})

    new_floor = current_balance * (algo.MIN_BALANCE_FLOOR_PCT / 100)
    if not ok:
        return jsonify({"ok": False, "message": "Couldn't queue the recalibration — try again."})
    return jsonify({
        "ok": True,
        "message": (
            f"Recalibration queued (baseline was ${old_all_time:.2f}) -> "
            f"will apply ${current_balance:.2f} (new floor: ${new_floor:.2f}) "
            f"on the bot's very next loop. Daily-loss baseline also reset to today's actual balance."
        ) if old_all_time is not None else
        f"Baseline queued for the first time: ${current_balance:.2f} (floor: ${new_floor:.2f})."
    })


@app.route("/force_clear_position", methods=["POST"])
def force_clear_position():
    """EMERGENCY escape hatch. Clears this script's LOCAL tracking of an
    open position — it does NOT touch anything on the exchange itself.

    Use this only when a position is genuinely stuck (e.g. the exchange
    itself is rejecting every close attempt with something like
    'out_of_bankruptcy', usually caused by corrupted/glitched testnet
    price data on that specific position) and you've decided to deal
    with the real position on the exchange separately (manually, or via
    exchange support), rather than have it keep blocking this script from
    taking any new trades (since only one position is ever tracked at a
    time by design).

    After using this, the bot will believe it's flat and resume scanning
    for new entries. If the real position on the exchange is still open,
    the next restart's reconciliation will find it again and re-adopt it
    — so this is meant as a temporary unblock, not a permanent fix for
    whatever is stuck on the exchange side.

    ---- FIXED 2026-08-10: this used to ONLY write state["position"] =
    None into state.json. That looked correct, but the main trading
    LOOP is a separate running process holding its own in-memory copy of
    `state`, only reading the file fresh at startup — its next
    save_state(state) call would overwrite the file with its own STALE
    in-memory position, silently undoing this button's effect. In
    practice, the button only actually worked if you Stopped and
    Started the bot afterward, which wasn't obvious and looked like the
    button was simply broken.
    On closer review the SAME bug existed in 4 more buttons (the breaker
    -resets and recalibrate), so this was generalized into one shared
    queue_state_update() mechanism (see trend_scalp_live.py) that all of
    them now use — the loop applies + clears the whole queue at the
    start of every iteration, making all these buttons take effect
    immediately, even mid-run. ----"""
    ok = algo.queue_state_update({"action": "clear_position"})
    # Best-effort immediate state.json write too (harmless, and covers the
    # rare case where the bot isn't running at all right now, so there's no
    # loop to pick up the queue — a later restart would need it in state.json).
    state = algo.load_state()
    had_position = state.get("position") is not None
    state["position"] = None
    algo.save_state(state)
    if had_position:
        msg = ("Local position tracking cleared — takes effect on the bot's very "
               "next loop, even while it keeps running (no Stop/Start needed anymore). "
               "Remember: this did NOT close anything on the exchange itself; if a "
               "real position is still open there, handle it separately.") if ok else (
               "Cleared in state.json, but couldn't queue the live-update — if the bot "
               "is currently running, you may need to Stop then Start it.")
    else:
        msg = "No position was being tracked locally — nothing to clear."
    return jsonify({"ok": True, "message": msg})


@app.route("/clear_trades_log", methods=["POST"])
def clear_trades_log():
    """Deletes the trades log CSV (for the given strategy) from both
    possible locations (primary and fallback), so old/malformed historical
    rows don't keep corrupting the reconstructed table display."""
    from flask import request
    strategy = request.args.get("strategy", "A")
    deleted = []
    if strategy == "B":
        primary = algo.TRADES_LOG_B
    elif strategy == "C":
        primary = algo.TRADES_LOG_C
    else:
        primary = algo.TRADES_LOG
    fallback = os.path.join(algo.FALLBACK_DIR, os.path.basename(primary))
    for path in (primary, fallback):
        if os.path.exists(path):
            try:
                os.remove(path)
                deleted.append(path)
            except OSError as e:
                return jsonify({"ok": False, "error": str(e)})
    return jsonify({"ok": True, "deleted": deleted})


@app.route("/trades")
def trades_page():
    from flask import request
    strategy = request.args.get("strategy", "A")
    trades, path = reconstruct_trades(strategy)
    closed = [t for t in trades if t["status"] == "CLOSED"]
    wins = sum(1 for t in closed if t["pnl_pct"] and float(t["pnl_pct"]) >= 0)
    losses = len(closed) - wins
    cum_pnl = sum(float(t["pnl_pct"]) for t in closed if t["pnl_pct"])
    cum_net_pnl = sum(float(t["net_pnl_pct"]) for t in closed if t["net_pnl_pct"])

    day_summary = {}
    for t in closed:
        day = (t["exit_time"] or "")[:10]
        if day not in day_summary:
            day_summary[day] = {"count": 0, "wins": 0, "losses": 0, "pnl": 0.0}
        day_summary[day]["count"] += 1
        pnl_val = float(t["pnl_pct"]) if t["pnl_pct"] else 0.0
        day_summary[day]["pnl"] += pnl_val
        if pnl_val >= 0:
            day_summary[day]["wins"] += 1
        else:
            day_summary[day]["losses"] += 1

    return render_template_string(
        TRADES_PAGE_TEMPLATE, trades=trades, total_trades=len(closed),
        wins=wins, losses=losses, cum_pnl=cum_pnl, cum_net_pnl=cum_net_pnl, day_summary=day_summary,
        csv_path=path or "not found yet", strategy=strategy)


class WebLogRedirector:
    """Captures print() output from the algo into our log buffer."""
    def write(self, message):
        if message.strip():
            log_queue.put(message)

    def flush(self):
        pass


def drain_log_queue():
    while True:
        try:
            msg = log_queue.get_nowait()
            log_buffer.append(msg.rstrip("\n"))
            if len(log_buffer) > MAX_LOG_LINES:
                del log_buffer[0]
        except queue.Empty:
            break


PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Trend-Scalp Algo Dashboard</title>
    <meta charset="utf-8">
    <style>
        body { background:#1e1e1e; color:#ddd; font-family: 'Segoe UI', sans-serif; margin:0; padding:20px; }
        h1 { text-align:center; }
        #status { text-align:center; font-size:28px; font-weight:bold; padding:15px;
                   border-radius:8px; margin-bottom:20px; }
        .running { background:#2e7d32; color:white; }
        .stopped { background:#8a1f1f; color:white; }
        .buttons { text-align:center; margin-bottom:20px; }
        button { font-size:18px; padding:14px 30px; margin:0 10px; border:none;
                 border-radius:6px; cursor:pointer; font-weight:bold; color:white; }
        #startBtn { background:#2e7d32; }
        #stopBtn { background:#8a1f1f; }
        .modeBtn { background:#333; padding:8px 16px; font-size:13px; margin:0 4px; }
        .modeBtn.active { background:#2a5a8c; box-shadow: 0 0 6px #4A90D9; }
        button:disabled { opacity:0.4; cursor:not-allowed; }
        #log { background:#0d0d0d; color:#00ff66; font-family:Consolas,monospace;
               font-size:13px; padding:15px; height:520px; overflow-y:scroll;
               border-radius:6px; white-space:pre-wrap; }
        #pnlWidget { position:fixed; top:15px; right:15px; width:240px;
                     background:#0d0d0d; border:2px solid #444; border-radius:10px;
                     padding:14px; font-size:13px; box-shadow: 0 4px 12px rgba(0,0,0,0.5);
                     z-index: 1000; }
        #pnlWidget.flat { border-color:#555; }
        #pnlWidget.profit { border-color:#2e7d32; }
        #pnlWidget.loss { border-color:#8a1f1f; }
        #pnlWidget h3 { margin:0 0 8px 0; font-size:14px; color:#aaa; }
        #pnlValue { font-size:26px; font-weight:bold; }
        #pnlValue.pos { color:#4CAF50; }
        #pnlValue.neg { color:#E85D5D; }
        .pnl-row { display:flex; justify-content:space-between; margin-top:4px; color:#bbb; font-size:12px; }
        #regimeBanner { background:#8a1f1f; color:white; text-align:center; padding:12px 20px;
                         border-radius:8px; margin-bottom:18px; font-size:15px; font-weight:bold;
                         border: 2px solid #d94f4f; box-shadow: 0 0 10px rgba(217,79,79,0.5); }
        #regimeBanner .sub { font-size:12px; font-weight:normal; color:#f0c6c6; margin-top:4px; }
        #regimeBanner.hidden { display:none; }
        #oiWarmupBanner { color:white; text-align:center; padding:10px 20px;
                           border-radius:8px; margin:10px 0; font-size:14px; font-weight:bold; }
        #oiWarmupBanner.hidden { display:none; }
        #trendMeter { display:flex; gap:10px; margin-bottom:18px; }
        #trendMeter.hidden { display:none; }
        .trend-card { flex:1; text-align:center; padding:10px; border-radius:8px; font-size:13px; }
        .trend-card .sym { font-weight:bold; font-size:14px; margin-bottom:4px; }
        .trend-card .state { font-weight:bold; }
        .trend-uptrend-active { background:#1f4a2a; border:2px solid #4CAF50; }
        .trend-uptrend-forming { background:#1f3a2a; border:2px dashed #6ab577; }
        .trend-downtrend-active { background:#4a1f1f; border:2px solid #E85D5D; }
        .trend-downtrend-forming { background:#3a2222; border:2px dashed #c47a7a; }
        .trend-neutral { background:#333; border:2px solid #888; }
    </style>
</head>
<body>
    <div id="regimeBanner" class="hidden"></div>
    <div id="trendMeter" class="hidden"></div>
    <div id="pnlWidget" class="flat">
        <h3>Live Trade P&amp;L</h3>
        <div class="pnl-row" style="border-bottom:1px solid #333; padding-bottom:6px; margin-bottom:6px;">
            <span>Wallet Balance</span><span id="walletBalance"><b>—</b></span>
        </div>
        <div id="pnlBody">No open position</div>
    </div>
    <div style="text-align:center; color:#ff9933; font-size:22px; font-weight:bold; margin-bottom:8px; letter-spacing:1px;">
        🔱 जय महाकाल 🔱
    </div>
    <h1>Trend-Scalp Algo — Dashboard</h1>
    <div style="text-align:center; margin-bottom:15px;">
        <a href="/trades" style="color:#4A90D9; font-size:15px; text-decoration:none;">
            &#128202; View Trades Log (Excel-style) &rarr;
        </a>
    </div>
    <div style="text-align:center; margin-bottom:15px;">
        <span style="color:#aaa; font-size:13px; margin-right:10px;">Active Logic (tick any combination):</span>
        <label style="color:white; font-size:13px; margin:0 10px; cursor:pointer;">
            <input type="checkbox" id="logicChkA" onchange="toggleLogic('A')" style="margin-right:5px;">Logic A</label>
        <label style="color:white; font-size:13px; margin:0 10px; cursor:pointer;">
            <input type="checkbox" id="logicChkB" onchange="toggleLogic('B')" style="margin-right:5px;">Logic B</label>
        <label style="color:white; font-size:13px; margin:0 10px; cursor:pointer;">
            <input type="checkbox" id="logicChkC" onchange="toggleLogic('C')" style="margin-right:5px;">Logic C</label>
    </div>
    <div style="text-align:center; margin-bottom:15px;">
        <button onclick="setMode('A')" class="modeBtn" style="font-size:11px; padding:4px 10px;">A Only</button>
        <button onclick="setMode('B')" class="modeBtn" style="font-size:11px; padding:4px 10px;">B Only</button>
        <button onclick="setMode('C')" class="modeBtn" style="font-size:11px; padding:4px 10px;">C Only</button>
        <button onclick="setMode('BOTH')" class="modeBtn" style="font-size:11px; padding:4px 10px;">All (A+B+C)</button>
    </div>
    <div style="text-align:center; margin-bottom:15px;">

        <button onclick="resetBreaker()" style="background:#7a4a00; color:white; border:none;
                padding:8px 18px; border-radius:6px; cursor:pointer; font-size:13px;">
            &#9888; Reset Daily Circuit Breaker
        </button>
        <button onclick="resetMinBalanceBreaker()" style="background:#7a0000; color:white; border:none;
                padding:8px 18px; border-radius:6px; cursor:pointer; font-size:13px; margin-left:8px;">
            &#9888; Reset Min-Balance Floor Breaker
        </button>
        <button onclick="resetAbnormalFillBreaker()" style="background:#4a0033; color:white; border:none;
                padding:8px 18px; border-radius:6px; cursor:pointer; font-size:13px; margin-left:8px;">
            &#9888; Reset Abnormal-Fill Breaker
        </button>
        <button onclick="recalibrateBalanceFloor()" style="background:#0a4a2a; color:white; border:none;
                padding:8px 18px; border-radius:6px; cursor:pointer; font-size:13px; margin-left:8px;">
            &#128176; Recalibrate Balance Floor (after deposit/withdrawal)
        </button>
        <button onclick="forceClearPosition()" style="background:#333333; color:white; border:2px solid #ff4444;
                padding:8px 18px; border-radius:6px; cursor:pointer; font-size:13px; margin-left:8px;">
            &#9940; Force-Clear Stuck Position (emergency)
        </button>
        <button onclick="startOiWarmup()" style="background:#1a3a5a; color:white; border:2px solid #4a9ade;
                padding:8px 18px; border-radius:6px; cursor:pointer; font-size:13px; margin-left:8px;">
            &#9203; Start OI Warm-Up (before starting bot)
        </button>
    </div>
    <div id="oiWarmupBanner" class="hidden"></div>
    <script>
        function resetBreaker() {
            if (!confirm('Ye aaj ka daily-loss circuit breaker reset kar dega. Sure?')) return;
            fetch('/reset_circuit_breaker', {method:'POST'}).then(r => r.json()).then(data => {
                alert(data.message || 'Done');
            });
        }
        function resetMinBalanceBreaker() {
            if (!confirm('Ye account ka minimum-balance safety floor reset kar dega — balance kaafi gir chuka hai, isliye pehle wajah samajh lo phir hi reset karo. Sure?')) return;
            fetch('/reset_min_balance_breaker', {method:'POST'}).then(r => r.json()).then(data => {
                alert(data.message || 'Done');
            });
        }
        function resetAbnormalFillBreaker() {
            if (!confirm('Ye tabhi reset karo jab tumne exchange ke Order History mein dekh liya ho ki abnormal fill kya thi. Sure?')) return;
            fetch('/reset_abnormal_fill_breaker', {method:'POST'}).then(r => r.json()).then(data => {
                alert(data.message || 'Done');
            });
        }
        function recalibrateBalanceFloor() {
            if (!confirm('Ye daily-loss aur min-balance-floor DONO ki baseline ko ABHI ke real balance par reset kar dega. Sirf tab use karo jab tumne abhi deposit/withdraw kiya ho. Sure?')) return;
            fetch('/recalibrate_balance_floor', {method:'POST'}).then(r => r.json()).then(data => {
                alert(data.message || 'Done');
            });
        }
        function forceClearPosition() {
            if (!confirm('EMERGENCY: Ye sirf LOCAL tracking clear karega, exchange par kuch close NAHI hoga. Sirf tab use karo jab position genuinely stuck ho (out_of_bankruptcy jaisa error) aur exchange par usse alag se, manually deal karna decide kar liya ho. Sure?')) return;
            fetch('/force_clear_position', {method:'POST'}).then(r => r.json()).then(data => {
                alert(data.message || 'Done');
            });
        }
        function startOiWarmup() {
            if (!confirm('Ye OI-history fresh se build karega (' + '10' + ' minute tak naye entries ruke rahenge). Bot start karne se PEHLE isse use karo, restart ke turant baad nahi. Sure?')) return;
            fetch('/start_oi_warmup', {method:'POST'}).then(r => r.json()).then(data => {
                alert(data.message || 'Done');
                refreshOiWarmup();
            });
        }
        function refreshOiWarmup() {
            fetch('/oi_warmup_status').then(r => r.json()).then(data => {
                const el = document.getElementById('oiWarmupBanner');
                if (!data || !data.active) {
                    el.className = 'hidden';
                    return;
                }
                el.className = '';
                if (data.ready) {
                    el.style.background = '#0a4a2a';
                    el.style.border = '2px solid #4CAF50';
                    el.innerHTML = '&#9989; OI Warm-Up complete &mdash; history ready, bot resuming normal scanning.';
                } else {
                    el.style.background = '#1a3a5a';
                    el.style.border = '2px solid #4a9ade';
                    el.innerHTML = `&#9203; OI Warm-Up in progress: ${data.elapsed_minutes} / ${data.target_minutes} minutes &mdash; new entries paused.`;
                }
            });
        }
        setInterval(refreshOiWarmup, 5000);
        refreshOiWarmup();
    </script>
    <div id="status" class="stopped">STOPPED</div>
    <div class="buttons">
        <button id="startBtn" onclick="startAlgo()">&#9654; START</button>
        <button id="stopBtn" onclick="stopAlgo()" disabled>&#9632; STOP</button>
    </div>
    <div id="log">Waiting to start...</div>

    <script>
        function startAlgo() {
            fetch('/start', {method:'POST'}).then(r => r.json()).then(data => {
                if (data.already_running && data.message) {
                    alert(data.message);
                }
                refreshStatus();
            });
        }
        function stopAlgo() {
            fetch('/stop', {method:'POST'}).then(refreshStatus);
        }
        function refreshStatus() {
            fetch('/status').then(r => r.json()).then(data => {
                const statusDiv = document.getElementById('status');
                const startBtn = document.getElementById('startBtn');
                const stopBtn = document.getElementById('stopBtn');
                if (data.running) {
                    statusDiv.textContent = 'RUNNING';
                    statusDiv.className = 'running';
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                } else {
                    statusDiv.textContent = 'STOPPED';
                    statusDiv.className = 'stopped';
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                }
            });
        }
        function refreshLogs() {
            fetch('/logs').then(r => r.json()).then(data => {
                const logDiv = document.getElementById('log');
                const wasAtBottom = logDiv.scrollTop + logDiv.clientHeight >= logDiv.scrollHeight - 50;
                logDiv.textContent = data.logs.join('\\n');
                if (wasAtBottom) logDiv.scrollTop = logDiv.scrollHeight;
            });
        }
        // Quick-preset buttons — set the FULL enabled-set in one click
        function setMode(mode) {
            let enabled;
            if (mode === 'BOTH') enabled = ['A', 'B', 'C'];
            else enabled = [mode];
            applyEnabled(enabled);
        }
        // Checkbox toggle — flips just ONE logic on/off, keeping the rest as-is
        function toggleLogic(logic) {
            fetch('/logic_mode').then(r => r.json()).then(data => {
                let enabled = new Set(data.enabled);
                if (enabled.has(logic)) enabled.delete(logic);
                else enabled.add(logic);
                if (enabled.size === 0) {
                    alert('At least one logic must stay enabled.');
                    refreshMode();  // revert the checkbox visually
                    return;
                }
                applyEnabled(Array.from(enabled));
            });
        }
        function applyEnabled(enabled) {
            fetch('/logic_mode', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({enabled: enabled})
            }).then(r => r.json()).then(data => {
                if (data.ok) refreshMode();
                else alert('Error: ' + data.error);
            });
        }
        function refreshMode() {
            fetch('/logic_mode').then(r => r.json()).then(data => {
                const enabled = new Set(data.enabled);
                document.getElementById('logicChkA').checked = enabled.has('A');
                document.getElementById('logicChkB').checked = enabled.has('B');
                document.getElementById('logicChkC').checked = enabled.has('C');
            });
        }
        function refreshPnl() {
            fetch('/live_pnl').then(r => r.json()).then(data => {
                const walletEl = document.getElementById('walletBalance');
                const bal = data.wallet_balance;
                walletEl.innerHTML = (bal !== null && bal !== undefined) ? `<b>$${bal.toFixed(2)}</b>` : '<b>—</b>';

                const widget = document.getElementById('pnlWidget');
                const body = document.getElementById('pnlBody');
                if (!data.has_position) {
                    widget.className = 'flat';
                    body.innerHTML = 'No open position';
                    return;
                }
                const pnl = data.live_pnl_pct;
                const pnlClass = (pnl >= 0) ? 'pos' : 'neg';
                widget.className = (pnl >= 0) ? 'profit' : 'loss';
                const pnlText = (pnl !== null && pnl !== undefined) ? pnl.toFixed(3) + '%' : '—';
                body.innerHTML = `
                    <div id="pnlValue" class="${pnlClass}">${pnlText}</div>
                    <div class="pnl-row"><span>Strategy</span><span><b>Logic ${data.strategy}</b></span></div>
                    <div class="pnl-row"><span>${data.symbol}</span><span>${data.side.toUpperCase()}</span></div>
                    <div class="pnl-row"><span>Lot Size</span><span>${data.size}</span></div>
                    <div class="pnl-row"><span>Entry</span><span>${data.entry_price}</span></div>
                    <div class="pnl-row"><span>Current</span><span>${data.current_price ?? '—'}</span></div>
                    <div class="pnl-row"><span>Stop</span><span>${data.stop}</span></div>
                    <div class="pnl-row"><span>Target</span><span>${data.target}</span></div>
                    <div class="pnl-row"><span>Stop-loss steps locked</span><span>${data.milestones_locked} / 4</span></div>
                `;
            });
        }
        function refreshRegime() {
            fetch('/market_regime').then(r => r.json()).then(data => {
                const banner = document.getElementById('regimeBanner');
                if (!data || !data.favors) {
                    banner.className = 'hidden';
                    return;
                }
                banner.className = '';
                let mainText;
                if (data.favors === 'A') {
                    mainText = `Market Condition: ${data.label} \u2014 LOGIC A environment right now`;
                } else if (data.favors === 'B') {
                    mainText = `Market Condition: ${data.label} \u2014 LOGIC B environment right now`;
                } else {
                    mainText = `Market Condition: ${data.label} \u2014 mixed, neither logic strongly favored`;
                }
                const perSymbol = Object.entries(data.per_symbol || {})
                    .map(([sym, d]) => `${sym} ${d}% from VWAP`).join(' | ');
                banner.innerHTML = `${mainText}<div class="sub">Avg VWAP-distance: ${data.avg_vwap_dist_pct}% ` +
                    `(Logic A needs \u2264${data.threshold_pct}%) &mdash; ${perSymbol}</div>`;
            });
        }
        function refreshTrendMeter() {
            fetch('/trend_meter').then(r => r.json()).then(data => {
                const el = document.getElementById('trendMeter');
                const syms = Object.keys(data || {});
                if (syms.length === 0) {
                    el.className = 'hidden';
                    return;
                }
                el.className = '';
                const stateClass = {
                    'UPTREND-ACTIVE': 'trend-uptrend-active',
                    'UPTREND-FORMING': 'trend-uptrend-forming',
                    'DOWNTREND-ACTIVE': 'trend-downtrend-active',
                    'DOWNTREND-FORMING': 'trend-downtrend-forming',
                    'NEUTRAL': 'trend-neutral',
                };
                el.innerHTML = syms.map(sym => {
                    const d = data[sym];
                    const cls = stateClass[d.state] || 'trend-neutral';
                    return `<div class="trend-card ${cls}"><div class="sym">${sym}</div>` +
                           `<div class="state">${d.state}</div></div>`;
                }).join('');
            });
        }
        setInterval(refreshTrendMeter, 5000);
        refreshTrendMeter();
        setInterval(refreshRegime, 5000);
        refreshRegime();
        setInterval(refreshStatus, 2000);
        setInterval(refreshLogs, 2000);
        setInterval(refreshPnl, 3000);
        setInterval(refreshMode, 3000);
        refreshStatus();
        refreshLogs();
        refreshPnl();
        refreshMode();
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE_TEMPLATE)


start_lock = threading.Lock()


@app.route("/start", methods=["POST"])
def start():
    with start_lock:
        # Check the ACTUAL thread state, not just a flag — a flag can go
        # stale (e.g. if the old thread is still finishing its current
        # loop iteration after Stop was clicked, but hasn't set running=False
        # yet). Starting a second thread while the old one is still alive
        # causes TWO overlapping instances trading on the same account —
        # exactly what caused the earlier duplicate-position incident.
        old_thread = state_info.get("thread")
        if old_thread is not None and old_thread.is_alive():
            return jsonify({
                "ok": False,
                "already_running": True,
                "message": "Previous run is still shutting down — please wait a few "
                           "seconds and try Start again (this prevents two instances "
                           "running at once).",
            })

        state_info["stop_event"] = threading.Event()
        sys.stdout = WebLogRedirector()

        def run():
            try:
                algo.main_loop(stop_event=state_info["stop_event"])
            except Exception as e:
                log_queue.put(f"[FATAL ERROR] {e}")
            finally:
                state_info["running"] = False

        state_info["thread"] = threading.Thread(target=run, daemon=True)
        state_info["running"] = True
        state_info["thread"].start()
        return jsonify({"ok": True})


@app.route("/stop", methods=["POST"])
def stop():
    if state_info["stop_event"] is not None:
        state_info["stop_event"].set()
    state_info["running"] = False
    return jsonify({"ok": True})


@app.route("/status")
def status():
    # Derive from the actual thread object, not just the flag, so the UI
    # never shows "stopped" while a thread is still genuinely alive (or
    # vice versa) — keeps /start's liveness check and the UI in sync.
    thread = state_info.get("thread")
    truly_running = thread is not None and thread.is_alive()
    return jsonify({"running": truly_running})


@app.route("/logs")
def logs():
    drain_log_queue()
    return jsonify({"logs": log_buffer})


@app.route("/logic_mode", methods=["GET"])
@app.route("/logic_mode", methods=["GET"])
def get_logic_mode():
    return jsonify({"enabled": sorted(algo.LOGIC_MODE["enabled"])})


@app.route("/logic_mode", methods=["POST"])
def set_logic_mode():
    from flask import request
    data = request.get_json(force=True)
    enabled = data.get("enabled")
    if not isinstance(enabled, list):
        return jsonify({"ok": False, "error": "enabled must be a list, e.g. [\"A\",\"C\"]"}), 400
    valid = {"A", "B", "C"} & set(enabled)
    if not valid:
        return jsonify({"ok": False, "error": "at least one of A, B, C must be enabled"}), 400
    algo.LOGIC_MODE["enabled"] = valid
    # ---- Persisted into the state file too, so a restart picks up the
    # last choice instead of silently reverting (same reasoning as the
    # old single-mode version — see git history for the original bug this
    # fixed). Stored as a sorted LIST since JSON has no native set type. ----
    state = algo.load_state()
    state["logic_mode_enabled"] = sorted(valid)
    algo.save_state(state)
    return jsonify({"ok": True, "enabled": sorted(valid)})


@app.route("/market_regime")
def market_regime():
    regime = algo.LATEST_STATE.get("market_regime")
    if regime is None:
        return jsonify({"favors": None})
    return jsonify(regime)


@app.route("/trend_meter")
def trend_meter():
    meter = algo.LATEST_STATE.get("trend_meter")
    if meter is None:
        return jsonify({})
    return jsonify(meter)


@app.route("/start_oi_warmup", methods=["POST"])
def start_oi_warmup_route():
    """Kicks off OI Warm-Up: clears OI history and pauses all new entries
    for OI_WARMUP_MINUTES real minutes while a dense, fresh history
    builds up. See start_oi_warmup() / OI_WARMUP_MINUTES docstrings in
    trend_scalp_live.py for why this is a genuine shorter real-time wait,
    not an instant backfill (the exchange only exposes CURRENT OI).

    ---- FIXED 2026-08-13: this used to ALWAYS queue a pending-update
    AND write state.json directly, regardless of whether the bot was
    running. If pressed while the bot was STOPPED, the direct write
    started the real warm-up timer immediately — but the queued update
    stayed sitting in the pending-updates file (unprocessed, since no
    loop was running to pick it up). The very next time the bot was
    started, its first loop would apply that stale queued update and
    call start_oi_warmup() AGAIN, silently resetting the already-ticking
    timer back to 0 — exactly what was observed: warm-up looked like it
    restarted right after pressing Start. Fix: only queue the live
    -update when the bot is actually running (checked via algo state);
    when it's stopped, the direct state.json write alone is correct and
    sufficient — there's no running loop to apply a queued action, so
    queuing one is not just unnecessary but actively harmful the next
    time the bot starts. ----"""
    thread = state_info.get("thread")
    bot_running = thread is not None and thread.is_alive()
    state = algo.load_state()
    algo.start_oi_warmup(state)
    algo.save_state(state)
    if bot_running:
        algo.queue_state_update({"action": "start_oi_warmup"})
        msg = (f"OI Warm-Up started — new entries paused for {algo.OI_WARMUP_MINUTES} "
               "minutes while fresh OI-history builds up.")
    else:
        msg = (f"OI Warm-Up started — new entries paused for {algo.OI_WARMUP_MINUTES} "
               "minutes while fresh OI-history builds up. Bot is currently stopped — "
               "press Start whenever you're ready, the warm-up timer is already ticking "
               "in the background and won't reset when you do.")
    return jsonify({"ok": True, "message": msg})


@app.route("/oi_warmup_status")
def oi_warmup_status_route():
    status = algo.LATEST_STATE.get("oi_warmup")
    if status is None:
        status = algo.get_oi_warmup_status(algo.load_state())
    return jsonify(status)


@app.route("/options_status")
def options_status_route():
    """---- NEW 2026-08-14: status for the Options-Bot running in the
    SAME process/service as this futures-bot dashboard (see the
    threading.Thread(target=options_bot.bot_loop...) call at the bottom
    of this file). Genuinely independent state from the futures bot —
    reads options_bot's own state file directly. ----"""
    try:
        import options_bot
        state = options_bot.load_state()
        return jsonify({
            "dry_run": options_bot.DRY_RUN,
            "in_golden_window": options_bot.in_golden_window(),
            "position": state.get("position"),
            "cooldown_until": state.get("cooldown_until"),
            "recent_trades": state.get("trade_history", [])[-10:],
        })
    except Exception as e:
        return jsonify({"error": f"Options-bot status genuinely unavailable: {e}"})


@app.route("/live_pnl")
def live_pnl():
    pos = algo.LATEST_STATE.get("position")
    wallet_balance = algo.LATEST_STATE.get("wallet_balance")

    if pos is None:
        return jsonify({"has_position": False, "wallet_balance": wallet_balance})

    return jsonify({
        "has_position": True,
        "symbol": pos.get("symbol"),
        "side": pos.get("side"),
        "size": pos.get("size"),
        "entry_price": pos.get("entry_price"),
        "current_price": algo.LATEST_STATE.get("current_price"),
        "stop": pos.get("stop"),
        "target": pos.get("target"),
        "live_pnl_pct": algo.LATEST_STATE.get("live_pnl_pct"),
        "milestones_locked": pos.get("milestones_locked", 0),
        "strategy": pos.get("strategy", "A"),
        "wallet_balance": wallet_balance,
    })


@app.route("/keepalive")
def keepalive():
    """
    Deliberately tiny and fast — this exists ONLY to be hit by our own
    self-ping thread below (see _self_ping_loop), so Render's free-tier
    web service keeps seeing regular inbound traffic and doesn't spin the
    instance down as idle. Returns almost nothing on purpose — no need to
    touch algo.LATEST_STATE or do any real work for a keepalive.
    """
    return jsonify({"ok": True})


def _self_ping_loop():
    """
    ---- NEW: bot's OWN internal keepalive, replacing reliance on an
    external service (cron-job.org) ----
    Render's free-tier web services spin down after a period with no
    inbound HTTP traffic, and spinning back up loses this process's
    in-memory state (LATEST_STATE, any position the main loop was
    tracking) — this was observed directly, multiple times, as
    "Adopting untracked real position" appearing in the logs right after
    an unplanned restart, on a position that was still genuinely open on
    the exchange the whole time.

    This runs forever in the background, sleeping most of the time, and
    just GETs our own /keepalive endpoint every 10 minutes — the same
    idea as the external cron-job.org pings, just self-contained so the
    bot doesn't depend on a separate third-party service staying
    configured/active. Every failure is caught and logged, never allowed
    to crash this thread or the main app.

    Does nothing (and prints why) if RENDER_EXTERNAL_URL isn't set — e.g.
    when running locally, where this isn't needed at all.
    """
    external_url = os.getenv("RENDER_EXTERNAL_URL")
    if not external_url:
        print("[keepalive] RENDER_EXTERNAL_URL not set (probably running "
              "locally) — self-ping thread will not run.")
        return

    ping_url = external_url.rstrip("/") + "/keepalive"
    print(f"[keepalive] Self-ping thread started — will GET {ping_url} every 10 minutes.")
    while True:
        time.sleep(600)  # 10 minutes
        try:
            resp = requests.get(ping_url, timeout=15)
            print(f"[keepalive] Self-ping OK ({resp.status_code}) at {algo.now_ist()}")
        except Exception as e:
            print(f"[keepalive] Self-ping failed ({e}) — will retry in 10 minutes. "
                  f"Not fatal on its own, but if this keeps failing the service "
                  f"may still get spun down as idle.")


if __name__ == "__main__":
    print("Starting web dashboard at http://localhost:5000 ...")
    threading.Thread(target=_self_ping_loop, daemon=True).start()

    # ---- NEW 2026-08-14: Options-Bot, genuinely running in the SAME
    # Render service/process as this futures-bot dashboard (user's
    # explicit choice — one Render service, not two). Completely
    # independent state (options_bot_state.json vs state.json) and
    # completely independent loop — see options_bot.py's own docstring
    # for the full strategy. Starts in DRY_RUN mode by default (see
    # OPTIONS_BOT_LIVE env var in options_bot.py) — genuinely no real
    # orders until that's explicitly flipped. Import is wrapped in
    # try/except so a problem in the (separately-tested) options-bot
    # module can NEVER take down the existing, proven futures-bot
    # dashboard — the two are independent enough in code that they
    # shouldn't interfere, but this is an extra safety margin. ----
    try:
        import options_bot
        threading.Thread(target=options_bot.bot_loop, daemon=True).start()
        print(f"Options-bot thread genuinely started (DRY_RUN={options_bot.DRY_RUN}). "
              f"Status at /options_status.")
    except Exception as e:
        print(f"[WARN] Options-bot genuinely failed to start ({e}) — "
              f"futures-bot dashboard continues normally regardless.")

    app.run(host="0.0.0.0", port=5000, debug=False)
