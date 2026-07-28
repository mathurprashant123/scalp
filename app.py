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
    log_file = algo.TRADES_LOG if strategy == "A" else algo.TRADES_LOG_B
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
            <th>Entry Time</th><th>Symbol</th><th>Side</th><th>Entry Price</th>
            <th>Exit Price</th><th>Exit Time</th><th>Reason</th><th>Gross P&amp;L %</th><th>Net P&amp;L % (est. fees)</th><th>Status</th>
        </tr>
        {% for t in trades|reverse %}
        <tr class="{{ 'open' if t.status=='OPEN' else ('win' if (t.pnl_pct and t.pnl_pct|float >= 0) else 'loss') }}">
            <td>{{ t.entry_time }}</td>
            <td>{{ t.symbol }}</td>
            <td>{{ t.side }}</td>
            <td>{{ t.entry_price }}</td>
            <td>{{ t.exit_price if t.exit_price else '—' }}</td>
            <td>{{ t.exit_time if t.exit_time else '—' }}</td>
            <td>{{ t.reason if t.reason else '—' }}</td>
            <td>{{ '%.3f'|format(t.pnl_pct|float) + '%' if t.pnl_pct else '—' }}</td>
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
    Re-baselines start_of_day_balance to whatever the balance is right now."""
    state = algo.load_state()
    try:
        current_balance = algo.get_wallet_total_balance()
        state["start_of_day_balance"] = current_balance
    except Exception:
        pass  # if balance fetch fails, still clear the tripped flag below
    state["daily_breaker_tripped"] = False
    algo.save_state(state)
    return jsonify({"ok": True, "message": "Circuit breaker reset. New entries allowed again."})


@app.route("/reset_min_balance_breaker", methods=["POST"])
def reset_min_balance_breaker():
    """Manually clears the minimum-balance floor breaker (the second,
    independent safety net that blocks new trades if the account ever
    falls below MIN_BALANCE_FLOOR_PCT of its all-time starting balance).
    This does NOT re-baseline the all-time starting balance — that number
    stays fixed on purpose, so resetting this only un-blocks new entries,
    it doesn't move the floor itself. Use this deliberately, only after
    understanding why it tripped."""
    state = algo.load_state()
    state["min_balance_breaker_tripped"] = False
    algo.save_state(state)
    return jsonify({"ok": True, "message": "Minimum-balance floor breaker reset. New entries allowed again."})


@app.route("/clear_trades_log", methods=["POST"])
def clear_trades_log():
    """Deletes the trades log CSV (for the given strategy) from both
    possible locations (primary and fallback), so old/malformed historical
    rows don't keep corrupting the reconstructed table display."""
    from flask import request
    strategy = request.args.get("strategy", "A")
    deleted = []
    primary = algo.TRADES_LOG if strategy == "A" else algo.TRADES_LOG_B
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
    </style>
</head>
<body>
    <div id="pnlWidget" class="flat">
        <h3>Live Trade P&amp;L</h3>
        <div id="pnlBody">No open position</div>
    </div>
    <h1>Trend-Scalp Algo — Dashboard</h1>
    <div style="text-align:center; margin-bottom:15px;">
        <a href="/trades" style="color:#4A90D9; font-size:15px; text-decoration:none;">
            &#128202; View Trades Log (Excel-style) &rarr;
        </a>
    </div>
    <div style="text-align:center; margin-bottom:15px;">
        <span style="color:#aaa; font-size:13px; margin-right:10px;">Active Logic:</span>
        <button id="modeABtn" onclick="setMode('A')" class="modeBtn">Logic A Only</button>
        <button id="modeBBtn" onclick="setMode('B')" class="modeBtn">Logic B Only</button>
        <button id="modeBothBtn" onclick="setMode('BOTH')" class="modeBtn">Both</button>
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
    </div>
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
        function setMode(mode) {
            fetch('/logic_mode', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({mode: mode})
            }).then(r => r.json()).then(data => {
                if (data.ok) refreshMode();
                else alert('Error: ' + data.error);
            });
        }
        function refreshMode() {
            fetch('/logic_mode').then(r => r.json()).then(data => {
                document.getElementById('modeABtn').classList.toggle('active', data.mode === 'A');
                document.getElementById('modeBBtn').classList.toggle('active', data.mode === 'B');
                document.getElementById('modeBothBtn').classList.toggle('active', data.mode === 'BOTH');
            });
        }
        function refreshPnl() {
            fetch('/live_pnl').then(r => r.json()).then(data => {
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
                    <div class="pnl-row"><span>Entry</span><span>${data.entry_price}</span></div>
                    <div class="pnl-row"><span>Current</span><span>${data.current_price ?? '—'}</span></div>
                    <div class="pnl-row"><span>Stop</span><span>${data.stop}</span></div>
                    <div class="pnl-row"><span>Target</span><span>${data.target}</span></div>
                    <div class="pnl-row"><span>Stop-loss steps locked</span><span>${data.strategy === 'A' ? data.milestones_locked + ' / 4' : 'N/A (fixed SL/TP)'}</span></div>
                `;
            });
        }
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
def get_logic_mode():
    return jsonify({"mode": algo.LOGIC_MODE["active"]})


@app.route("/logic_mode", methods=["POST"])
def set_logic_mode():
    from flask import request
    data = request.get_json(force=True)
    mode = data.get("mode")
    if mode not in ("A", "B", "BOTH"):
        return jsonify({"ok": False, "error": "mode must be A, B, or BOTH"}), 400
    algo.LOGIC_MODE["active"] = mode
    return jsonify({"ok": True, "mode": mode})


@app.route("/live_pnl")
def live_pnl():
    pos = algo.LATEST_STATE.get("position")
    if pos is None:
        return jsonify({"has_position": False})

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
    })


if __name__ == "__main__":
    print("Starting web dashboard at http://localhost:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False)
