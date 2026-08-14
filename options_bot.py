"""
============================================================
OPTIONS TREND-BOT — Full System (DRY-RUN safe by default)
============================================================
Genuinely-independent from the futures bot (trend_scalp_live.py /
app.py) — separate state, separate loop, separate Render service.

STRATEGY (as agreed, 2026-08-14):
  Window:    5:30 PM - 10:30 PM IST (Golden Window, post daily-expiry)
  Condition: Market-Regime = "B" (Strong-Trend) + Trend-Meter UP/DOWN
             + ADX >= 25
  Entry:     ATM option, LIMIT-order-chasing (10s cycle, re-price+retry)
  TP:        2-strikes-ITM's current premium value (LIMIT-chasing)
  SL:        3-strikes-OTM's current premium value (LIMIT-chasing)
  Time-exit: 15-20 minutes if neither TP nor SL hit
  Cooldown:  15-20 minutes after any exit, then re-check from scratch
  Sizing:    1 lot, capped at ~10-15% of wallet balance per trade
  Leverage:  full (buying options — no liquidation risk)

SAFETY: DRY_RUN = True by default. In dry-run, every entry/exit is
fully computed and LOGGED as if it happened, but NO real order is
ever placed. Flip DRY_RUN to False (via the dashboard toggle or the
env var OPTIONS_BOT_LIVE=true) only after watching dry-run behavior
and being comfortable with it — same cautious pattern used throughout
this project's other new features.
============================================================
"""

import os
import time
import json
import threading
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from flask import Flask, jsonify

# ============================================================
# Config
# ============================================================
BASE_URL = "https://api.india.delta.exchange"
IST_OFFSET = timedelta(hours=5, minutes=30)


def now_ist():
    """Naive datetime whose wall-clock values are IST — deliberately no
    tzinfo attached, matching the exact pattern used in the futures
    bot's now_ist() (trend_scalp_live.py). Genuinely IMPORTANT: mixing
    naive and aware datetimes causes a hard TypeError on subtraction —
    this bug was caught by testing (manage_open_position crashed
    comparing an aware now_ist() against a naive strptime()-parsed
    entry_time) and fixed before this ever reached a live deployment."""
    return (datetime.now(timezone.utc) + IST_OFFSET).replace(tzinfo=None)

API_KEY = os.environ.get("DELTA_API_KEY", "")
API_SECRET = os.environ.get("DELTA_API_SECRET", "")

DRY_RUN = os.environ.get("OPTIONS_BOT_LIVE", "false").lower() != "true"

STATE_FILE = "options_bot_state.json"

# ---- Strategy constants (see module docstring) ----
WINDOW_START_HOUR, WINDOW_START_MIN = 17, 30
WINDOW_END_HOUR, WINDOW_END_MIN = 22, 30
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
VWAP_PROXIMITY_PCT = 0.25          # same base threshold as the futures bot
REGIME_B_MULTIPLIER = 2.0          # avg VWAP-dist >= this*threshold => "B" (Strong-Trend)
TP_ITM_STRIKES = 2
SL_OTM_STRIKES = 3
TIME_EXIT_MINUTES = 17.5           # midpoint of the agreed 15-20 min window
COOLDOWN_MINUTES = 17.5            # midpoint of the agreed 15-20 min window
LIMIT_CHASE_INTERVAL_SECONDS = 10
LOOP_INTERVAL_SECONDS = 20
BUDGET_PCT_OF_WALLET = 0.125       # midpoint of agreed 10-15%
UNDERLYINGS = [("BTC", "BTCUSD"), ("ETH", "ETHUSD")]

LATEST_STATE = {"position": None, "last_action": None}


# ============================================================
# State persistence
# ============================================================
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"position": None, "trade_history": [], "cooldown_until": None}


def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, default=str)
    except Exception as e:
        print(f"  [WARN] Couldn't save state: {e}")


# ============================================================
# Market-data helpers (self-contained — no dependency on
# trend_scalp_live.py, per "genuinely alag file" agreement)
# ============================================================
def fetch_candles(symbol, hours=6, resolution="15m"):
    end = int(datetime.now(timezone.utc).timestamp())
    start = end - hours * 3600
    r = requests.get(f"{BASE_URL}/v2/history/candles",
                      params={"symbol": symbol, "resolution": resolution,
                              "start": start, "end": end}, timeout=15)
    r.raise_for_status()
    data = r.json().get("result", [])
    if not data:
        return None
    df = pd.DataFrame(data).rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df.sort_values("timestamp").reset_index(drop=True)


def compute_vwap(df):
    df = df.copy()
    typical = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap"] = (typical * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, np.nan)
    return df


def compute_ema(df, period):
    df = df.copy()
    df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
    return df


def compute_adx(df, period=ADX_PERIOD):
    """Standard Wilder's ADX — same formula used in the futures bot's
    compute_adx (kept in sync deliberately, copied not imported, since
    this file is meant to stand alone)."""
    df = df.copy()
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    prev_close = close.shift()
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
                    axis=1).max(axis=1)
    atr_smooth = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * (plus_dm_smooth / atr_smooth.replace(0, np.nan))
    minus_di = 100 * (minus_dm_smooth / atr_smooth.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return df


def get_spot_price(symbol):
    r = requests.get(f"{BASE_URL}/v2/tickers/{symbol}", timeout=10)
    r.raise_for_status()
    data = r.json().get("result", {})
    price = data.get("close") or data.get("mark_price") or data.get("spot_price")
    return float(price) if price else None


def in_golden_window():
    now = now_ist()
    start = now.replace(hour=WINDOW_START_HOUR, minute=WINDOW_START_MIN, second=0, microsecond=0)
    end = now.replace(hour=WINDOW_END_HOUR, minute=WINDOW_END_MIN, second=0, microsecond=0)
    return start <= now <= end


def get_todays_expiry_date():
    """See options_chain_check.py's identical function for the full
    reasoning (5:30 PM IST daily-expiry cutoff, verified 2026-08-14)."""
    now = now_ist()
    cutoff = now.replace(hour=17, minute=30, second=0, microsecond=0)
    expiry_date = now.date() if now < cutoff else (now + timedelta(days=1)).date()
    return expiry_date.strftime("%d-%m-%Y")


def get_option_chain(underlying_symbol, expiry_date=None):
    if expiry_date is None:
        expiry_date = get_todays_expiry_date()
    r = requests.get(f"{BASE_URL}/v2/tickers",
                      params={"contract_types": "call_options,put_options",
                              "underlying_asset_symbols": underlying_symbol,
                              "expiry_date": expiry_date}, timeout=10)
    r.raise_for_status()
    data = r.json()
    return data.get("result", []) if data.get("success") else []


def find_atm_strike(chain, spot_price, option_type):
    candidates = [t for t in chain if t.get("contract_type") == option_type
                  and t.get("strike_price") is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda t: abs(float(t["strike_price"]) - spot_price))


def find_strike_n_away(chain, atm_strike_price, option_type, n, direction):
    candidates = sorted(
        [t for t in chain if t.get("contract_type") == option_type
         and t.get("strike_price") is not None],
        key=lambda t: float(t["strike_price"]))
    strikes = [float(t["strike_price"]) for t in candidates]
    if atm_strike_price not in strikes:
        return None
    atm_idx = strikes.index(atm_strike_price)
    is_call = option_type == "call_options"
    if (direction == "itm" and is_call) or (direction == "otm" and not is_call):
        target_idx = atm_idx - n
    else:
        target_idx = atm_idx + n
    if 0 <= target_idx < len(candidates):
        return candidates[target_idx]
    return None


# ============================================================
# Signal detection: Genuinely REUSES the futures bot's OWN live
# Market-Regime + Trend-Meter (via LATEST_STATE) — user's explicit
# instruction, 2026-08-14: "future wale environment/trending mode
# lagao, alag se kuch nahi" — no separate re-implementation. Since
# options_bot.py runs in the SAME process as the futures-bot dashboard
# (see app.py's threading.Thread(target=options_bot.bot_loop...)),
# trend_scalp_live's own loop is ALREADY computing these every ~20
# seconds and writing them into trend_scalp_live.LATEST_STATE — this
# just reads that directly, genuinely the exact same numbers the
# futures-bot dashboard itself shows. ADX stays as options-bot's own
# additional check (not present in the futures bot at all originally,
# added here as agreed 2026-08-14 for the options-specific "genuinely
# fast enough move" requirement) — but the underlying direction+regime
# read is now genuinely identical to the futures bot's.
# ============================================================
def detect_signal(underlying_symbol, futures_symbol):
    """Returns ("long"/"short"/None, debug_info_dict)."""
    try:
        import trend_scalp_live as algo
    except ImportError:
        return None, {"reason": "trend_scalp_live genuinely not importable — "
                                 "options-bot must run in the same process as app.py"}

    regime = algo.LATEST_STATE.get("market_regime")
    if regime is None or regime.get("favors") != "B":
        label = regime.get("label") if regime else "not-computed-yet"
        return None, {"reason": f"Futures-bot's own regime is genuinely NOT 'B' "
                                 f"(currently: {label})"}

    meter = algo.LATEST_STATE.get("trend_meter")
    if meter is None or underlying_symbol + "USD" not in meter:
        return None, {"reason": f"Futures-bot's own Trend-Meter has no reading yet for "
                                 f"{underlying_symbol}"}
    symbol_meter = meter[underlying_symbol + "USD"]
    state_label = symbol_meter.get("state")

    # ---- ADX: options-bot's own additional speed/consistency check,
    # computed here since the futures bot doesn't use ADX at all. ----
    df = fetch_candles(futures_symbol, hours=6, resolution="15m")
    if df is None or len(df) < ADX_PERIOD * 2 + 5:
        return None, {"reason": "not enough candle data yet for ADX"}
    df = compute_adx(df, ADX_PERIOD)
    adx_val = df["adx"].iloc[-1]
    if pd.isna(adx_val) or adx_val < ADX_TREND_THRESHOLD:
        return None, {"reason": f"Futures-bot regime is 'B' and Trend-Meter says "
                                 f"{state_label}, but ADX={adx_val:.1f} is below "
                                 f"threshold {ADX_TREND_THRESHOLD}"}

    if state_label in ("UPTREND-ACTIVE", "UPTREND-FORMING"):
        return "long", {"reason": f"Futures-bot regime=B, Trend-Meter={state_label}, "
                                   f"ADX={adx_val:.1f} confirmed", "adx": round(adx_val, 1)}
    elif state_label in ("DOWNTREND-ACTIVE", "DOWNTREND-FORMING"):
        return "short", {"reason": f"Futures-bot regime=B, Trend-Meter={state_label}, "
                                    f"ADX={adx_val:.1f} confirmed", "adx": round(adx_val, 1)}
    return None, {"reason": f"Trend-Meter state '{state_label}' genuinely not "
                             f"UP/DOWN (probably NEUTRAL)"}


# ============================================================
# Order execution — Limit-order-chasing (entry, TP, SL all use this)
# ============================================================
def sign_request(method, path, query_string, body, timestamp):
    import hashlib
    import hmac
    message = method + timestamp + path + query_string + body
    return hmac.new(API_SECRET.encode(), message.encode(), hashlib.sha256).hexdigest()


def place_order(product_id, side, size, limit_price, reduce_only=False):
    """Places a single LIMIT order. Real API call — only reached when
    DRY_RUN is False. Returns the order response dict."""
    path = "/v2/orders"
    body_dict = {
        "product_id": product_id, "size": size, "side": side,
        "order_type": "limit_order", "limit_price": str(limit_price),
        "reduce_only": reduce_only,
    }
    body = json.dumps(body_dict)
    timestamp = str(int(time.time()))
    signature = sign_request("POST", path, "", body, timestamp)
    headers = {
        "api-key": API_KEY, "signature": signature, "timestamp": timestamp,
        "User-Agent": "options-trend-bot", "Content-Type": "application/json",
    }
    r = requests.post(f"{BASE_URL}{path}", data=body, headers=headers, timeout=10)
    return r.json()


def cancel_order(order_id, product_id):
    path = "/v2/orders"
    body = json.dumps({"id": order_id, "product_id": product_id})
    timestamp = str(int(time.time()))
    signature = sign_request("DELETE", path, "", body, timestamp)
    headers = {
        "api-key": API_KEY, "signature": signature, "timestamp": timestamp,
        "User-Agent": "options-trend-bot", "Content-Type": "application/json",
    }
    r = requests.delete(f"{BASE_URL}{path}", data=body, headers=headers, timeout=10)
    return r.json()


def get_order_state(order_id):
    path = f"/v2/orders/{order_id}"
    timestamp = str(int(time.time()))
    signature = sign_request("GET", path, "", "", timestamp)
    headers = {"api-key": API_KEY, "signature": signature, "timestamp": timestamp,
               "User-Agent": "options-trend-bot"}
    r = requests.get(f"{BASE_URL}{path}", headers=headers, timeout=10)
    return r.json()


def limit_order_chase(product_id, side, size, get_current_price_fn, reduce_only=False,
                       max_seconds=120, label=""):
    """
    ---- Genuinely-shared "chasing" logic for entry, TP, and SL, exactly
    as agreed: place limit at current best price, wait
    LIMIT_CHASE_INTERVAL_SECONDS, if not filled cancel and re-place at
    the (possibly moved) current price, repeat up to max_seconds total.
    In DRY_RUN, this just logs what WOULD happen and returns a fake
    "filled" result immediately at the first quoted price — genuinely
    useful for watching decision-quality before risking real money. ----
    """
    start = time.time()
    price = get_current_price_fn()
    print(f"    [{label}] limit-chase starting @ {price} (DRY_RUN={DRY_RUN})")

    if DRY_RUN:
        return {"filled": True, "fill_price": price, "dry_run": True}

    order = place_order(product_id, side, size, price, reduce_only)
    order_id = order.get("result", {}).get("id")
    while time.time() - start < max_seconds:
        time.sleep(LIMIT_CHASE_INTERVAL_SECONDS)
        status = get_order_state(order_id)
        state = status.get("result", {}).get("state")
        if state == "closed":
            fill_price = status.get("result", {}).get("limit_price")
            print(f"    [{label}] genuinely filled @ {fill_price}")
            return {"filled": True, "fill_price": float(fill_price), "dry_run": False}
        # not filled yet — cancel and re-price
        try:
            cancel_order(order_id, product_id)
        except Exception as e:
            print(f"    [{label}] cancel failed (may already be filled/gone): {e}")
        new_price = get_current_price_fn()
        print(f"    [{label}] not filled in {LIMIT_CHASE_INTERVAL_SECONDS}s, "
              f"re-pricing {price} -> {new_price}")
        price = new_price
        order = place_order(product_id, side, size, price, reduce_only)
        order_id = order.get("result", {}).get("id")

    print(f"    [{label}] genuinely gave up chasing after {max_seconds}s")
    return {"filled": False, "fill_price": None, "dry_run": False}


# ============================================================
# Main trading loop
# ============================================================
def run_one_cycle(state):
    now = now_ist()

    if not in_golden_window():
        return  # silently idle outside 5:30-10:30 PM IST

    # ---- Cooldown check ----
    if state.get("cooldown_until"):
        cooldown_until = datetime.strptime(state["cooldown_until"], "%Y-%m-%d %H:%M:%S.%f")
        if now < cooldown_until:
            remaining = (cooldown_until - now).total_seconds() / 60
            print(f"  [COOLDOWN] {remaining:.1f} min remaining before next check.")
            return
        state["cooldown_until"] = None

    # ---- Already in a position? Manage it, don't look for new entries ----
    if state.get("position"):
        manage_open_position(state)
        return

    # ---- Look for a fresh entry across both underlyings ----
    for underlying, futures_symbol in UNDERLYINGS:
        side, debug = detect_signal(underlying, futures_symbol)
        print(f"  {underlying}: {debug.get('reason')}")
        if side is None:
            continue

        spot = get_spot_price(futures_symbol)
        option_type = "call_options" if side == "long" else "put_options"
        chain = get_option_chain(underlying)
        if not chain:
            print(f"  [WARN] {underlying}: empty option-chain (auction window? wrong expiry?)")
            continue

        atm = find_atm_strike(chain, spot, option_type)
        if atm is None:
            print(f"  [WARN] {underlying}: no ATM strike found")
            continue
        atm_strike = float(atm["strike_price"])

        tp_ref = find_strike_n_away(chain, atm_strike, option_type, TP_ITM_STRIKES, "itm")
        sl_ref = find_strike_n_away(chain, atm_strike, option_type, SL_OTM_STRIKES, "otm")
        if tp_ref is None or sl_ref is None:
            print(f"  [WARN] {underlying}: couldn't find TP/SL reference strikes (edge of chain)")
            continue

        product_id = atm["product_id"]

        def get_ask():
            r = requests.get(f"{BASE_URL}/v2/tickers/{atm['symbol']}", timeout=10)
            return float(r.json().get("result", {}).get("quotes", {}).get("best_ask")
                         or atm.get("mark_price"))

        print(f"  {underlying}: ENTERING {side.upper()} via {atm['symbol']} "
              f"(TP-ref={tp_ref.get('mark_price')}, SL-ref={sl_ref.get('mark_price')})")
        result = limit_order_chase(product_id, "buy", 1, get_ask, label=f"{underlying}-ENTRY")
        if not result["filled"]:
            print(f"  {underlying}: entry genuinely didn't fill, skipping this cycle")
            continue

        state["position"] = {
            "underlying": underlying, "symbol": atm["symbol"], "product_id": product_id,
            "side": side, "entry_price": result["fill_price"],
            "entry_time": now.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "tp_target_premium": float(tp_ref.get("mark_price")),
            "sl_target_premium": float(sl_ref.get("mark_price")),
            "option_type": option_type,
        }
        save_state(state)
        LATEST_STATE["position"] = state["position"]
        print(f"  {underlying}: position OPENED — {json.dumps(state['position'])}")
        return  # one position at a time, genuinely simple


def manage_open_position(state):
    pos = state["position"]
    now = now_ist()
    entry_time = datetime.strptime(pos["entry_time"], "%Y-%m-%d %H:%M:%S.%f")
    elapsed_min = (now - entry_time).total_seconds() / 60

    r = requests.get(f"{BASE_URL}/v2/tickers/{pos['symbol']}", timeout=10)
    ticker = r.json().get("result", {})
    current_premium = float(ticker.get("mark_price", 0))

    hit_tp = current_premium >= pos["tp_target_premium"]
    hit_sl = current_premium <= pos["sl_target_premium"]
    hit_time = elapsed_min >= TIME_EXIT_MINUTES

    print(f"  [POSITION] {pos['symbol']} entry={pos['entry_price']} current={current_premium} "
          f"elapsed={elapsed_min:.1f}min TP-target={pos['tp_target_premium']} "
          f"SL-target={pos['sl_target_premium']}")

    if not (hit_tp or hit_sl or hit_time):
        return

    reason = "TP" if hit_tp else "SL" if hit_sl else "TIME_EXIT"

    def get_bid():
        r2 = requests.get(f"{BASE_URL}/v2/tickers/{pos['symbol']}", timeout=10)
        return float(r2.json().get("result", {}).get("quotes", {}).get("best_bid")
                     or r2.json().get("result", {}).get("mark_price"))

    print(f"  [EXIT] {pos['symbol']} closing due to {reason}")
    result = limit_order_chase(pos["product_id"], "sell", 1, get_bid, reduce_only=True,
                                label=f"{pos['underlying']}-EXIT-{reason}")
    exit_price = result.get("fill_price") or current_premium
    pnl = exit_price - pos["entry_price"]

    trade_record = {**pos, "exit_price": exit_price, "exit_reason": reason,
                     "exit_time": now.strftime("%Y-%m-%d %H:%M:%S.%f"), "pnl": pnl}
    state.setdefault("trade_history", []).append(trade_record)
    state["trade_history"] = state["trade_history"][-100:]
    state["position"] = None
    state["cooldown_until"] = (now + timedelta(minutes=COOLDOWN_MINUTES)).strftime("%Y-%m-%d %H:%M:%S.%f")
    save_state(state)
    LATEST_STATE["position"] = None
    LATEST_STATE["last_action"] = trade_record
    print(f"  [EXIT] genuinely closed — PnL={pnl:.4f} — cooldown for {COOLDOWN_MINUTES}min")


def bot_loop(stop_event=None):
    """
    ---- CHANGED 2026-08-14: now accepts an optional stop_event, so this
    can genuinely be started/stopped via the dashboard's own "Options"
    button (see app.py's /start_options, /stop_options), same pattern
    as the futures bot's own Start/Stop. No longer auto-starts on
    process-boot (see the bottom of this file) — user's explicit
    request: 3 independent buttons (Signal / Future / Options), not
    Options tied to the process lifecycle. ----
    """
    def should_stop():
        return stop_event is not None and stop_event.is_set()

    state = load_state()
    print(f"Options-bot starting. DRY_RUN={DRY_RUN}")
    while not should_stop():
        try:
            run_one_cycle(state)
        except Exception as e:
            print(f"  [ERROR] loop exception: {e}")
        for _ in range(LOOP_INTERVAL_SECONDS):
            if should_stop():
                break
            time.sleep(1)
    print("Options-bot genuinely stopped.")


# ============================================================
# Flask app (for Render — status/dashboard only, matches the
# existing bot's pattern of a lightweight web-service wrapper)
# ============================================================
app = Flask(__name__)


@app.route("/")
def dashboard():
    state = load_state()
    return jsonify({
        "dry_run": DRY_RUN,
        "in_golden_window": in_golden_window(),
        "position": state.get("position"),
        "cooldown_until": state.get("cooldown_until"),
        "recent_trades": state.get("trade_history", [])[-10:],
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "time_ist": now_ist().strftime("%Y-%m-%d %H:%M:%S")})


if __name__ == "__main__":
    # ---- Genuinely for standalone testing/running this file directly
    # ONLY (e.g. python3 options_bot.py on its own machine). When
    # deployed as part of the combined app.py service (the actual
    # Render setup), THIS block never runs — app.py's own dashboard
    # buttons (/start_options etc.) control the loop instead. ----
    t = threading.Thread(target=bot_loop, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
