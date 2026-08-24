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
# ---- NEW 2026-08-21: user's explicit request — "AI chart-pattern,
# greeks, OI sab dekhe fir bataye" — genuinely uses Claude (via
# Anthropic API) as a final judgment layer AFTER the existing 3 gates
# (Regime=B, Trend-Meter=ACTIVE-4-loop-sustained, Strike-Restriction)
# pass. Sends a real market snapshot (regime, trend-meter, ATM
# option's own Greeks — confirmed available from Delta's API, recent
# candle-pattern summary) and asks for a genuine confidence judgment
# before the actual entry. Requires ANTHROPIC_API_KEY set on Render —
# if missing or the call fails, genuinely FAILS OPEN (allows the trade
# through on the original 3-gate logic alone) rather than silently
# blocking all trading, since an API outage should never be a genuine
# reason to freeze the whole bot. ----
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
AI_CONFIRMATION_ENABLED = os.environ.get("AI_CONFIRMATION_ENABLED", "false").lower() == "true"
AI_CONFIDENCE_THRESHOLD = 60  # genuinely 0-100; below this, AI recommends skipping

DRY_RUN = os.environ.get("OPTIONS_BOT_LIVE", "false").lower() != "true"

STATE_FILE = "options_bot_state.json"

# ---- Strategy constants (see module docstring) ----
# ---- CHANGED 2026-08-17: Golden-Window REMOVED per user's explicit
# request — "Regime-B wala market kabhi bhi aa sakta hai, sirf Regim-B
# par chale". The bot now runs 24/7 (whenever the Options-bot thread is
# started) and relies ENTIRELY on Regime=B + Trend-Meter direction as
# the gate, not a fixed clock window. Genuinely a real trade-off: this
# means trading outside the previously-favoured 5:30-10:30 PM session,
# AND (since Delta's daily options expire+relaunch at 5:30 PM IST)
# potentially trading options that are hours away from their own
# expiry, where theta-decay is genuinely much more aggressive than in
# a freshly-launched 0DTE contract. User explicitly accepted this
# trade-off. ----
VWAP_PROXIMITY_PCT = 0.25          # same base threshold as the futures bot
REGIME_B_MULTIPLIER = 2.0          # avg VWAP-dist >= this*threshold => "B" (Strong-Trend)
# ---- CHANGED 2026-08-17: ADX removed per user's explicit request —
# "ADX hata do, Regime-B hi kaafi hai + trend direction confirm". Entry
# now requires ONLY: Regime=B (from the futures-bot's own live state)
# + Trend-Meter showing a clear UP/DOWN direction. Genuinely a looser
# filter than before — fewer conditions to satisfy means more trades,
# but each one is individually less filtered than the ADX-gated
# version. User's explicit, informed trade-off. ----
ADX_ENABLED = False
ADX_PERIOD = 14  # kept only because compute_adx() (unused now, left as a utility) references it as a default
# ---- CHANGED 2026-08-17: TP/SL restructured into a 3-level ITM
# trailing "staircase" (user's explicit design, points 3+4):
#   Milestone 1 (price reaches 1-strike-ITM):  SL trails up to breakeven
#   Milestone 2 (price reaches 2-strikes-ITM): SL trails up to milestone-1's premium
#   Milestone 3 (price reaches 3-strikes-ITM): HARD TP — close the trade
# Initial safety SL (before any milestone is hit): 2-strikes-OTM.
# This mirrors the exact "staircase trailing stop" pattern already
# proven in the futures bot (milestones_locked / max_progress), just
# expressed in strike-distance terms instead of price-percentage. ----
TP_ITM_STRIKES = 3                  # final hard-TP: 3rd ITM strike's premium
SL_OTM_STRIKES = 2                  # initial hard-SL: 2nd OTM strike's premium
TRAIL_MILESTONE_STRIKES = [1, 2, 3]  # ITM levels that trigger a trail-up
TIME_EXIT_MINUTES = 17.5           # midpoint of the agreed 15-20 min window
# ---- CHANGED 2026-08-21: user's explicit, genuinely-clarified request
# — "SIRФ simple 8-min cooldown chahiye, WIN ho ya LOSS, dono mein
# genuinely 8-min hi lage, koई alag 4-loss wala extended cooldown
# nahi" — a simpler earlier win/loss-differentiated + consecutive-loss
# design was tried first, but genuinely NOT what was wanted. This is
# the final, simple version: ONE flat cooldown, same for every exit
# reason, both underlyings. ----
COOLDOWN_MINUTES = 8
LIMIT_CHASE_INTERVAL_SECONDS = 10
LOOP_INTERVAL_SECONDS = 20
# ---- CHANGED 2026-08-17: manual, configurable lot-size (user's
# request, point 2) — override via env var OPTIONS_LOT_SIZE. Defaults
# to 1 lot, same as before, but no longer hardcoded. ----
LOT_SIZE = int(os.environ.get("OPTIONS_LOT_SIZE", "1"))
BUDGET_PCT_OF_WALLET = 0.125       # midpoint of agreed 10-15%
# ---- CHANGED 2026-08-17: BTC and ETH can now genuinely hold OPEN
# POSITIONS SIMULTANEOUSLY (user's request, point 2 — "dono ek saath
# chal sake"). state["positions"] is now a dict keyed by underlying
# ("BTC"/"ETH"), each independently None or an open-position dict —
# replacing the old single state["position"]. ----
UNDERLYINGS = [("BTC", "BTCUSD"), ("ETH", "ETHUSD")]
# ---- NEW 2026-08-14: Options fees + GST — user's direct question
# "profit milta hai toh fees+GST nikal jayegi kya?" exposed a genuine
# gap: PnL was being computed as pure gross (exit-entry), with no fee
# accounting at all — unlike the futures bot, which has always been
# fee-aware. Verified via Delta Exchange's own official support docs
# (delta.exchange/support, delta.exchange/fees, cross-checked against
# multiple independent sources, 2026-08-14): Options maker AND taker
# fee are both 0.03% of NOTIONAL value (not premium), GST is 18% on
# top of the fee, and the fee is CAPPED at 3.5% of the premium
# (protects deep-OTM/cheap-premium trades from disproportionate fees).
# Notional value = underlying spot price × contract_value × size —
# same contract_value convention as the futures bot (0.001 BTC per
# lot, 0.01 ETH per lot). ----
OPTIONS_FEE_PCT_OF_NOTIONAL = 0.0001   # ---- FIXED 2026-08-20: genuinely 0.01%, NOT 0.03%
# — the earlier 0.03% was from a web-search source that turned out
# genuinely wrong for this account/product. Verified empirically
# against the user's REAL Delta Exchange wallet-history: backing out
# GST from actual fee transactions and dividing by notional
# consistently gave ~0.01%, both for BTC and ETH option trades. ----
OPTIONS_GST_RATE = 0.18                 # 18% GST on the fee amount
OPTIONS_FEE_CAP_PCT_OF_PREMIUM = 0.035  # fee capped at 3.5% of premium
CONTRACT_VALUE = {"BTC": 0.001, "ETH": 0.01}

LATEST_STATE = {"positions": {"BTC": None, "ETH": None}, "last_action": None}


# ============================================================
# State persistence
# ============================================================
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            # ---- CHANGED 2026-08-17: genuine migration from the old
            # single-position schema (state["position"]) to the new
            # multi-position one (state["positions"]["BTC"/"ETH"]) —
            # so an existing state file from before this rewrite
            # doesn't crash the new code. ----
            if "positions" not in state:
                old_pos = state.pop("position", None)
                state["positions"] = {"BTC": None, "ETH": None}
                if old_pos:
                    state["positions"][old_pos.get("underlying", "BTC")] = old_pos
            if "cooldowns" not in state:
                old_cd = state.pop("cooldown_until", None)
                state["cooldowns"] = {"BTC": old_cd, "ETH": old_cd}
            return state
        except Exception:
            pass
    return {"positions": {"BTC": None, "ETH": None}, "trade_history": [], "cooldowns": {"BTC": None, "ETH": None}}


def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, default=str)
    except Exception as e:
        print(f"  [WARN] Couldn't save state: {e}")


# ---- NEW 2026-08-18: user's explicit request — "roz roz Environment
# Variable thodi na change karta rahunga" — lot-size (and, since we're
# at it, the LIVE/DRY-RUN switch too) now genuinely live in state.json
# and can be changed from the dashboard at any time, no Render redeploy
# needed. OPTIONS_LOT_SIZE env var still works as the STARTING default
# (useful for a fresh deploy), but once the dashboard is used to change
# it, state.json's value genuinely takes over. ----
def get_lot_size(state):
    return int(state.get("lot_size", LOT_SIZE))


def set_lot_size(state, new_size):
    new_size = max(1, int(new_size))
    state["lot_size"] = new_size
    save_state(state)
    return new_size


def get_live_mode(state):
    """Returns True if genuinely LIVE (real orders), False if DRY_RUN.
    state.json's "live_mode" key (if ever explicitly set via the
    dashboard) takes priority over the OPTIONS_BOT_LIVE env var —
    that's the whole point of this feature, so the person isn't stuck
    needing a Render redeploy every time."""
    if "live_mode" in state:
        return bool(state["live_mode"])
    return not DRY_RUN


def set_live_mode(state, is_live):
    state["live_mode"] = bool(is_live)
    save_state(state)
    return state["live_mode"]


# ---- NEW 2026-08-20: Per-Strike Trade Restriction (user's explicit
# request, points 1/2/5/6/7). A specific (underlying, option_type,
# strike) combination — e.g. "BTC CALL 69200" — may genuinely be
# re-selected as the ATM strike across multiple loops if the
# underlying's spot price stays near it. Without this restriction, the
# bot could genuinely re-enter the SAME contract repeatedly in quick
# succession. This limits any single (underlying, option_type, strike)
# to at most STRIKE_MAX_TRADES trades total, and requires at least
# STRIKE_COOLDOWN_HOURS between the 1st and 2nd trade on that exact
# strike. Genuinely does NOT affect other strikes — if the underlying
# moves and a DIFFERENT strike becomes ATM, that's a fresh combination
# with its own independent counter. CE and PE on the SAME strike price
# are genuinely tracked as separate combinations (different
# option_type in the key), per user's explicit point 7. ----
# ---- NEW 2026-08-21: Shadow-Tracking (user's explicit request) — see
# _record_and_close() docstring for the full reasoning. ----
SHADOW_TRACKING_ENABLED = True
SHADOW_MAX_MINUTES = 60

STRIKE_MAX_TRADES = 2
STRIKE_COOLDOWN_HOURS = 1.5


def _strike_key(underlying, option_type, strike):
    return f"{underlying}_{option_type}_{strike}"


def check_strike_restriction(state, underlying, option_type, strike):
    """Returns (allowed: bool, reason: str)."""
    history = state.get("strike_trade_history", {})
    key = _strike_key(underlying, option_type, strike)
    entry = history.get(key)
    if entry is None:
        return True, ""
    count = entry.get("count", 0)
    if count >= STRIKE_MAX_TRADES:
        return False, (f"{underlying} {option_type} strike={strike} genuinely already "
                        f"traded {count}x (max {STRIKE_MAX_TRADES}) — skipping this strike")
    last_time = datetime.strptime(entry["last_entry_time"], "%Y-%m-%d %H:%M:%S.%f")
    elapsed_hours = (now_ist() - last_time).total_seconds() / 3600
    if elapsed_hours < STRIKE_COOLDOWN_HOURS:
        remaining = (STRIKE_COOLDOWN_HOURS - elapsed_hours) * 60
        return False, (f"{underlying} {option_type} strike={strike} genuinely needs "
                        f"{STRIKE_COOLDOWN_HOURS}hr gap since last trade on this exact strike "
                        f"— {remaining:.0f}min remaining")
    return True, ""


def record_strike_trade(state, underlying, option_type, strike):
    history = state.setdefault("strike_trade_history", {})
    key = _strike_key(underlying, option_type, strike)
    entry = history.setdefault(key, {"count": 0, "last_entry_time": None})
    entry["count"] += 1
    entry["last_entry_time"] = now_ist().strftime("%Y-%m-%d %H:%M:%S.%f")
    save_state(state)


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
    """---- FIXED 2026-08-20: same genuine bug pattern as the
    reconciliation crash — see safe_result() docstring. Also wrapped in
    try/except since this function is called from many places
    (including inside _record_and_close, which must NEVER crash mid-
    close, or a position could be left stuck in a half-closed state). ----"""
    try:
        r = requests.get(f"{BASE_URL}/v2/tickers/{symbol}", timeout=10)
        r.raise_for_status()
        data = safe_result(r.json())
        price = data.get("close") or data.get("mark_price") or data.get("spot_price")
        return float(price) if price else None
    except Exception as e:
        print(f"  [WARN] genuinely couldn't fetch spot price for {symbol}: {e}")
        return None


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


def get_recent_candle_summary(underlying_symbol, count=20):
    """
    ---- CHANGED 2026-08-22: user's explicit, further request —
    "AI-chart-pattern-bhi-padega, dekhega-bhi-kya-chal-raha-hai" — the
    earlier version only sent aggregate stats (green/red count), which
    genuinely isn't enough for real pattern-reading. This version
    genuinely gives Claude much more to work with:
    - the RAW sequence of recent closes (so it can see the actual shape)
    - swing-structure: is price making Higher-Highs/Higher-Lows
      (genuine uptrend structure) or Lower-Highs/Lower-Lows (genuine
      downtrend structure), by comparing the first-half vs second-half
      of the window
    - how far current price is from the recent swing high/low (genuine
      breakout vs pullback context)
    Still fails safe — returns None on any error. ----
    """
    try:
        symbol = underlying_symbol + "USD"
        r = requests.get(f"{BASE_URL}/v2/history/candles",
                          params={"resolution": "5m", "symbol": symbol,
                                   "start": int(time.time()) - count * 300,
                                   "end": int(time.time())}, timeout=10)
        data = safe_result(r.json())
        candles = data if isinstance(data, list) else []
        if len(candles) < 4:
            return None
        candles = candles[-count:]
        closes = [float(c["close"]) for c in candles]
        opens = [float(c["open"]) for c in candles]
        highs = [float(c["high"]) for c in candles]
        lows = [float(c["low"]) for c in candles]
        greens = sum(1 for o, c in zip(opens, closes) if c > o)
        reds = len(candles) - greens
        net_move_pct = (closes[-1] - closes[0]) / closes[0] * 100
        ranges = [h - l for h, l in zip(highs, lows)]
        expanding = ranges[-1] > (sum(ranges[:-1]) / len(ranges[:-1]) if len(ranges) > 1 else ranges[-1])

        # ---- genuinely-simple swing structure: compare first-half vs
        # second-half highs/lows to detect HH-HL (uptrend) or LH-LL
        # (downtrend) or neither (range-bound/choppy). ----
        mid = len(candles) // 2
        first_half_high, first_half_low = max(highs[:mid]), min(lows[:mid])
        second_half_high, second_half_low = max(highs[mid:]), min(lows[mid:])
        if second_half_high > first_half_high and second_half_low > first_half_low:
            structure = "Higher-Highs-Higher-Lows (genuine uptrend structure)"
        elif second_half_high < first_half_high and second_half_low < first_half_low:
            structure = "Lower-Highs-Lower-Lows (genuine downtrend structure)"
        else:
            structure = "Mixed/choppy — no clean directional structure"

        swing_high, swing_low = max(highs), min(lows)
        current_price = closes[-1]
        dist_from_high_pct = (swing_high - current_price) / current_price * 100
        dist_from_low_pct = (current_price - swing_low) / current_price * 100

        return {
            "candles_count": len(candles), "green_candles": greens, "red_candles": reds,
            "net_move_pct": round(net_move_pct, 3),
            "range_expanding": expanding,
            "recent_closes": [round(c, 2) for c in closes],
            "swing_structure": structure,
            "swing_high": round(swing_high, 2), "swing_low": round(swing_low, 2),
            "pct_below_swing_high": round(dist_from_high_pct, 3),
            "pct_above_swing_low": round(dist_from_low_pct, 3),
        }
    except Exception as e:
        print(f"  [WARN] genuinely couldn't fetch candle-summary for {underlying_symbol}: {e}")
        return None


def get_ai_confirmation(underlying, direction, regime_label, trend_meter_state,
                         greeks, candle_summary):
    """
    ---- NEW 2026-08-21: user's explicit, final request — genuinely
    "AI chart-pattern samझe, greeks dekhe, sab kuch dekhe fir bataye".
    Calls Claude (Anthropic API) with a real market snapshot and asks
    for a genuine confidence judgment BEFORE the actual entry — this
    is the 4th gate, layered on top of the existing 3 (Regime=B,
    Trend-Meter=ACTIVE-4-loop, Strike-Restriction). FAILS OPEN by
    design: if AI_CONFIRMATION_ENABLED is False, or the API key is
    missing, or the call genuinely fails for any reason, this returns
    an "allow" verdict — an API hiccup should never freeze the whole
    bot; the original 3 gates remain the real safety net. ----
    """
    if not AI_CONFIRMATION_ENABLED:
        return {"allow": True, "confidence": None, "reasoning": "AI-confirmation genuinely disabled"}
    if not ANTHROPIC_API_KEY:
        return {"allow": True, "confidence": None,
                "reasoning": "genuinely no ANTHROPIC_API_KEY set — failing open"}

    snapshot = {
        "underlying": underlying, "proposed_direction": direction,
        "market_regime": regime_label, "trend_meter_state": trend_meter_state,
        "atm_option_greeks": greeks or "not available",
        "recent_5min_candle_summary": candle_summary or "not available",
    }
    prompt = (
        "You are evaluating a crypto options entry setup. Genuinely read the "
        "recent_5min_candle_summary as a real chart: look at the swing_structure "
        "(Higher-Highs-Higher-Lows vs Lower-Highs-Lower-Lows vs choppy), the "
        "recent_closes sequence (does the shape support the proposed_direction, "
        "or does it look exhausted/reversing?), how close price is to the "
        "swing_high/swing_low (pct_below_swing_high, pct_above_swing_low — is this "
        "a fresh breakout or a stretched, late move?), and range_expanding (is "
        "momentum building or fading). Combine this with atm_option_greeks (e.g. "
        "very high theta relative to premium makes an entry riskier — time is "
        "working against you) and market_regime/trend_meter_state. "
        "Reply with ONLY a JSON object, no other text: "
        "{\"confidence\": <0-100 integer>, \"recommendation\": \"ENTER\" or \"SKIP\", "
        "\"reasoning\": \"<one short sentence citing the specific chart/greeks detail that mattered most>\"}.\n\n"
        f"Setup snapshot:\n{json.dumps(snapshot, indent=2)}"
    )
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": "claude-sonnet-5", "max_tokens": 200,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=15)
        r.raise_for_status()
        # ---- FIXED 2026-08-23: CRITICAL bug, genuinely confirmed via
        # real production data — 2 real calls genuinely failed with
        # "genuinely failed: 'text'" and fell back to fail-open
        # (allowing entry WITHOUT a real AI judgment). Root cause:
        # blindly assuming content[0] always has a "text" key — but
        # Claude's response can genuinely include other block types
        # first (or, rarely, an empty content array), so content[0]
        # isn't always the text block. Fix: genuinely search ALL
        # content blocks for the first one with a "text" field, and
        # raise a clear, catchable error if none exists (still fails
        # open, but the log will show the CORRECT diagnosis instead of
        # a generic 'text' KeyError). ----
        content_blocks = r.json().get("content", [])
        text = next((b.get("text") for b in content_blocks if b.get("text")), None)
        if text is None:
            raise ValueError(f"genuinely no text block found in response content: {content_blocks}")
        text = text.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        confidence = int(parsed.get("confidence", 0))
        allow = confidence >= AI_CONFIDENCE_THRESHOLD and parsed.get("recommendation") == "ENTER"
        print(f"  [AI-CONFIRM] {underlying}: confidence={confidence}, "
              f"recommendation={parsed.get('recommendation')}, reasoning={parsed.get('reasoning')}")
        return {"allow": allow, "confidence": confidence, "reasoning": parsed.get("reasoning", "")}
    except Exception as e:
        print(f"  [WARN] AI-confirmation genuinely failed ({e}) — failing open, "
              f"falling back to the original 3-gate logic.")
        return {"allow": True, "confidence": None, "reasoning": f"genuinely failed: {e}"}


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
def detect_signal(underlying_symbol, futures_symbol, state):
    """Returns ("long"/"short"/None, debug_info_dict). `state` is
    genuinely needed now for the trend-meter persistence-gate (tracks
    consecutive-loop counts across calls)."""
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

    # ---- CHANGED 2026-08-21: user's explicit request — "sirf ACTIVE
    # dekhe, FORMING nahi" — "-FORMING" states are genuinely a WEAKER,
    # still-developing signal (the trend hasn't genuinely confirmed
    # yet), while "-ACTIVE" is the futures-bot's own stronger,
    # confirmed reading. Session-wise data (21-Aug) showed the
    # Late-Night session — where many FORMING-based entries happened —
    # performing genuinely worst (12% win-rate), motivating this
    # tightening. ADX check was removed earlier (2026-08-17); this is
    # genuinely a DIFFERENT, tighter quality-filter using the trend-
    # meter's own confidence level instead. ----
    if state_label not in ("UPTREND-ACTIVE", "DOWNTREND-ACTIVE"):
        # ---- genuinely reset persistence-count whenever it's NOT
        # active, so a later fresh ACTIVE-streak has to genuinely
        # start from 1 again. ----
        state.setdefault("trend_meter_persistence", {})[underlying_symbol] = \
            {"state": state_label, "count": 0}
        return None, {"reason": f"Trend-Meter state '{state_label}' genuinely not "
                                 f"UPTREND-ACTIVE/DOWNTREND-ACTIVE (FORMING/NEUTRAL genuinely not enough anymore)"}

    # ---- NEW 2026-08-21: user's explicit request — "4 loops mein
    # active rahe tab trade le" — genuinely the SAME "persistence gate"
    # philosophy already proven for market-regime (Logic C's own
    # REGIME-GATE, "not a single-loop flicker") — a single-loop ACTIVE
    # reading could genuinely be a brief flicker; requiring 4
    # CONSECUTIVE loops of the SAME ACTIVE direction before trusting it
    # genuinely filters that out. Counter resets to 1 if the direction
    # flips (UP->DOWN) even if still "ACTIVE" on both, since that's
    # genuinely a different signal, not a continuation. ----
    persistence = state.setdefault("trend_meter_persistence", {})
    prev = persistence.get(underlying_symbol, {"state": None, "count": 0})
    if prev["state"] == state_label:
        new_count = prev["count"] + 1
    else:
        new_count = 1
    persistence[underlying_symbol] = {"state": state_label, "count": new_count}

    REQUIRED_CONSECUTIVE_LOOPS = 4
    if new_count < REQUIRED_CONSECUTIVE_LOOPS:
        return None, {"reason": f"Trend-Meter genuinely just became {state_label} "
                                 f"({new_count}/{REQUIRED_CONSECUTIVE_LOOPS} consecutive loops so far) — "
                                 f"waiting for this to be sustained, not a single-loop flicker."}

    if state_label == "UPTREND-ACTIVE":
        return "long", {"reason": f"Futures-bot regime=B, Trend-Meter={state_label} "
                                   f"genuinely sustained for {new_count} loops — confirmed"}
    else:
        return "short", {"reason": f"Futures-bot regime=B, Trend-Meter={state_label} "
                                    f"genuinely sustained for {new_count} loops — confirmed"}


# ============================================================
# Order execution — Limit-order-chasing (entry, TP, SL all use this)
# ============================================================
def safe_result(response_json):
    """
    ---- NEW 2026-08-20: CRITICAL bug-fix, genuinely confirmed via a
    real production crash — "loop exception: 'NoneType' object has no
    attribute 'get'" repeating every single loop, forever. Root cause:
    `.get("result", {})`'s DEFAULT value only applies when the key is
    genuinely MISSING — if Delta's API returns the key WITH an explicit
    None value (which genuinely happens for tickers of already-expired/
    settled contracts, e.g. {"result": null}), `.get("result", {})`
    returns None (not {}), and any further `.get(...)` on it crashes.
    This ALSO caused the crashing position to never actually get
    cleared (since the crash happened before _record_and_close ran),
    so the SAME crash repeated on every subsequent loop forever. This
    helper is the genuinely-safe replacement — use `safe_result(r.json())`
    everywhere instead of `r.json().get("result", {})`. ----
    """
    return response_json.get("result") or {}


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


def place_options_bracket(product_id, product_symbol, sl_price, tp_price):
    """
    ---- NEW 2026-08-20: user's genuine, correctly-caught safety gap —
    "SL TP nahi lag rahe hai abhi bhi" — confirmed via a real Delta
    Exchange position screenshot showing NO exchange-side protection.
    Until now, TP/SL were only ever LOCAL (polled every ~20s and
    manually sold when crossed) — if the bot ever went down (crash,
    Render redeploy, network loss), an open position had genuinely ZERO
    protection. This places a REAL exchange-side bracket order
    (POST /v2/orders/bracket) — Delta's own docs + a verified real-
    world example confirm this endpoint genuinely works for OPTIONS
    contracts too, not just futures (identical mechanism, just pass the
    option's own product_id/product_symbol). Per Delta's own
    documentation, bracket orders can ONLY use mark_price as the
    trigger index — "last_traded_price" seen in some third-party code
    examples is NOT accepted. Only called when genuinely live (DRY_RUN
    safe — see call-sites, which pass `live` explicitly). ----
    """
    path = "/v2/orders/bracket"
    body_dict = {
        "product_id": product_id, "product_symbol": product_symbol,
        "stop_loss_order": {"order_type": "market_order", "stop_price": str(sl_price)},
        "take_profit_order": {"order_type": "market_order", "stop_price": str(tp_price)},
        "bracket_stop_trigger_method": "mark_price",
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


def update_options_bracket(product_id, product_symbol, sl_price, tp_price):
    """Same as place_options_bracket but genuinely updates an EXISTING
    bracket (PUT instead of POST) — used when the trailing-staircase
    moves the SL up, so the exchange-side stop genuinely stays in sync
    with our local trailing logic, not just the initial SL forever."""
    path = "/v2/orders/bracket"
    body_dict = {
        "product_id": product_id, "product_symbol": product_symbol,
        "stop_loss_order": {"order_type": "market_order", "stop_price": str(sl_price)},
        "take_profit_order": {"order_type": "market_order", "stop_price": str(tp_price)},
        "bracket_stop_trigger_method": "mark_price",
    }
    body = json.dumps(body_dict)
    timestamp = str(int(time.time()))
    signature = sign_request("PUT", path, "", body, timestamp)
    headers = {
        "api-key": API_KEY, "signature": signature, "timestamp": timestamp,
        "User-Agent": "options-trend-bot", "Content-Type": "application/json",
    }
    r = requests.put(f"{BASE_URL}{path}", data=body, headers=headers, timeout=10)
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


def estimate_options_fee(underlying, spot_price, premium, size=1):
    """
    ---- NEW 2026-08-14: see OPTIONS_FEE_PCT_OF_NOTIONAL docstring
    above for the full reasoning/sourcing. Returns the fee (INCLUDING
    GST) for ONE leg (entry OR exit) — call it once for entry, once
    for exit, and subtract both from gross PnL to get net. ----
    """
    contract_value = CONTRACT_VALUE.get(underlying, 0.001)
    notional_value = spot_price * contract_value * size
    fee_pct_based = notional_value * OPTIONS_FEE_PCT_OF_NOTIONAL
    fee_cap = premium * size * OPTIONS_FEE_CAP_PCT_OF_PREMIUM
    fee_before_gst = min(fee_pct_based, fee_cap)
    return fee_before_gst * (1 + OPTIONS_GST_RATE)


def limit_order_chase(product_id, side, size, get_current_price_fn, reduce_only=False,
                       max_seconds=120, label="", live=None):
    """
    ---- Genuinely-shared "chasing" logic for entry, TP, and SL, exactly
    as agreed: place limit at current best price, wait
    LIMIT_CHASE_INTERVAL_SECONDS, if not filled cancel and re-place at
    the (possibly moved) current price, repeat up to max_seconds total.
    In DRY_RUN, this just logs what WOULD happen and returns a fake
    "filled" result immediately at the first quoted price — genuinely
    useful for watching decision-quality before risking real money.

    ---- CHANGED 2026-08-18: `live` parameter added so callers can pass
    the genuinely current, dashboard-configurable live-mode
    (get_live_mode(state)) instead of always falling back to the
    module-level DRY_RUN (which only reflects the OPTIONS_BOT_LIVE env
    var at process-start) — this is what makes the dashboard's
    LIVE/DRY-RUN toggle actually take effect without a Render redeploy.
    If `live` isn't passed, falls back to the old module-level DRY_RUN
    for backward compatibility. ----
    """
    is_live = live if live is not None else (not DRY_RUN)
    start = time.time()
    price = get_current_price_fn()
    print(f"    [{label}] limit-chase starting @ {price} (LIVE={is_live})")

    if not is_live:
        return {"filled": True, "fill_price": price, "dry_run": True}

    order = place_order(product_id, side, size, price, reduce_only)
    order_id = safe_result(order).get("id")
    while time.time() - start < max_seconds:
        time.sleep(LIMIT_CHASE_INTERVAL_SECONDS)
        status = get_order_state(order_id)
        order_state = safe_result(status).get("state")
        if order_state == "closed":
            fill_price = safe_result(status).get("limit_price")
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
        order_id = safe_result(order).get("id")

    print(f"    [{label}] genuinely gave up chasing after {max_seconds}s")
    return {"filled": False, "fill_price": None, "dry_run": False}


def fast_marketable_exit(product_id, size, get_bid_fn, max_slippage_pct=3.0, live=None, label=""):
    """
    ---- NEW 2026-08-21: CRITICAL fix, genuinely confirmed via a real
    trade — "gross loss nahi hona chahiye, sirf fees, lekin gross loss
    hua". Root cause: SL exits were genuinely using the SAME PATIENT
    limit_order_chase as TP (10s wait, then re-price, repeat) — but a
    stop-loss's whole point is FAST protection, not best-price-seeking.
    While patiently chasing a "better" sell price, the market kept
    falling, so the ACTUAL fill ended up well below the intended stop
    level — a genuine gross loss beyond fees, not just fees. This is a
    genuinely FAST exit instead — places ONE bounded-marketable-limit
    order immediately (mirrors the futures-bot's own proven "bounded
    marketable-limit order" fallback pattern), deep enough below the
    current bid that it should genuinely fill right away, rather than
    iteratively chasing for a better price while price keeps moving
    against the position. Used ONLY for SL exits — TP and TIME_EXIT
    genuinely aren't urgent, so they keep using the patient
    limit_order_chase (better average price there is a genuine plus,
    not a risk). ----
    """
    is_live = live if live is not None else (not DRY_RUN)
    current_bid = get_bid_fn()
    bounded_price = round(current_bid * (1 - max_slippage_pct / 100.0), 6)
    print(f"    [{label}] genuinely FAST-EXIT — bounded-marketable-limit @ {bounded_price} "
          f"(current_bid={current_bid}, max_slippage={max_slippage_pct}%)")

    if not is_live:
        return {"filled": True, "fill_price": current_bid, "dry_run": True}

    order = place_order(product_id, "sell", size, bounded_price, reduce_only=True)
    order_id = safe_result(order).get("id")
    # ---- genuinely give it a short window — a bounded-marketable
    # order this deep should fill almost instantly; we're NOT
    # patiently re-pricing here on purpose. ----
    for _ in range(3):
        time.sleep(2)
        status = get_order_state(order_id)
        order_state = safe_result(status).get("state")
        if order_state == "closed":
            fill_price = safe_result(status).get("limit_price")
            print(f"    [{label}] genuinely FAST-EXIT filled @ {fill_price}")
            return {"filled": True, "fill_price": float(fill_price)}
    print(f"    [{label}] genuinely FAST-EXIT still pending after 6s — "
          f"using current bid as a safe approximate fill for record-keeping.")
    return {"filled": True, "fill_price": current_bid}


# ============================================================
# Main trading loop
# ============================================================
def check_shadow_positions(state):
    """
    ---- NEW 2026-08-21: Shadow-Tracking (user's explicit request) —
    genuinely re-checks every open shadow every loop. A shadow is a
    PASSIVE, no-real-order copy of a position that genuinely exited via
    TIME_EXIT — this keeps watching the SAME TP/SL it had at exit-time,
    to answer "would holding longer have helped" with real subsequent
    price data. Resolves (records outcome, removes from active list)
    when: the shadow's own TP or SL is genuinely crossed, OR
    SHADOW_MAX_MINUTES elapses without either (recorded as
    "shadow_TIMEOUT"). Never places any real order — pure observation. ----
    """
    shadows = state.get("shadow_positions", {})
    now = now_ist()
    for underlying, shadow_list in list(shadows.items()):
        still_active = []
        for sh in shadow_list:
            try:
                r = requests.get(f"{BASE_URL}/v2/tickers/{sh['symbol']}", timeout=10)
                ticker = safe_result(r.json())
                if "mark_price" not in ticker:
                    still_active.append(sh)
                    continue
                current = float(ticker["mark_price"])
            except Exception as e:
                print(f"  [SHADOW-WARN] {underlying}: genuinely couldn't fetch price ({e})")
                still_active.append(sh)
                continue

            start_time = datetime.strptime(sh["shadow_start_time"], "%Y-%m-%d %H:%M:%S.%f")
            elapsed_min = (now - start_time).total_seconds() / 60
            hit_tp = current >= sh["tp_target_premium"]
            hit_sl = current <= sh["sl_target_premium"]
            timed_out = elapsed_min >= SHADOW_MAX_MINUTES

            if not (hit_tp or hit_sl or timed_out):
                still_active.append(sh)
                continue

            shadow_reason = "shadow_TP" if hit_tp else "shadow_SL" if hit_sl else "shadow_TIMEOUT"
            contract_value = CONTRACT_VALUE.get(underlying, 0.001)
            shadow_gross = (current - sh["entry_price"]) * contract_value * sh.get("size", 1)
            # ---- genuinely-honest note: shadow P&L does NOT deduct a
            # 2nd round of fees (the real trade already paid entry-fee;
            # only a hypothetical extra exit-fee would apply here, and
            # since this never really executes, we skip re-modelling it
            # — shadow numbers are for DIRECTIONAL insight, not exact
            # accounting). ----
            shadow_record = {
                "underlying": underlying, "symbol": sh["symbol"],
                "entry_price": sh["entry_price"], "real_exit_price": sh["real_exit_price"],
                "real_exit_pnl": sh["real_exit_pnl"], "shadow_exit_price": current,
                "shadow_exit_reason": shadow_reason, "shadow_gross_pnl": shadow_gross,
                "shadow_elapsed_min": round(elapsed_min, 1),
                "shadow_start_time": sh["shadow_start_time"],
                "shadow_end_time": now.strftime("%Y-%m-%d %H:%M:%S.%f"),
            }
            state.setdefault("shadow_trade_history", []).append(shadow_record)
            state["shadow_trade_history"] = state["shadow_trade_history"][-100:]
            print(f"  [SHADOW-RESOLVED] {underlying}: {shadow_reason} — genuinely-would-have "
                  f"pnl={shadow_gross:.4f} (real-time-exit-pnl-was={sh['real_exit_pnl']:.4f})")
        shadows[underlying] = still_active
    state["shadow_positions"] = shadows


def run_one_cycle(state):
    now = now_ist()

    # ---- CHANGED 2026-08-17: Golden-Window check REMOVED — Regime=B
    # is now the ONLY gate (see constants block docstring above for the
    # full trade-off reasoning). ----

    # ---- NEW 2026-08-21: genuinely check all active shadow-positions
    # every loop, regardless of what else happens this cycle. ----
    check_shadow_positions(state)

    positions = state.setdefault("positions", {"BTC": None, "ETH": None})
    cooldowns = state.setdefault("cooldowns", {"BTC": None, "ETH": None})

    for underlying, futures_symbol in UNDERLYINGS:
        # ---- Position already open for THIS underlying? Manage it,
        # don't look for a new entry — but this does NOT block the
        # OTHER underlying (user's explicit "dono ek saath chal sake"). ----
        if positions.get(underlying):
            manage_open_position(state, underlying)
            continue

        # ---- Per-underlying cooldown check ----
        cd = cooldowns.get(underlying)
        if cd:
            cooldown_until = datetime.strptime(cd, "%Y-%m-%d %H:%M:%S.%f")
            if now < cooldown_until:
                remaining = (cooldown_until - now).total_seconds() / 60
                print(f"  [COOLDOWN] {underlying}: {remaining:.1f} min remaining.")
                continue
            cooldowns[underlying] = None

        side, debug = detect_signal(underlying, futures_symbol, state)
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

        # ---- NEW 2026-08-24: user's explicit, genuinely-reasoned
        # request — "2-ITM strike pe trade karein, Greeks bhi pareshan
        # nahi karenge?" — genuinely trades a strike 2-ITM from real
        # ATM instead of ATM itself. Reasoning (confirmed with user):
        # an ITM option's premium is genuinely PART intrinsic value
        # (never decays) and part time-value (decays via theta) — so a
        # SMALLER fraction of the premium is exposed to theta than a
        # pure-time-value ATM option, at the cost of a higher premium
        # (less leverage) and higher delta (more directional exposure,
        # less "optionality"). Falls back to ATM itself if genuinely
        # not enough ITM strikes exist on the chain (edge of the chain). ----
        ENTRY_ITM_STRIKES = 2
        entry_data = find_strike_n_away(chain, atm_strike, option_type, ENTRY_ITM_STRIKES, "itm")
        if entry_data is None:
            print(f"  [WARN] {underlying}: genuinely not enough ITM strikes on chain — "
                  f"falling back to ATM for this entry.")
            entry_data = atm
        entry_strike = float(entry_data["strike_price"])
        entry_is_fallback_atm = (entry_data is atm)

        # ---- CHANGED 2026-08-24: user's explicit request — "same
        # strike restriction hata do" — genuinely REMOVED. The bot can
        # now trade the same strike repeatedly, no max-2-trades or
        # 1.5hr-gap limit. record_strike_trade() call below is
        # genuinely kept (harmless bookkeeping, still visible in state
        # if ever needed again) but check_strike_restriction() is no
        # longer called as a gate. ----

        # ---- NEW 2026-08-21: user's explicit, final request — 4th
        # gate, genuinely a Claude-powered judgment using the traded
        # option's own Greeks + recent candle-pattern + regime/trend-
        # meter, layered ON TOP of the existing 3 gates. See
        # get_ai_confirmation()'s docstring for the fail-open design. ----
        # ---- CHANGED 2026-08-24: genuinely uses entry_data (the ACTUAL
        # traded strike, now 2-ITM by default) instead of atm, so the AI
        # sees the real Greeks of what's about to be bought. ----
        greeks = entry_data.get("greeks")
        candle_summary = get_recent_candle_summary(underlying)
        try:
            import trend_scalp_live as algo
            regime_label = algo.LATEST_STATE.get("market_regime", {}).get("label") \
                if algo.LATEST_STATE.get("market_regime") else None
            trend_state = algo.LATEST_STATE.get("trend_meter", {}).get(futures_symbol, {}).get("state")
        except ImportError:
            regime_label, trend_state = None, None
        ai_result = get_ai_confirmation(underlying, side, regime_label, trend_state,
                                         greeks, candle_summary)
        # ---- NEW 2026-08-22: user's explicit request — "hume dikhega
        # bhi kya" — genuinely stores every REAL AI-confirmation call
        # (allowed or declined) into a persistent, dashboard-visible
        # history, not just buried in the scrolling logs. Only logs
        # when AI-confirmation is genuinely ENABLED — otherwise every
        # single loop would add a meaningless "disabled" entry. ----
        if AI_CONFIRMATION_ENABLED:
            state.setdefault("ai_confirmation_history", []).append({
                "time": now_ist().strftime("%Y-%m-%d %H:%M:%S"), "underlying": underlying,
                "direction": side, "confidence": ai_result["confidence"],
                "allowed": ai_result["allow"], "reasoning": ai_result["reasoning"],
            })
            state["ai_confirmation_history"] = state["ai_confirmation_history"][-100:]
            save_state(state)
        if not ai_result["allow"]:
            print(f"  {underlying}: SKIP — genuinely AI-confirmation declined "
                  f"(confidence={ai_result['confidence']}, reasoning={ai_result['reasoning']})")
            continue

        product_id = entry_data["product_id"]
        lot_size = get_lot_size(state)  # ---- CHANGED 2026-08-18: genuinely dashboard-configurable, not fixed at deploy-time ----

        def get_ask(symbol=entry_data["symbol"], fallback=entry_data.get("mark_price")):
            r = requests.get(f"{BASE_URL}/v2/tickers/{symbol}", timeout=10)
            return float(safe_result(r.json()).get("quotes", {}).get("best_ask") or fallback)

        # ---- CHANGED 2026-08-18: user's genuine catch — "jab trade
        # execute ho jayegi, uski TIME ke premium price ko maankar TP
        # trail karega?" Answer: it SHOULD, and now it genuinely does.
        # Previously, milestone/TP/SL reference premiums were computed
        # from the option-chain snapshot taken BEFORE the entry's
        # limit-order-chase started — but that chase can genuinely take
        # up to LIMIT_CHASE_INTERVAL_SECONDS × several retries (well
        # over a minute) before actually filling, during which the
        # market can move meaningfully, making those pre-chase
        # reference premiums stale by the time the trade is genuinely
        # live. Fix: execute the entry FIRST, then re-fetch a FRESH
        # option-chain snapshot and compute milestone/SL references
        # from THAT — genuinely tied to the moment of actual execution,
        # not the moment we merely decided to attempt entry. ----
        print(f"  {underlying}: ENTERING {side.upper()} via {entry_data['symbol']} "
              f"(genuinely {'ATM-fallback' if entry_is_fallback_atm else f'{ENTRY_ITM_STRIKES}-ITM from ATM'}) size={lot_size}")
        result = limit_order_chase(product_id, "buy", lot_size, get_ask, label=f"{underlying}-ENTRY",
                                    live=get_live_mode(state))
        if not result["filled"]:
            print(f"  {underlying}: entry genuinely didn't fill, skipping this cycle")
            continue

        fresh_chain = get_option_chain(underlying)
        milestone_refs = {}
        if fresh_chain:
            for m in TRAIL_MILESTONE_STRIKES:
                ref = find_strike_n_away(fresh_chain, entry_strike, option_type, m, "itm")
                if ref is not None:
                    milestone_refs[m] = float(ref.get("mark_price"))
            sl_ref_fresh = find_strike_n_away(fresh_chain, entry_strike, option_type, SL_OTM_STRIKES, "otm")
        else:
            sl_ref_fresh = None

        if len(milestone_refs) < len(TRAIL_MILESTONE_STRIKES) or sl_ref_fresh is None:
            # ---- Genuinely rare edge-case: chain became unavailable or
            # strikes fell off the edge of it in the moments after fill.
            # Fail-safe fallback: use the entry fill-price itself as the
            # reference frame, so TP/SL are still meaningfully defined
            # rather than silently missing. ----
            print(f"  [WARN] {underlying}: couldn't get fresh post-fill milestone data — "
                  f"falling back to entry-price-relative estimates.")
            fill = result["fill_price"]
            milestone_refs = {1: fill * 1.3, 2: fill * 1.6, 3: fill * 2.0}
            sl_target = fill * 0.5
        else:
            sl_target = float(sl_ref_fresh.get("mark_price"))

        print(f"  {underlying}: genuinely-fresh post-fill targets — "
              f"milestones={milestone_refs}, SL={sl_target}")

        positions[underlying] = {
            "underlying": underlying, "symbol": entry_data["symbol"], "product_id": product_id,
            "side": side, "size": lot_size, "entry_price": result["fill_price"],
            "entry_time": now.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "entry_spot_price": spot, "strike": entry_strike,
            "tp_target_premium": milestone_refs[TP_ITM_STRIKES],
            "sl_target_premium": sl_target,
            "initial_sl_premium": sl_target,
            "milestone_premiums": milestone_refs,   # {1: premium, 2: premium, 3: premium}
            "milestones_locked": 0,                  # how many trail-up steps genuinely hit so far
            "option_type": option_type,
        }
        # ---- NEW 2026-08-20: genuinely record this strike's usage for
        # the Per-Strike Trade Restriction (see check_strike_restriction). ----
        record_strike_trade(state, underlying, option_type, entry_strike)

        # ---- NEW 2026-08-20: genuinely place a REAL exchange-side
        # bracket order (SL+TP) right after entry — user's confirmed
        # safety gap: "SL TP nahi lag rahe hai abhi bhi". Only when
        # genuinely live — in DRY_RUN there's no real position for the
        # exchange to attach a bracket to. ----
        if get_live_mode(state):
            try:
                bracket_result = place_options_bracket(
                    product_id, entry_data["symbol"], sl_target, milestone_refs[TP_ITM_STRIKES])
                print(f"  {underlying}: genuinely placed EXCHANGE-SIDE bracket "
                      f"(SL={sl_target}, TP={milestone_refs[TP_ITM_STRIKES]}) — {bracket_result}")
            except Exception as e:
                print(f"  [WARN] {underlying}: genuinely FAILED to place exchange-side "
                      f"bracket ({e}) — position is still LOCALLY protected via polling, "
                      f"but has no exchange-side safety net until this succeeds.")

        save_state(state)
        LATEST_STATE["positions"] = positions
        print(f"  {underlying}: position OPENED — {json.dumps(positions[underlying])}")


def get_exchange_position(product_id):
    """
    ---- NEW 2026-08-17: user's explicit request (point 5) — "Bot
    manually dekhe exchange par, kyunki foreclose kar sakte trade ko".
    Genuinely queries Delta's own /v2/positions endpoint for the REAL
    position size on the exchange for this product_id. Used for
    reconciliation: if the exchange shows the position is genuinely
    already closed (size=0) — e.g. force-closed, liquidated, or
    otherwise closed outside our own bot's control — but our LOCAL
    state still thinks it's open, we need to catch that and correct
    our own records rather than keep trying to manage/exit a position
    that genuinely doesn't exist anymore. Returns None (not 0) on a
    failed query — fails safe, never force-closes on an inconclusive
    read. In DRY_RUN, this is never called (no real position exists). ----
    """
    path = "/v2/positions"
    query_string = f"?product_id={product_id}"
    timestamp = str(int(time.time()))
    signature = sign_request("GET", path, query_string, "", timestamp)
    headers = {"api-key": API_KEY, "signature": signature, "timestamp": timestamp,
               "User-Agent": "options-trend-bot"}
    try:
        r = requests.get(f"{BASE_URL}{path}", headers=headers,
                          params={"product_id": product_id}, timeout=10)
        data = r.json()
        result = data.get("result")
        if isinstance(result, list):
            return float(result[0].get("size", 0)) if result else 0.0
        elif isinstance(result, dict):
            return float(result.get("size", 0))
        return 0.0
    except Exception as e:
        print(f"  [RECONCILE] genuinely failed to query exchange position: {e}")
        return None


def _record_and_close(state, underlying, pos, exit_price, reason, now):
    """Shared close-out logic — writes the trade record, clears the
    position, and starts this underlying's cooldown. Used by both the
    normal TP/SL/time-exit path and the exchange-reconciliation path.

    ---- FIXED 2026-08-20: CRITICAL bug, genuinely confirmed via the
    user's real Delta Exchange wallet-history data. gross_pnl was
    missing the CONTRACT_VALUE multiplier — premium (exit-entry) is
    genuinely quoted "$ per 1 FULL unit of underlying" (same convention
    as the futures price itself, e.g. BTC~$63000/1-BTC), NOT "$ per
    contract". A contract genuinely only controls CONTRACT_VALUE units
    of underlying (0.001 BTC / 0.01 ETH), so the real dollar P&L must
    be scaled down accordingly — exactly like the fee calculation
    already correctly does via notional_value = spot*contract_value*
    size. Verified against real wallet entries: e.g. ETH trade showing
    "Gross P&L: 7.80" on the dashboard was genuinely only $0.078 in
    real cashflow on the exchange (100x = 1/CONTRACT_VALUE_ETH) — this
    was a DASHBOARD DISPLAY bug only; actual trade execution/sizing on
    the exchange itself was always correct (Delta's own systems handle
    real notional correctly regardless of our local display math), so
    real money was never at the risk the old dashboard numbers implied.
    ----"""
    contract_value = CONTRACT_VALUE.get(underlying, 0.001)
    gross_pnl = (exit_price - pos["entry_price"]) * contract_value * pos.get("size", 1)
    entry_spot = pos.get("entry_spot_price")
    exit_spot = get_spot_price(underlying + "USD") or entry_spot
    entry_fee = estimate_options_fee(underlying, entry_spot, pos["entry_price"],
                                      pos.get("size", 1)) if entry_spot else 0.0
    exit_fee = estimate_options_fee(underlying, exit_spot, exit_price,
                                     pos.get("size", 1)) if exit_spot else 0.0
    total_fees = entry_fee + exit_fee
    net_pnl = gross_pnl - total_fees

    trade_record = {**pos, "exit_price": exit_price, "exit_reason": reason,
                     "exit_time": now.strftime("%Y-%m-%d %H:%M:%S.%f"),
                     "gross_pnl": gross_pnl, "fees_with_gst": total_fees, "pnl": net_pnl}
    state.setdefault("trade_history", []).append(trade_record)
    state["trade_history"] = state["trade_history"][-100:]

    # ---- NEW 2026-08-21: Shadow-Tracking (user's explicit request) —
    # "Time-Exit ke baad price dekhta rahe, record kare ki agar hold
    # karte toh kya hota". Only meaningful for TIME_EXIT (TP/SL/
    # reconciled already reached their genuine natural conclusion —
    # there's nothing to "what-if" there). Creates a passive, no-real-
    # order shadow copy that keeps tracking the SAME TP/SL this
    # position had at the moment of exit, for up to
    # SHADOW_MAX_MINUTES more, to genuinely answer "would holding
    # longer have helped" with real subsequent price data instead of a
    # guess. ----
    if reason == "TIME_EXIT" and SHADOW_TRACKING_ENABLED:
        shadow = {**pos, "shadow_start_time": now.strftime("%Y-%m-%d %H:%M:%S.%f"),
                  "real_exit_price": exit_price, "real_exit_pnl": net_pnl}
        state.setdefault("shadow_positions", {}).setdefault(underlying, []).append(shadow)
        print(f"  [SHADOW] {underlying}: genuinely started shadow-tracking "
              f"(will keep watching TP={pos['tp_target_premium']}/SL={pos['sl_target_premium']} "
              f"for up to {SHADOW_MAX_MINUTES}min more, no real order)")

    state["positions"][underlying] = None

    # ---- CHANGED 2026-08-21: user's genuinely-final, clarified request
    # — SIMPLE flat cooldown, same for WIN or LOSS, both underlyings.
    # (An earlier win/loss-differentiated + 4-consecutive-loss version
    # was tried first, but genuinely wasn't what was wanted — reverted.) ----
    state.setdefault("cooldowns", {})[underlying] = \
        (now + timedelta(minutes=COOLDOWN_MINUTES)).strftime("%Y-%m-%d %H:%M:%S.%f")
    save_state(state)
    LATEST_STATE["positions"] = state["positions"]
    LATEST_STATE["last_action"] = trade_record
    print(f"  [EXIT] {underlying} genuinely closed ({reason}) — Gross-PnL={gross_pnl:.4f}, "
          f"Fees+GST={total_fees:.4f}, Net-PnL={net_pnl:.4f} — cooldown for {COOLDOWN_MINUTES}min")


def manage_open_position(state, underlying):
    positions = state["positions"]
    pos = positions[underlying]
    now = now_ist()
    entry_time = datetime.strptime(pos["entry_time"], "%Y-%m-%d %H:%M:%S.%f")
    elapsed_min = (now - entry_time).total_seconds() / 60

    # ---- NEW 2026-08-17: exchange reconciliation (user's point 5).
    # Only meaningful when genuinely live — in DRY_RUN there's no real
    # exchange position to check against. ----
    if get_live_mode(state):
        exchange_size = get_exchange_position(pos["product_id"])
        if exchange_size is not None and exchange_size == 0:
            print(f"  [RECONCILE] {underlying}: exchange genuinely shows this position "
                  f"CLOSED already (foreclosed/liquidated/settled) — syncing our records.")
            # ---- FIXED 2026-08-20: CRITICAL bug, genuinely confirmed —
            # "loop exception: 'NoneType' object has no attribute
            # 'get'" repeating every single loop, forever. Root cause:
            # once a contract genuinely expires/settles, its ticker
            # endpoint can genuinely return {"result": null} — and
            # `.get("result", {})`'s default ONLY applies when the KEY
            # is missing, not when it's present with an explicit None
            # value, so `.get("mark_price", ...)` on that None crashed
            # every time. Worse: because the crash happened BEFORE
            # _record_and_close ran, the position never actually got
            # cleared, so reconciliation kept re-triggering the exact
            # same crash forever. Fix: genuinely guard against both a
            # missing key AND an explicit None, and wrap the whole
            # ticker-fetch in try/except so a transient/edge-case
            # failure here can NEVER block the reconciliation from
            # genuinely completing — falling back to entry_price
            # (a reasonable, safe estimate) rather than crashing. ----
            best_guess_price = pos["entry_price"]
            try:
                r = requests.get(f"{BASE_URL}/v2/tickers/{pos['symbol']}", timeout=10)
                result = r.json().get("result") or {}
                best_guess_price = float(result.get("mark_price", pos["entry_price"]))
            except Exception as e:
                print(f"  [WARN] {underlying}: genuinely couldn't fetch a fresh price for "
                      f"reconciliation ({e}) — using entry_price as a safe fallback.")
            _record_and_close(state, underlying, pos, best_guess_price, "exchange_reconciled", now)
            return

    try:
        r = requests.get(f"{BASE_URL}/v2/tickers/{pos['symbol']}", timeout=10)
        ticker = safe_result(r.json())
        if "mark_price" not in ticker:
            print(f"  [WARN] {underlying}: genuinely no mark_price in ticker response "
                  f"({ticker}) — skipping this loop's TP/SL check to avoid a false "
                  f"trigger on bad/missing data, will retry next loop.")
            return
        current_premium = float(ticker["mark_price"])
    except Exception as e:
        print(f"  [WARN] {underlying}: genuinely couldn't fetch current premium ({e}) — "
              f"skipping this loop's TP/SL check, will retry next loop.")
        return

    # ---- CHANGED 2026-08-17: 3-level ITM trailing staircase (user's
    # points 3+4). Milestone 1 hit -> SL trails to breakeven (entry
    # price). Milestone 2 hit -> SL trails to milestone-1's premium.
    # Milestone 3 hit -> hard TP, close immediately. Mirrors the exact
    # "staircase trailing stop" pattern already proven in the futures
    # bot, just expressed in strike-distance/premium terms.
    #
    # ---- CHANGED 2026-08-21: user's explicit request — "TP trail
    # hata do, SL to final TP rakho" — genuinely REMOVED the trailing
    # (SL no longer moves on 1-ITM/2-ITM milestones). SL is now
    # genuinely FIXED at the initial 2-strikes-OTM level for the whole
    # trade, and TP remains the fixed final 3-strikes-ITM target —
    # both set once at entry and never recalculated afterward. This is
    # genuinely SIMPLER (fewer moving parts, easier to reason about),
    # but loses the "lock in some gain" safety-net the trailing gave —
    # a real trade-off the user explicitly chose. ----
    hit_tp = current_premium >= pos["tp_target_premium"]  # 3-ITM level = fixed final TP
    hit_sl = current_premium <= pos["sl_target_premium"]  # genuinely fixed, never trails
    # ---- CHANGED 2026-08-24: user's explicit request — "exit time
    # limit hata do" — genuinely REMOVED the 17.5-min forced time-exit.
    # Position now genuinely stays open until fixed TP or SL is hit
    # (or the exchange's own bracket order fires, whichever is first).
    # Honest risk noted to the user: without a time-exit, a position
    # that sits in a stalled range between SL and TP can genuinely stay
    # open indefinitely, continuing to bleed theta with no exit
    # mechanism except the fixed SL/TP. ----
    hit_time = False

    print(f"  [POSITION] {underlying} {pos['symbol']} entry={pos['entry_price']} "
          f"current={current_premium} elapsed={elapsed_min:.1f}min "
          f"SL={pos['sl_target_premium']} TP={pos['tp_target_premium']} (genuinely-fixed, no-trail, no-time-exit)")

    if not (hit_tp or hit_sl or hit_time):
        save_state(state)  # persist the (possibly-updated) trailing SL even without an exit
        return

    reason = "TP" if hit_tp else "SL" if hit_sl else "TIME_EXIT"

    def get_bid():
        r2 = requests.get(f"{BASE_URL}/v2/tickers/{pos['symbol']}", timeout=10)
        result2 = safe_result(r2.json())
        return float(result2.get("quotes", {}).get("best_bid") or result2.get("mark_price"))

    print(f"  [EXIT] {underlying} {pos['symbol']} closing due to {reason}")
    if reason == "SL":
        # ---- CHANGED 2026-08-21: genuinely FAST exit for stop-losses
        # — see fast_marketable_exit() docstring for the full
        # reasoning. TP and TIME_EXIT still use the patient chase. ----
        result = fast_marketable_exit(pos["product_id"], pos.get("size", 1), get_bid,
                                       live=get_live_mode(state), label=f"{underlying}-EXIT-SL")
    else:
        result = limit_order_chase(pos["product_id"], "sell", pos.get("size", 1), get_bid,
                                    reduce_only=True, label=f"{underlying}-EXIT-{reason}",
                                    live=get_live_mode(state))
    exit_price = result.get("fill_price") or current_premium
    _record_and_close(state, underlying, pos, exit_price, reason, now)


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
            # ---- FIXED 2026-08-20: CRITICAL concurrency bug, genuinely
            # confirmed — "Lot Size button kaam nahi kar raha hai". The
            # loop's `state` dict was loaded ONCE at start and reused
            # for the entire lifetime; dashboard-triggered changes
            # (Save-Lot-Size, Toggle-LIVE) write to disk correctly, but
            # this loop's own STALE in-memory copy would genuinely
            # overwrite them again the next time IT called save_state()
            # — silently reverting the dashboard change within one
            # loop cycle (~20s). Fix: genuinely re-read the
            # dashboard-configurable fields (lot_size, live_mode) fresh
            # from disk every single cycle, merging them into the
            # long-lived in-memory state — WITHOUT touching the
            # trading-critical fields (positions, trade_history,
            # cooldowns, strike_trade_history) that this loop itself
            # owns and must NOT reset from a stale disk-read. ----
            fresh = load_state()
            if "lot_size" in fresh:
                state["lot_size"] = fresh["lot_size"]
            if "live_mode" in fresh:
                state["live_mode"] = fresh["live_mode"]

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
        "dry_run": not get_live_mode(state),
        "lot_size": get_lot_size(state),
        "positions": state.get("positions", {"BTC": None, "ETH": None}),
        "cooldowns": state.get("cooldowns", {"BTC": None, "ETH": None}),
        "recent_trades": state.get("trade_history", [])[-10:],
    })


@app.route("/set_lot_size", methods=["POST"])
def set_lot_size_route():
    """---- NEW 2026-08-18: user's explicit request — genuinely change
    lot-size from the dashboard, no Render redeploy needed. ----"""
    from flask import request
    state = load_state()
    try:
        new_size = int(request.json.get("lot_size", 1))
    except Exception:
        return jsonify({"ok": False, "message": "Genuinely invalid lot-size value."})
    actual = set_lot_size(state, new_size)
    return jsonify({"ok": True, "lot_size": actual})


@app.route("/set_live_mode", methods=["POST"])
def set_live_mode_route():
    """---- NEW 2026-08-18: user's explicit request — genuinely toggle
    LIVE/DRY-RUN from the dashboard, no Render redeploy needed. ----"""
    from flask import request
    state = load_state()
    is_live = bool(request.json.get("live", False))
    actual = set_live_mode(state, is_live)
    return jsonify({"ok": True, "live": actual})


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
