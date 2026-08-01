"""
LIVE TESTNET TREND-SCALPING ALGO — 200 EMA + VWAP + CVD confluence.

v2 — Major fixes and upgrades based on real testnet issues found:
  1. FIXED: get_live_orders() returns a LIST, not a dict — status checks
     were silently always failing before.
  2. FIXED: create_order()/cancel_order() response parsing — the SDK
     returns the order dict directly (no nested "result" key), so order_id
     extraction was broken, which broke cancel_order() too.
  3. FIXED: closing orders now send reduce_only=True, so closing a
     position doesn't get treated as a fresh trade requiring new margin
     (this was the direct cause of "insufficient_margin" on exits).
  4. NEW: startup reconciliation — checks REAL exchange positions/orders
     before doing anything, instead of blindly trusting the local state
     file. Adopts a real open position if one exists, and cancels any
     stray leftover orders for our watched products to prevent duplicates.
  5. NEW: the whole loop body is wrapped so ONE bad iteration (a crash,
     an API hiccup) does NOT stop the script — it logs the error and
     keeps going. The script only stops on Ctrl+C / GUI stop button.
  6. NEW: partial take-profit — books 50% of the position once price
     reaches 50% of the way to target, and moves the stop to breakeven
     for the remaining size.
  7. NEW: trailing stop-loss — once in meaningful profit, the stop
     continuously trails behind price (never loosens).
  8. NEW: smart near-target exit — if price gets close to target (80%+
     of the way there) and then starts reversing, exits immediately with
     whatever profit is available instead of risking a round-trip back
     to the stop.
  9. Strictly ONE position at a time, enforced by reconciliation.

Requires: pip install delta-rest-client pandas requests
Fill in config.py with your TESTNET api_key/api_secret before running.
"""
import time
import json
import os
import traceback
from datetime import datetime, timezone, timedelta

# All logging/display timestamps below use IST (UTC+5:30) instead of the
# server's raw system time (Render's containers run on UTC) — purely for
# readability, so log times match what the user actually sees on their
# clock in India without needing to mentally add 5:30 every time. This
# does NOT affect any actual date/time-based LOGIC — the daily circuit
# breaker's day-boundary reset (check_daily_loss_circuit_breaker) and any
# other real time-based decisions still explicitly use UTC, unaffected.
IST_OFFSET = timedelta(hours=5, minutes=30)


def now_ist():
    """Naive datetime whose wall-clock values are IST, for display in
    logs only — deliberately has no tzinfo attached so it prints as a
    clean 'YYYY-MM-DD HH:MM:SS' with no confusing UTC-offset suffix.
    Never used for any actual time-based logic/comparisons."""
    return (datetime.now(timezone.utc) + IST_OFFSET).replace(tzinfo=None)

import requests
import pandas as pd
from delta_rest_client import DeltaRestClient

import config
from trend_scalp_indicators import (compute_ema, compute_vwap, compute_cvd, cvd_rising,
                                     cvd_falling, compute_rsi, compute_atr)

# ---------------- SETTINGS ----------------
SYMBOLS_TO_WATCH = [
    # RESTRICTED back to BTC/ETH only (2026-07-30). SOL/XRP were added
    # temporarily on 2026-07-29 while BTC/ETH had stuck residual positions
    # blocking new trades — that issue has since cleared. Meanwhile SOLUSD
    # just produced ANOTHER catastrophic-fill incident (three LONG entries
    # in ~20 minutes, real fill ~$77.85 vs signal price ~$73.95 — a ~5.2%
    # jump), confirming the same testnet-liquidity problem that caused
    # every prior incident on the thinner altcoins (DOGE, ADA). The
    # slippage safety-net caught each one and closed immediately, so no
    # major loss occurred — but the underlying cause (SOL's testnet
    # liquidity) hasn't improved, so back to BTC/ETH-only.
    "BTCUSD", "ETHUSD",
    # Still excluded (all previously problematic): "DOGEUSD", "ADAUSD", "SOLUSD", "XRPUSD",
]

# Logic B specifically only watches these — the smaller, low-priced coins
# (SOL/XRP/DOGE/ADA) kept getting skipped constantly because their ATR is
# too tiny relative to price, making MIN_STOP_DIST_PCT_B unreachable most
# of the time. BTC/ETH have large enough absolute price moves for Logic B's
# ATR-based stop to clear that floor regularly.
LOGIC_B_SYMBOLS = ["BTCUSD", "ETHUSD"]
RISK_PER_TRADE_PCT = 1.5

# Safety cap for Logic A: if a large slippage/gap on the real entry fill
# makes the ACTUAL stop distance (relative to the real fill price) more
# than this many times the stop distance the position size was originally
# calculated for, the real dollar-risk on the trade has ballooned well
# beyond what RISK_PER_TRADE_PCT was meant to cap — close the position
# immediately instead of holding it. See look_for_entry_a() for where
# this is used (found in practice: a 3% testnet slippage turned an
# intended 0.51% risk into a realized 3.41% risk, a 6.7x blowup).
MAX_SLIPPAGE_RISK_MULT = 2.0
LEVERAGE = 10
MARGIN_SAFETY_FACTOR = 0.90    # only use 90% of calculated max notional, leaves buffer
EMA_PERIOD = 200
VWAP_PROXIMITY_PCT = 0.25  # Widened from 0.15% -> 0.25% (see conversation): after
                           # restricting Logic A to BTC/ETH only, this filter was
                           # found to be starving Logic A of almost all entries,
                           # since these two coins trend/move away from VWAP more
                           # than the altcoins previously watched. 0.25% was chosen
                           # as the value that captures real near-miss cases seen
                           # in practice (0.23-0.245% distances) without loosening
                           # much further than that.
CVD_LOOKBACK = 10
SWING_LOOKBACK = 15
STOP_BUFFER_PCT = 0.05
RISK_REWARD_MULT = 2.5

# ---- Minimum stop-distance floor (Logic A) ----
# Logic B already has this (MIN_STOP_DIST_PCT_B below) — Logic A never
# had the equivalent. Found in practice (comparing dashboard trades vs
# exchange history): during quiet/low-volatility stretches, the raw
# swing-based stop (swing_low/swing_high over SWING_LOOKBACK candles)
# can end up razor-tight (e.g. 0.05-0.10%). Since position size below is
# risk_amount / stop_dist_pct, a razor-tight stop makes size balloon
# straight to the max-leverage cap (verified: one BTCUSD trade's notional
# landed within $6 of the exact max_notional ceiling) — and since target
# = stop_dist_pct * RISK_REWARD_MULT, the target ends up just as tiny,
# so ordinary market noise (and the staircase trailing-stop's early
# milestones) satisfies it almost immediately. Net effect: heavy lots,
# but tiny realized gains, trades "cut" almost as soon as they open.
# Same fix as Logic B: WIDEN the stop to this floor instead of skipping
# the trade — position size shrinks accordingly to keep the same $ risk,
# and the target/staircase milestones become real, capturable moves.
MIN_STOP_DIST_PCT_A = 0.30
LOOKBACK_HOURS = 8              # Logic B (1-min candles) lookback

# ---- BUG FIX (Logic B): early-exit delayed from 1.0R to 1.5R ----
# See the full explanation next to its usage in manage_bracket_position_b.
EARLY_EXIT_R_MULTIPLE = 1.5
LOGIC_A_RESOLUTION = "15m"      # Logic A now runs on 15-min candles (better fit for 200 EMA)
LOGIC_A_LOOKBACK_HOURS = 60     # 60 hours = 240 fifteen-min candles, comfortably > 200 needed for EMA200
LOOP_INTERVAL_SECONDS = 20

MAKER_FEE_PCT = 0.04
TAKER_FEE_PCT = 0.06
ROUND_TRIP_FEE_PCT = MAKER_FEE_PCT * 2
SAFETY_MARGIN_MULT = 2.0
MIN_TARGET_PCT = ROUND_TRIP_FEE_PCT * SAFETY_MARGIN_MULT  # Logic A (limit-first, maker-fee based)

# Logic B ALWAYS uses market orders (never gets the cheaper maker fee), so
# its fee-aware filter must be based on TAKER fees, not the maker-based one above.
MIN_TARGET_PCT_B = (TAKER_FEE_PCT * 2) * SAFETY_MARGIN_MULT

LIMIT_ORDER_TIMEOUT_SECONDS = 45
LIMIT_OFFSET_PCT = 0.02

# Safety bound for CLOSING a position when the limit-first attempt above
# doesn't fill in time. Previously this fell back to a true, UNBOUNDED
# market order — found in practice to be dangerous on testnet's thin
# order books: one ADAUSD close filled at $0.00001 (real price was
# ~$0.155 at the time) because a plain market order accepts literally
# ANY price with no limit. Instead, fall back to a "marketable limit"
# order — a limit order priced this many percent worse than the last
# known price, in the direction that guarantees a fill under normal
# liquidity, but which CANNOT fill beyond this bound no matter how thin
# the order book is. This trades a small chance of a delayed close
# (extremely rare on liquid pairs) for eliminating the catastrophic-fill
# scenario entirely, by construction rather than just detecting it after.
MAX_CLOSE_SLIPPAGE_PCT = 3.0
BOUNDED_CLOSE_WAIT_SECONDS = 10

# Logic B safety net: if the exchange-side bracket sync has been failing
# continuously for this many seconds (e.g. persistent CloudFront 403s
# blocking the lookup needed to update it — observed in practice lasting
# 30+ minutes straight), and price crosses the LOCALLY-intended (tighter)
# stop in the meantime, close the position directly via our own market
# order instead of waiting indefinitely for the exchange sync to succeed.
# The OLD, wider bracket level is still live on the exchange the whole
# time as a backstop — this just adds protection at the INTENDED tighter
# level too, closing the gap Logic A already had via its own local
# candle-based stop check (which Logic B never had, since B was designed
# to rely entirely on the exchange-side bracket).
LOGIC_B_LOCAL_FALLBACK_SECONDS = 120

# Sanity check: if a reported fill price differs from the signal price by
# more than this %, treat it as corrupted/wrong exchange data rather than
# a genuine slippage event (real market-order slippage this large in under
# a second is implausible for these liquid pairs) — fall back to the
# signal price instead of computing stop/target from a bad number.
MAX_REASONABLE_SLIPPAGE_PCT = 2.0

# ---- Staircase trailing-stop settings ----
# As price progresses toward target, the stop-loss ratchets up (or down for
# shorts) to lock in the PREVIOUS checkpoint each time a new one is reached.
# No partial position closing — pure stop-loss trailing, full close only at
# hard stop or full target (100%). Stop only ever moves in the favorable
# direction, never loosens.
#   Reach 20% progress -> stop locks at breakeven (0%)
#   Reach 50% progress -> stop locks at the 20% level
#   Reach 75% progress -> stop locks at the 50% level
#   Reach 90% progress -> stop locks at the 75% level
#   Reach 100% (target) -> full close, booked
STAIRCASE_TRIGGERS = [0.20, 0.50, 0.75, 0.90]   # progress levels that trigger a stop move
STAIRCASE_LOCKS =    [0.00, 0.20, 0.50, 0.75]   # where the stop locks to, for each trigger above

STATE_FILE = "trend_live_state.json"
TRADES_LOG = "trend_trades_log.csv"     # Logic A trades
TRADES_LOG_B = "trend_trades_log_B.csv"  # Logic B trades — kept separate

# ================================================================
# LOGIC B — fast trend-following scalper (EMA 5/13 cross + RSI + ATR)
# Runs ALONGSIDE Logic A (the 200EMA+VWAP+CVD strategy above). Only ONE
# trade is ever open at a time system-wide — whichever logic finds a
# valid setup FIRST in a given scan takes the trade (Logic A is checked
# first each loop, then Logic B, if A found nothing that loop).
# ================================================================
# Controls which logic(s) are active — changeable LIVE from the dashboard
# without restarting the algo. Values: "A", "B", or "BOTH".
LOGIC_MODE = {"active": "A"}  # Default changed from "BOTH" to "A" per user
                              # request — even in a worst-case where the
                              # saved state file is somehow completely lost
                              # (e.g. a full Render redeploy that wipes the
                              # ephemeral disk), the bot now falls back to
                              # Logic A only, not silently to BOTH. Combined
                              # with the state-persistence fix above, this
                              # is now protected on two independent levels.

# "crossover" = only signal on a FRESH EMA cross (fewer, more selective entries)
# "trend"     = signal on EVERY loop where price/EMAs stay aligned with the
#               trend and RSI confirms momentum (matches the reference bot's
#               DEFAULT mode — far more frequent, continuously re-enters as
#               long as the trend holds, not just at the moment it starts)
STRATEGY_MODE_B = "trend"

EMA_FAST = 5
EMA_SLOW = 13
RSI_PERIOD = 14
ATR_PERIOD = 14
RSI_LONG_MAX = 80
RSI_LONG_MIN = 50
RSI_SHORT_MIN = 20
RSI_SHORT_MAX = 50
SL_ATR_MULT = 1.0
TP_ATR_MULT = 1.5
LOGIC_B_RISK_PER_TRADE_PCT = 2.0
LOGIC_B_LEVERAGE = 5
MIN_STOP_DIST_PCT_B = 0.30   # skip trades where ATR-based stop distance is
                              # tinier than this — prevents forced max-leverage
                              # sizing that can trigger exchange liquidation rejects

# ---- Bracket order + ATR-expansion trailing stop (Logic B only) ----
# Matches the reference bot's design: entry uses a native Delta "bracket
# order" (SL+TP attached in one call) instead of our own polling-based
# check. TP is forced to a fixed risk:reward ratio off the SL distance
# (not independently ATR-sized), and the stop trails once volatility
# (ATR) expands significantly from what it was at entry.
TP_RR_MULT = 2.0              # take-profit = SL distance * this (fixed R:R)
BRACKET_WIDEN_PCT = 0.15      # % to widen SL/TP by if exchange rejects as "immediate execution"
BRACKET_MAX_RETRIES = 3
TRAIL_MULT_B = 1.5            # trailing stop = TRAIL_MULT_B * ATR behind the best price
                                # (matches reference bot's TRAIL_ATR_MULT default of 1.5 —
                                # we previously had this at 1.0, which trailed tighter than intended)
ATR_EXPANSION_TRIGGER_PCT = 20  # ATR must expand this much (%) vs entry-time ATR to trail

MAX_DAILY_LOSS_PCT = 5.0   # circuit breaker — applies across BOTH logics combined

# Absolute safety net BEYOND the daily circuit breaker: the daily breaker
# only ever looks at loss WITHIN a single UTC day (it resets every day, so
# several days of losses just under 5% each would never trip it, even
# though the account could have quietly shrunk a lot over time). This is
# a second, independent floor based on the account's balance the very
# FIRST time this bot ever ran — if the account ever drops below this
# percentage of that original balance, ALL new trades (both logics) are
# blocked until manually reset, regardless of what day it is.
MIN_BALANCE_FLOOR_PCT = 50.0
COOLDOWN_SECONDS = 10      # min gap after any close before a new entry
                            # (note: our loop only checks once every
                            # LOOP_INTERVAL_SECONDS=60s anyway, so a 10s
                            # cooldown has no extra effect beyond that —
                            # kept here for transparency/config completeness)
FIXED_SIZE = 1             # ---- CHANGED for real-account initial verification ----
                           # 1 = hamesha exactly 1 lot, risk-based-sizing formula
                           # bypass karke. Maksad: real-account pe execution-mechanics
                           # (order-placement, bracket-attach, close) ko chhote,
                           # controlled risk (~$0.06-$0.19 per trade) ke saath verify
                           # karna, sizing-formula ko ek saath test kiye bina. Jab
                           # confidence build ho jaaye, ise wapas 0 par set karke
                           # risk-based-sizing on karo.
                           # (Fixed-1-lot mode was tried and reverted per user
                           # request — code support for it is left in place
                           # below in case it's wanted again later.)
# ================================================================


client = DeltaRestClient(base_url=config.BASE_URL, api_key=config.API_KEY, api_secret=config.API_SECRET)


# ============================================================
# Basic helpers
# ============================================================

TRADE_LOG_COLUMNS = [
    "time", "symbol", "action", "side", "size", "entry_price", "exit_price",
    "stop", "target", "reason", "approx_gross_pnl_pct",
    "approx_net_pnl_pct_after_fees", "gross_pnl_amount", "fees_amount",
    "net_pnl_amount", "fill_method", "order_response", "strategy",
]

# Known contract-values for the symbols this bot trades — used to convert
# %-based P&L into actual dollar amounts for the trades log. These are
# stable, well-known constants (confirmed repeatedly via the exchange's
# own "Setting leverage..." startup log each run), not fetched live here
# to keep log_trade_event() simple and independent of any API call.
CONTRACT_VALUES = {
    "BTCUSD": 0.001, "ETHUSD": 0.01, "XRPUSD": 1, "SOLUSD": 1,
    "DOGEUSD": 1, "ADAUSD": 1,
}


def estimate_net_pnl_pct(gross_pnl_pct, strategy):
    """
    Rough fee-adjusted P&L estimate for REPORTING/AWARENESS only — this does
    NOT filter or block any trade, it just subtracts an assumed round-trip
    fee from the gross P&L so the trades log shows both numbers side by
    side. This lets us judge, from real trade history, how much fees are
    actually eating into Logic B's results before deciding whether to
    change anything about its entries.

    Assumption used (approximate, not exact per-trade):
      - Logic A: assumed maker fee both legs (its normal entry path is
        LIMIT-first) -> ROUND_TRIP_FEE_PCT. If a trade actually fell back
        to a market fill on one or both legs, real fees were higher than
        this estimate — this is a best-case approximation, not exact.
      - Logic B: ALWAYS uses market/bracket orders (taker fee) on both
        entry and exit -> taker round-trip (TAKER_FEE_PCT * 2). This one
        should be fairly accurate since Logic B's execution path is fixed.
    """
    if strategy == "B":
        round_trip_fee_pct = TAKER_FEE_PCT * 2
    else:
        round_trip_fee_pct = ROUND_TRIP_FEE_PCT
    return gross_pnl_pct - round_trip_fee_pct


def log_trade_event(**fields):
    """
    Writes a trade log row using a FIXED, consistent set of columns every
    time (filling missing ones with empty string), regardless of which
    action type (OPEN/PARTIAL_CLOSE/CLOSE) is being logged.

    Routes to TRADES_LOG (Logic A) or TRADES_LOG_B (Logic B) based on the
    "strategy" field ("A" or "B") — kept as separate files/spreadsheets
    per the user's request, so each strategy's performance can be judged
    independently.

    IMPORTANT: without the fixed-column approach, appending rows with
    different column sets to the same CSV via pandas causes column
    MISALIGNMENT (values silently shift into the wrong columns) since the
    file's header is only written once, from whichever row happened to
    be first — a real bug this fixes.

    ---- NEW: also computes actual DOLLAR-amount P&L and fees (not just
    %), using entry_price * size * contract_value as the notional. This
    is computed here centrally (rather than at every call site) so every
    CLOSE event automatically gets both percentage and dollar figures. ----
    """
    row = {col: fields.get(col, "") for col in TRADE_LOG_COLUMNS}

    gross_pct = fields.get("approx_gross_pnl_pct")
    net_pct = fields.get("approx_net_pnl_pct_after_fees")
    entry_price = fields.get("entry_price")
    size = fields.get("size")
    symbol = fields.get("symbol")

    # ---- BUG FIX: prefer REAL exit price / REAL fee from Delta's own
    # /v2/fills endpoint (see get_real_fill_and_fee()) over the old
    # candle-price proxy and flat-%-estimate fee, whenever the caller
    # managed to fetch them. This is what actually corrects the two
    # mismatches found by comparing this log against Delta's exported
    # Trade-History/Wallet-History CSVs (wrong exit price on bracket-
    # closed trades; estimated-not-real fees on every trade). If the
    # caller couldn't fetch real data (e.g. CloudFront blocked the
    # lookup), real_exit_price/real_fee_amount are simply absent here
    # and everything falls back to the exact old estimate-based
    # behavior — a close is never blocked or left un-logged over this.
    real_exit_price = fields.get("real_exit_price")
    real_fee_amount = fields.get("real_fee_amount")
    if real_exit_price is not None and entry_price:
        try:
            entry_f = float(entry_price)
            real_exit_f = float(real_exit_price)
            was_long = fields.get("side") == "sell"  # close-side "sell" means the
                                                       # position being closed was long
            if was_long:
                gross_pct = (real_exit_f - entry_f) / entry_f * 100
            else:
                gross_pct = (entry_f - real_exit_f) / entry_f * 100
            row["exit_price"] = real_exit_f
            row["approx_gross_pnl_pct"] = round(gross_pct, 4)
        except (TypeError, ValueError):
            pass  # malformed real data — keep whatever the caller originally passed

    if gross_pct is not None and entry_price and size and symbol:
        cv = CONTRACT_VALUES.get(symbol, 1)
        try:
            notional = float(entry_price) * float(size) * cv
            gross_amount = float(gross_pct) / 100 * notional
            if real_fee_amount is not None:
                fees_amount = float(real_fee_amount)
                net_amount = gross_amount - fees_amount
                net_pct = (net_amount / notional * 100) if notional else net_pct
                row["approx_net_pnl_pct_after_fees"] = round(net_pct, 4) if net_pct is not None else ""
            else:
                net_amount = (float(net_pct) / 100 * notional) if net_pct is not None else gross_amount
                fees_amount = gross_amount - net_amount
            row["gross_pnl_amount"] = round(gross_amount, 4)
            row["net_pnl_amount"] = round(net_amount, 4)
            row["fees_amount"] = round(fees_amount, 4)
        except (TypeError, ValueError):
            pass  # leave the $ columns blank if inputs are malformed

    strategy = fields.get("strategy", "A")
    if strategy == "B":
        append_csv(TRADES_LOG_B, row, fallback_key="log_b")
    else:
        append_csv(TRADES_LOG, row, fallback_key="log")


def append_csv(path, row_dict, fallback_key="log"):
    target = path
    if _using_fallback[fallback_key]:
        target = os.path.join(FALLBACK_DIR, os.path.basename(path))

    df = pd.DataFrame([row_dict])
    for attempt in range(3):
        try:
            header = not os.path.exists(target)
            df.to_csv(target, mode="a", header=header, index=False)
            return
        except OSError as e:
            if attempt < 2:
                time.sleep(1)
            else:
                if not _using_fallback[fallback_key]:
                    print(f"  [WARN] Can't write to {path} ({e}). "
                          f"Switching permanently to fallback location: {FALLBACK_DIR}")
                    _using_fallback[fallback_key] = True
                    return append_csv(path, row_dict, fallback_key)
                else:
                    print(f"  [WARN] Even fallback CSV write failed ({e}) — "
                          f"this trade record may be lost, continuing anyway.")


def load_state():
    fallback_path = os.path.join(FALLBACK_DIR, os.path.basename(STATE_FILE))
    if _using_fallback["state"] and os.path.exists(fallback_path):
        with open(fallback_path) as f:
            return json.load(f)
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    if os.path.exists(fallback_path):
        _using_fallback["state"] = True
        with open(fallback_path) as f:
            return json.load(f)
    return {"position": None}


import tempfile

# Fallback location if the script's own folder has persistent write
# restrictions (some Windows setups block writes to Desktop/Documents
# even after retries — antivirus, Controlled Folder Access, sync-locked
# folders, etc.). This is always writable.
FALLBACK_DIR = os.path.join(tempfile.gettempdir(), "trend_scalp_algo_data")
os.makedirs(FALLBACK_DIR, exist_ok=True)
_using_fallback = {"state": False, "log": False, "log_b": False}


def _safe_write_json(primary_path, fallback_key, data):
    """
    Tries the primary path first (atomic write via temp-file-then-rename,
    which holds the file handle for less time and avoids some lock issues).
    Falls back permanently to a temp-dir copy if the primary path keeps
    failing, so state is never silently lost.
    """
    target = primary_path
    if _using_fallback[fallback_key]:
        target = os.path.join(FALLBACK_DIR, os.path.basename(primary_path))

    for attempt in range(3):
        try:
            tmp_path = target + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, target)  # atomic on both Windows and Linux
            return
        except OSError as e:
            if attempt < 2:
                time.sleep(1)
            else:
                if not _using_fallback[fallback_key]:
                    print(f"  [WARN] Can't write to {primary_path} ({e}). "
                          f"Switching permanently to fallback location: {FALLBACK_DIR}")
                    _using_fallback[fallback_key] = True
                    return _safe_write_json(primary_path, fallback_key, data)
                else:
                    print(f"  [WARN] Even fallback write failed ({e}) — "
                          f"state not saved this cycle, continuing anyway.")


def save_state(state):
    _safe_write_json(STATE_FILE, "state", state)


def get_product_map():
    """Testnet intentionally — product IDs must match the account we place
    orders on (production and testnet product_ids can differ per symbol)."""
    r = requests.get(f"{config.BASE_URL}/v2/products", timeout=15)
    r.raise_for_status()
    products = r.json()["result"]
    return {p["symbol"]: p for p in products if p["symbol"] in SYMBOLS_TO_WATCH}


def set_leverage_for_all(products):
    """
    Actually SETS the leverage on the exchange for each product — without
    this, the LEVERAGE variable was only used in our own math, but never
    told to Delta, so actual account leverage could silently differ from
    what our sizing calculation assumed.
    """
    print("\n  Contract values (used in lot-size calculations):")
    for sym, product in products.items():
        cv = product.get("contract_value", "unknown")
        print(f"    {sym}: contract_value = {cv}")

    for sym, product in products.items():
        try:
            resp = client.set_leverage(product["id"], LEVERAGE)
            print(f"  Leverage set for {sym}: {LEVERAGE}x -> {resp}")
        except Exception as e:
            print(f"  [WARN] Could not set leverage for {sym}: {e}")


def get_liquidation_distance_pct(entry_price, stop_price, side, leverage):
    """
    Rough estimate of how far liquidation is from entry, as a sanity check
    that our stop-loss triggers LONG before liquidation would ever be a risk.
    This is an approximation (actual liquidation price depends on Delta's
    exact maintenance margin formula, which can vary by product) — treat it
    as a safety sanity-check, not an exact number.
    """
    approx_liq_distance_pct = (1 / leverage) * 100 * 0.9  # rough, conservative estimate
    stop_distance_pct = (abs(entry_price - stop_price) / entry_price) * 100
    return approx_liq_distance_pct, stop_distance_pct


def fetch_candles(symbol, hours=LOOKBACK_HOURS, resolution="1m"):
    # Uses REAL_DATA_BASE_URL (production exchange), not testnet — testnet's
    # price feed is thin/simulated. Read-only public data, no key needed.
    end = int(datetime.now(timezone.utc).timestamp())
    start = end - hours * 3600
    url = f"{config.REAL_DATA_BASE_URL}/v2/history/candles"
    params = {"symbol": symbol, "resolution": resolution, "start": start, "end": end}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json().get("result", [])
    if not data:
        return None
    df = pd.DataFrame(data).rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df.sort_values("timestamp").reset_index(drop=True)


def get_wallet_available_balance():
    # ---- BUG FIX: filter by asset_symbol == 'USD' explicitly ----
    # Delta's /v2/wallet/balances returns ONE entry PER ASSET the account
    # holds (this account trades USD-margined perpetuals, but could also
    # have leftover/dust balances in other assets — e.g. from a testnet
    # faucet claim, or a different settling asset). The old code just
    # grabbed the FIRST entry in the list with a positive value, with NO
    # check on which asset it belonged to — meaning if a non-USD entry
    # happened to sit earlier in the array (order isn't guaranteed), this
    # would silently return the WRONG number, since every $-based
    # calculation in this script (position sizing, circuit breakers, the
    # dashboard) assumes this is the USD balance.
    response = client.request("GET", "/v2/wallet/balances", auth=True)
    balances = response.json().get("result", [])
    for b in balances:
        if b.get("asset_symbol") == "USD":
            return float(b.get("available_balance", 0))
    print("  [WARN] No 'USD' entry found in wallet balances — returning 0.0 "
          "(check /v2/wallet/balances response; asset_symbol may differ).")
    return 0.0


def get_wallet_total_balance():
    """
    Returns TOTAL account balance (not "available_balance") — this is what
    the daily circuit-breaker must use. "available_balance" drops the
    moment a new position opens (since margin gets locked/reserved for
    it), which would make the breaker misread a normal margin-lock as a
    huge instant "loss" even though no money was actually lost. Total
    balance only changes with genuine realized P&L, not margin allocation.
    """
    # ---- BUG FIX: filter by asset_symbol == 'USD' explicitly ----
    # Same fix as get_wallet_available_balance() above — see its comment
    # for the full story. This function had the identical bug.
    response = client.request("GET", "/v2/wallet/balances", auth=True)
    balances = response.json().get("result", [])
    for b in balances:
        if b.get("asset_symbol") == "USD":
            total = b.get("balance")
            if total is not None:
                return float(total)
            # Fallback: if "balance" field isn't present for some reason,
            # fall back to available_balance rather than crashing.
            print("  [WARN] USD wallet entry had no 'balance' field — falling back to "
                  "available_balance for circuit-breaker (may misread margin-lock as loss).")
            return float(b.get("available_balance", 0))
    print("  [WARN] No 'USD' entry found in wallet balances — returning 0.0 "
          "(check /v2/wallet/balances response; asset_symbol may differ).")
    return 0.0


# ============================================================
# Order placement (FIXED response parsing + reduce_only support)
# ============================================================

def get_authoritative_entry_price(product_id, retries=5, delay=1.0):
    """
    After placing a market entry order, polls the exchange's own position
    endpoint to find the REAL entry price AND size it recorded — this is
    the single source of truth for what actually happened, more reliable
    than parsing the order response or guessing. Returns (price, size),
    or (None, None) if no position shows up in time (caller should fall
    back to its own tracking in that case).

    Checking size here too (not just price) catches a related issue:
    if this symbol already had a small pre-existing residual position
    (e.g. leftover from an earlier partial-close), Delta nets same-
    direction fills together automatically — so the REAL resulting size
    can differ from what we just requested. Observed in practice: we
    requested/expected 47 lots, but the real resulting position was only
    43 lots. The bracket protection isn't affected either way (Delta
    brackets auto-cover whatever the current real size is), but our own
    LOCAL size tracking (used for P&L math and for sizing later reduce-
    only close orders) needs the real number to stay accurate.
    """
    for _ in range(retries):
        time.sleep(delay)
        try:
            pos = client.get_position(product_id)
        except Exception:
            continue
        size = float(pos.get("size", 0)) if isinstance(pos, dict) else 0
        if size != 0:
            entry_price = pos.get("entry_price")
            if entry_price is not None:
                return float(entry_price), abs(size)
    return None, None


def _extract_fill_price(response):
    """
    Tries to find the actual average fill price from an order response.

    IMPORTANT: 'limit_price' is deliberately NOT in this list. It used to
    be, as a defensive fallback — but that caused a real, serious bug:
    for MARKET orders, 'limit_price' in Delta's response is just a
    meaningless placeholder (observed in practice: Delta echoes back
    something like 0.1 for market sell orders, visible in the exported
    order-history CSV's "Order Price" column too). When the real fill
    fields weren't found, this function was grabbing that placeholder
    and treating it as the actual fill price — producing a fabricated
    "-100% loss" trade-log entry (entry ~63781, "exit" 0.1) that never
    actually happened. If none of the REAL fill-price fields below are
    present, return None and let the caller fall back to its own
    reference price (e.g. the recent candle close) instead.
    """
    if not isinstance(response, dict):
        return None
    for key in ("average_fill_price", "avg_fill_price", "price"):
        val = response.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    result = response.get("result")
    if isinstance(result, dict):
        return _extract_fill_price(result)
    return None


def _extract_order_id(response):
    """
    The SDK returns the order dict directly (not nested under "result").
    Handles both possible shapes defensively.
    """
    if isinstance(response, dict):
        if "id" in response:
            return response["id"]
        if "result" in response and isinstance(response["result"], dict):
            return response["result"].get("id")
    return None


def get_real_fill_and_fee(product_id, retries=6, delay=1.5):
    """
    ---- BUG FIX (found via comparing dashboard trade log against Delta's
    own exported Trade-History / Wallet-History CSVs): ----

    Two separate accuracy problems this fixes, both by pulling the
    exchange's own /v2/fills record (the SAME data source Delta's app
    uses to generate the Trade-History CSV export) right after a
    position closes:

      1. EXIT PRICE proxy bug: a few close paths (exchange-side bracket
         silently closing the position, or the exchange saying
         "no_position_for_reduce_only" before our own close order landed)
         had no real fill to read from OUR order response, so they fell
         back to using the latest CANDLE close price as a stand-in exit
         price. Verified in practice: one BTCUSD bracket-close was logged
         at 64643.0, but the REAL exchange fill was 64601.50 — a $41.5
         difference, because the candle price moves in the ~20-40s gap
         before this script notices the exchange-side close on the next
         loop. Using the real fill removes this gap entirely.

      2. FEE estimate bug: even for NORMAL closes with a correct exit
         price, this script's dashboard $-fee figures came from a flat
         ASSUMED round-trip fee % (estimate_net_pnl_pct — maker*2 for
         Logic A, taker*2 for Logic B), not Delta's real per-fill fee.
         Verified in practice: real fees varied fill-to-fill (maker/taker
         mix isn't always what's assumed), causing the dashboard's Net
         P&L to be off by anywhere from $0.06 to over $2 per trade.

    Returns (real_exit_price, real_total_round_trip_fee) using the most
    recent 2 fills for this product (this bot only ever holds ONE
    position at a time, so the last 2 fills on a given product ARE this
    trade's entry + exit) — or (None, None) if the fills endpoint can't
    be reached or doesn't have fresh data yet in time (e.g. the same
    CloudFront 403 issue that affects other API calls on this testnet).
    Callers MUST fall back to their old proxy/estimate behaviour if this
    returns (None, None) — a close should never go un-logged just
    because this extra lookup failed.
    """
    for attempt in range(retries):
        try:
            response = client.fills({"product_ids": str(product_id)}, page_size=2)
            if isinstance(response, dict):
                result = response.get("result", [])
            elif isinstance(response, list):
                result = response
            else:
                result = []
            if isinstance(result, list) and len(result) >= 2:
                # Delta's /v2/fills returns newest-first.
                exit_fill, entry_fill = result[0], result[1]
                exit_price = exit_fill.get("price")
                exit_fee = exit_fill.get("commission")
                entry_fee = entry_fill.get("commission")
                if exit_price is not None and exit_fee is not None and entry_fee is not None:
                    return float(exit_price), float(exit_fee) + float(entry_fee)
        except Exception as e:
            print(f"  [WARN] get_real_fill_and_fee: couldn't fetch real fill/fee "
                  f"data from /v2/fills on attempt {attempt + 1}/{retries}: {e}")
        time.sleep(delay)
    print(f"  [WARN] get_real_fill_and_fee: gave up after {retries} attempts — "
          f"falling back to the old estimated exit-price/fee for this close.")
    return None, None


def _order_is_still_open(product_id, order_id):
    """
    FIXED: get_live_orders() returns a LIST of order dicts directly, not a
    dict with a "result" key. Iterate it directly.
    """
    try:
        live_orders = client.get_live_orders({"product_id": product_id})
        if isinstance(live_orders, dict):
            live_orders = live_orders.get("result", [])
        return any(o.get("id") == order_id for o in live_orders)
    except Exception as e:
        print(f"  [WARN] couldn't check order status: {e}")
        return None  # unknown — caller should treat cautiously


def place_market_order_direct(product_id, side, size, reduce_only=False):
    """
    Places a MARKET order immediately — no limit-order wait, no timeout.
    Used by Logic B, which is designed for instant execution (fast
    scalping) rather than Logic A's cheaper-but-slower limit-first approach.
    """
    order = {"product_id": product_id, "size": size, "side": side, "order_type": "market_order"}
    if reduce_only:
        order["reduce_only"] = True
    print(f"  Placing MARKET order (instant, Logic B): {order}")
    response = client.create_order(order)
    return response, "market_direct"


def get_current_ticker_price(symbol):
    """Public endpoint — current mark/last price, used when widening a
    rejected bracket order (need fresh price to nudge levels away from)."""
    r = requests.get(f"{config.REAL_DATA_BASE_URL}/v2/tickers/{symbol}", timeout=10)
    r.raise_for_status()
    data = r.json().get("result", {})
    return float(data.get("close") or data.get("mark_price") or data.get("spot_price"))


def get_exec_price(symbol, real_market_price):
    """
    The real-market price (config.REAL_DATA_BASE_URL) decides the trade
    DIRECTION and the ATR (volatility) used for SL/TP distance. But the
    actual order/position lives on the TESTNET account (config.BASE_URL),
    which can sit at a genuinely different price than the real market
    (this is a real, structural difference between the two feeds — not
    corrupted data, as we mistakenly assumed earlier in this project).
    So before placing an entry, we fetch the testnet account's OWN current
    price and anchor the initial SL/TP/sizing estimate to it instead —
    reducing the gap between our pre-trade estimate and the actual fill,
    matching the reference bot's get_exec_price() design.
    Falls back to real_market_price if the testnet ticker fetch fails.
    """
    try:
        r = requests.get(f"{config.BASE_URL}/v2/tickers/{symbol}", timeout=10)
        r.raise_for_status()
        data = r.json().get("result", {})
        testnet_price = float(data.get("close") or data.get("mark_price") or data.get("spot_price"))
        return testnet_price
    except Exception as e:
        print(f"    [WARN] Couldn't fetch testnet's own price for {symbol} ({e}) — "
              f"using real-market price instead for the initial estimate.")
        return real_market_price


def place_bracket_order_raw(product_id, product_symbol, side, size, sl_price, tp_price,
                             trigger_method="last_traded_price"):
    """
    Attaches a stop-loss + take-profit bracket to an EXISTING position via
    Delta's native /v2/orders/bracket endpoint. Does NOT open a position —
    must be called right after a market entry.
    """
    body = {
        "product_id": product_id,
        "product_symbol": product_symbol,
        "stop_loss_order": {"order_type": "market_order", "stop_price": str(round(sl_price, 6))},
        "take_profit_order": {"order_type": "market_order", "stop_price": str(round(tp_price, 6))},
        "bracket_stop_trigger_method": trigger_method,
    }
    return client.request("POST", "/v2/orders/bracket", body, auth=True)


def get_bracket_stop_loss_order_id(product_id, retries=3, delay=2):
    """
    Finds the ACTUAL stop-loss leg order that Delta created when the
    bracket was attached (a separate resting/pending order, distinct from
    our original MARKET entry order — which fills instantly and moves to
    'closed' state, so its id can't be used to edit the bracket: Delta
    rejects that with 'open_order_not_found'). Returns the order id of
    the open/pending stop_loss_order for this product, or None if not found.

    Retries a few times with a short delay on failure — this call was
    observed hitting transient CloudFront-level 403s (generic "too much
    traffic" CDN errors, not a Delta-specific auth/permission error),
    which a short retry can ride out rather than immediately falling back
    to the more disruptive cancel+recreate path below.
    """
    last_error = None
    for attempt in range(retries):
        try:
            r = client.request("GET", "/v2/orders", {"product_ids": str(product_id),
                                                       "states": "open,pending"}, auth=True)
            orders = r.json().get("result", [])
            for o in orders:
                if o.get("stop_order_type") == "stop_loss_order":
                    return o.get("id")
            return None  # request succeeded, just no matching order found
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(delay)
    print(f"  [WARN] Couldn't fetch open orders to find bracket's stop-loss leg "
          f"after {retries} attempts: {last_error}")
    return None


def get_bracket_sl_tp_prices(product_id, retries=5, delay=3):
    """
    Fetches BOTH legs of an existing bracket order (if any) for this
    product — the stop-loss trigger price and the take-profit trigger
    price. Used after a restart to fingerprint which strategy (A or B)
    an adopted, previously-untracked position most likely belongs to:
    Logic A always uses a 2.5x reward:risk ratio (RISK_REWARD_MULT) and
    Logic B always uses a fixed 2.0x ratio (TP_RR_MULT) — different
    enough to distinguish reliably from the bracket's actual levels,
    without needing any of the local state that a restart just wiped.
    Returns (sl_price, tp_price), each None if that leg wasn't found.

    Retries several times with backoff on failure — this specific lookup
    was observed hitting CloudFront-level 403s that outlasted a shorter
    3-attempt retry (used elsewhere for the more frequent trailing-stop
    lookup). It's worth waiting longer here specifically: a WRONG guess
    here doesn't just fail once, it silently sticks for the entire rest
    of the trade (once a strategy is assigned, later restarts just keep
    it as-is rather than re-checking) — so a slower, more persistent
    retry is worth the extra delay during this one-time startup check.
    """
    last_error = None
    for attempt in range(retries):
        try:
            r = client.request("GET", "/v2/orders", {"product_ids": str(product_id),
                                                       "states": "open,pending"}, auth=True)
            orders = r.json().get("result", [])
            sl_price = tp_price = None
            for o in orders:
                otype = o.get("stop_order_type")
                trigger = o.get("stop_price")
                if trigger is None:
                    continue
                if otype == "stop_loss_order":
                    sl_price = float(trigger)
                elif otype == "take_profit_order":
                    tp_price = float(trigger)
            return sl_price, tp_price
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))  # 3s, 6s, 9s, 12s — backs off
    print(f"  [WARN] Couldn't fetch open orders to read bracket SL/TP levels "
          f"after {retries} attempts: {last_error}")
    return None, None


def edit_bracket_order(order_id, product_id, product_symbol, sl_price, tp_price,
                        trigger_method="last_traded_price"):
    """
    Modifies an EXISTING bracket's SL/TP levels via Delta's PUT
    /v2/orders/bracket endpoint. This is the CORRECT way to update a
    trailing stop — Delta only allows ONE bracket per open position, so
    cancelling all orders and POSTing a fresh bracket (the old approach)
    fails with 'bracket_order_exists' since the existing bracket is still
    attached to the position and wasn't actually removed by a plain order
    cancel.

    order_id MUST be the bracket's own resting stop-loss leg order id
    (fetched via get_bracket_stop_loss_order_id) — NOT the original entry
    order's id. Confirmed against Delta's own support forum: the entry
    order becomes 'closed' the instant it fills, so PUT-ing against it
    is rejected as 'open order not found'. Also per Delta's docs: size
    doesn't need to be (and can't be) specified here — an existing
    bracket automatically covers however much of the position is
    currently open, so this call updates SL/TP levels only.
    """
    body = {
        "id": order_id,
        "product_id": product_id,
        "product_symbol": product_symbol,
        "bracket_stop_loss_price": str(round(sl_price, 6)),
        "bracket_take_profit_price": str(round(tp_price, 6)),
        "bracket_stop_trigger_method": trigger_method,
    }
    return client.request("PUT", "/v2/orders/bracket", body, auth=True)


def place_bracket_with_retry(product_id, product_symbol, side, size, sl_price, tp_price,
                              direction, trigger_method="last_traded_price",
                              max_retries=BRACKET_MAX_RETRIES, widen_pct=BRACKET_WIDEN_PCT):
    """
    Same as place_bracket_order_raw, but recovers from Delta's
    'bracket_order_immediate_execution' rejection (happens when price has
    already moved past the SL/TP level by the time the bracket is placed —
    common with a lagging/jumpy testnet feed). Widens both levels further
    from the live price and retries, up to max_retries times.
    """
    attempt = 0
    while True:
        try:
            return place_bracket_order_raw(
                product_id, product_symbol, side, size, sl_price, tp_price, trigger_method)
        except Exception as e:
            if "bracket_order_immediate_execution" not in str(e) or attempt >= max_retries:
                raise
            attempt += 1
            try:
                current_price = get_current_ticker_price(product_symbol)
            except Exception:
                current_price = sl_price
            widen = current_price * (widen_pct / 100.0)
            if direction == "long":
                sl_price -= widen
                tp_price += widen
            else:
                sl_price += widen
                tp_price -= widen
            print(f"    [WARN] Bracket rejected as immediate-execution, widening and "
                  f"retrying (attempt {attempt}/{max_retries}): sl={sl_price:.6f} tp={tp_price:.6f}")
            time.sleep(0.5)


def place_order_with_fallback(product_id, side, size, limit_price, reduce_only=False):
    """
    Places a LIMIT order first (cheaper maker fee). If not filled within
    LIMIT_ORDER_TIMEOUT_SECONDS, cancels it and places a MARKET order
    instead. reduce_only=True must be used for CLOSING/reducing a position
    so the exchange doesn't treat it as a fresh trade requiring new margin.
    Used by Logic A (limit-first philosophy — cheaper fees, willing to wait).
    """
    order = {
        "product_id": product_id, "size": size, "side": side,
        "order_type": "limit_order", "limit_price": str(round(limit_price, 6)),
    }
    if reduce_only:
        order["reduce_only"] = True

    print(f"  Placing LIMIT order: {order}")
    response = client.create_order(order)
    order_id = _extract_order_id(response)

    if order_id is None:
        print(f"  [WARN] couldn't read order id from response ({response}) — "
              f"treating as unfilled, will fall back to market order.")

    waited = 0
    filled = False
    while waited < LIMIT_ORDER_TIMEOUT_SECONDS:
        time.sleep(5)
        waited += 5
        if order_id is None:
            continue
        still_open = _order_is_still_open(product_id, order_id)
        if still_open is False:
            print(f"  Limit order filled after {waited}s.")
            filled = True
            break

    if filled:
        return response, "limit"

    print(f"  Limit order not filled after {LIMIT_ORDER_TIMEOUT_SECONDS}s — "
          f"cancelling and using market order.")
    if order_id is not None:
        try:
            client.cancel_order(product_id, order_id)
        except Exception as e:
            print(f"  [WARN] cancel failed (may have filled in the meantime): {e}")

        # ---- Verify the cancel actually took effect (BUG FIX) ----
        # cancel_order() not raising an exception doesn't guarantee Delta
        # actually removed the order — this was observed in practice:
        # repeated close attempts each left a NEW stray reduce-only limit
        # order resting on the exchange (multiple showed up simultaneously
        # in "Open Orders"), while the market-order fallback kept failing
        # with "out_of_bankruptcy" — almost certainly because these
        # accumulating stray reduce-only orders confused Delta's margin/
        # bankruptcy-price calculation for the position. Explicitly
        # verify and retry the cancel a couple of times before moving on.
        for _ in range(3):
            still_open = _order_is_still_open(product_id, order_id)
            if still_open is not True:
                break
            print(f"  [WARN] Limit order {order_id} still shows as open after "
                  f"cancel — retrying cancel.")
            time.sleep(2)
            try:
                client.cancel_order(product_id, order_id)
            except Exception as e:
                print(f"  [WARN] retry cancel failed: {e}")

    # ---- Sweep ANY other stray orders left on this product (BUG FIX) ----
    # Defensive cleanup regardless of whether the single cancel above
    # looked successful — earlier failed close attempts (e.g. from a
    # previous loop iteration hitting a transient error) can leave
    # additional stray reduce-only orders behind that the single
    # cancel_order() call above wouldn't touch. A hard sweep here avoids
    # letting them pile up and interfere with the market fallback below.
    try:
        cancel_all_orders_for_product(product_id)
    except Exception as e:
        print(f"  [WARN] Sweep of stray orders on product {product_id} failed "
              f"({e}) — continuing anyway, market order may still fail if "
              f"stray orders remain.")

    # ---- Bounded marketable-limit fallback (BUG FIX — see
    # MAX_CLOSE_SLIPPAGE_PCT above for why this replaced a plain market
    # order here). Priced worse than the reference price by up to
    # MAX_CLOSE_SLIPPAGE_PCT, in whichever direction guarantees a fill
    # under normal liquidity — but the exchange can never fill it beyond
    # that bound, unlike a true market order. ----
    bounded_price = (limit_price * (1 - MAX_CLOSE_SLIPPAGE_PCT / 100) if side == "sell"
                     else limit_price * (1 + MAX_CLOSE_SLIPPAGE_PCT / 100))
    bounded_order = {
        "product_id": product_id, "size": size, "side": side,
        "order_type": "limit_order", "limit_price": str(round(bounded_price, 6)),
    }
    if reduce_only:
        bounded_order["reduce_only"] = True

    print(f"  Placing bounded marketable-limit order (max {MAX_CLOSE_SLIPPAGE_PCT}% "
          f"slippage from {limit_price:.6f}): {bounded_order}")
    response = client.create_order(bounded_order)
    order_id = _extract_order_id(response)

    waited = 0
    while order_id is not None and waited < BOUNDED_CLOSE_WAIT_SECONDS:
        time.sleep(2)
        waited += 2
        if _order_is_still_open(product_id, order_id) is False:
            print(f"  Bounded marketable-limit order filled after {waited}s.")
            return response, "market_fallback"

    if order_id is not None:
        try:
            client.cancel_order(product_id, order_id)
        except Exception as e:
            print(f"  [WARN] cancel of bounded fallback order failed: {e}")

    # ---- Absolute last resort: true unbounded market order. Reaching
    # this point means even a {MAX_CLOSE_SLIPPAGE_PCT}%-bounded order
    # couldn't fill — genuinely abnormal, so this is flagged loudly by
    # the abnormal-fill check that runs after every close regardless. ----
    print(f"  [WARN] Bounded marketable-limit order didn't fill within "
          f"{BOUNDED_CLOSE_WAIT_SECONDS}s — this is unusual. Falling back to a true "
          f"unbounded market order as an absolute last resort.")
    market_order = {"product_id": product_id, "size": size, "side": side, "order_type": "market_order"}
    if reduce_only:
        market_order["reduce_only"] = True
    response = client.create_order(market_order)
    return response, "market_fallback"


# ============================================================
# Startup reconciliation — check the REAL exchange state, not just
# the local JSON file, before doing anything else.
# ============================================================

def guess_strategy_for_adopted_position(sym, product_id, entry_price, side):
    """
    After a restart, an untracked-but-real open position could have been
    opened by EITHER Logic A or Logic B (both share the single position
    slot). Guessing wrong matters: each has a different stop/target
    formula and different ongoing management (A's staircase-trailing vs
    B's ATR-expansion trailing + bracket), so misclassifying a real B
    trade as A (or vice versa) would manage it with the wrong logic
    entirely. Since a restart wipes the local state that would have said
    which one it was, this reconstructs a best-effort answer from what's
    still knowable on the exchange itself:

      1. Symbol restriction: Logic B ONLY trades LOGIC_B_SYMBOLS (BTC/ETH).
         Any other symbol can only be a Logic A position — no guessing
         needed there.
      2. For BTC/ETH: if a bracket order is still attached, its actual
         reward:risk ratio fingerprints which strategy set it — Logic A
         always targets RISK_REWARD_MULT (2.5x), Logic B always targets
         TP_RR_MULT (2.0x). Whichever is closer wins.
      3. If neither of the above resolves it (no symbol restriction hit,
         and no bracket found to fingerprint), fall back to the current
         LOGIC_MODE setting: if only one logic is active, it must be that
         one. If both are active and there's truly no other signal, default
         to Logic A but print a loud warning so this can be checked manually.
    """
    if sym not in LOGIC_B_SYMBOLS:
        return "A"

    sl_price, tp_price = get_bracket_sl_tp_prices(product_id)
    if sl_price is not None and tp_price is not None:
        risk = abs(entry_price - sl_price)
        reward = abs(tp_price - entry_price)
        if risk > 0:
            ratio = reward / risk
            dist_to_a = abs(ratio - RISK_REWARD_MULT)
            dist_to_b = abs(ratio - TP_RR_MULT)
            guess = "B" if dist_to_b < dist_to_a else "A"
            print(f"    Strategy fingerprint for {sym}: existing bracket reward:risk "
                  f"ratio is {ratio:.2f} (A targets {RISK_REWARD_MULT}x, B targets "
                  f"{TP_RR_MULT}x) -> guessing Logic {guess}.")
            return guess

    mode = LOGIC_MODE.get("active", "BOTH")
    if mode == "A":
        return "A"
    if mode == "B":
        return "B"
    print(f"    [WARN] Couldn't fingerprint {sym} (no bracket found and both logics "
          f"are active) — defaulting to Logic A as a guess. Please verify manually "
          f"against the exchange's own order history if this matters.")
    return "A"


def reconcile_with_exchange(state, products):
    print("\n--- Startup reconciliation: checking real exchange state ---")

    real_position = None
    for sym, product in products.items():
        try:
            pos = client.get_position(product["id"])
        except Exception as e:
            print(f"  [WARN] couldn't fetch position for {sym}: {e}")
            continue
        size = float(pos.get("size", 0)) if isinstance(pos, dict) else 0
        if size != 0:
            real_position = (sym, product, pos, size)
            print(f"  FOUND real open position on exchange: {sym}, size={size}")
            break

    if real_position is None:
        if state["position"] is not None:
            print("  Local state had a position recorded, but exchange shows none "
                  "(it must have closed already). Clearing local state.")
            state["position"] = None
        else:
            print("  No open position on exchange, and none tracked locally. Clean start.")
    else:
        sym, product, pos, size = real_position

        # ---- BUG FIX: re-verify the size RIGHT NOW, one more time, before
        # actually adopting it. The initial detection loop above could be
        # followed by strategy-fingerprinting, candle fetches, etc. — all
        # of which take real time, during which the exchange's actual
        # position size could change (e.g. if it merges with another
        # trade opened around the same time — Delta automatically nets
        # same-direction positions on the same symbol together). Adopting
        # a STALE, smaller size than what's really open would mean any
        # bracket we attach only covers PART of the real position, leaving
        # the rest silently unprotected. This was observed in practice:
        # reconciliation read size=1.0, but the real, current position was
        # actually 5.0 by the time adoption ran (confirmed via a
        # 'bracket_order_exists' rejection — a bracket already existed for
        # a DIFFERENT size than what we were about to track). ----
        try:
            fresh_pos = client.get_position(product["id"])
            if isinstance(fresh_pos, dict) and fresh_pos.get("size"):
                fresh_size = float(fresh_pos["size"])
                if fresh_size != size:
                    print(f"  [INFO] {sym}: re-verified size just before adopting — "
                          f"was {size}, now {fresh_size} (likely merged with another "
                          f"position in the meantime). Using the fresher value.")
                    size = fresh_size
                    pos = fresh_pos
        except Exception as e:
            print(f"  [WARN] {sym}: couldn't re-verify size before adopting ({e}) — "
                  f"proceeding with the earlier reading ({size}).")

        adopt_real_position(state, sym, product, pos, size)

    # Cancel any stray open orders for our watched products that don't
    # belong to a currently-tracked position (prevents duplicate-order buildup)
    tracked_product_id = state["position"]["product_id"] if state["position"] else None
    for sym, product in products.items():
        try:
            live_orders = client.get_live_orders({"product_id": product["id"]})
            if isinstance(live_orders, dict):
                live_orders = live_orders.get("result", [])
        except Exception as e:
            print(f"  [WARN] couldn't fetch open orders for {sym}: {e}")
            continue
        for o in live_orders:
            oid = o.get("id")
            if product["id"] != tracked_product_id:
                print(f"  Cancelling stray leftover order on {sym} (id={oid})")
                try:
                    client.cancel_order(product["id"], oid)
                except Exception as e:
                    print(f"  [WARN] couldn't cancel stray order {oid}: {e}")

    print("--- Reconciliation done ---\n")
    return state


def adopt_real_position(state, sym, product, pos, size):
    """
    ---- Extracted into a reusable function ----
    This used to live ONLY inline inside reconcile_with_exchange (startup-
    only). Now also called right after a create_order timeout during a
    normal entry attempt, when a follow-up check finds a REAL position
    now exists on the exchange (the order actually landed and filled,
    even though the client-side request timed out before confirming it).
    Without this, that trade would just be silently forgotten by this
    script — protected by its own exchange-side bracket at best, but with
    zero staircase/trailing management from here on.

    Recomputes stop/target from current data (the original signal candle
    isn't available for a position we didn't place ourselves in this
    call), attaches a safety-net bracket, and populates state["position"]
    so the normal per-loop management picks it up starting next loop.
    """
    entry_price = float(pos.get("entry_price", 0))
    side = "long" if size > 0 else "short"
    if state["position"] is not None and state["position"]["symbol"] == sym:
        print(f"  Local state already tracking {sym} — keeping existing stop/target.")
        return
    print(f"  Adopting untracked real position: {sym} {side} @ {entry_price}. "
          f"Recomputing stop/target from current data since original signal "
          f"candle isn't available.")
    guessed_strategy = guess_strategy_for_adopted_position(sym, product["id"], entry_price, side)
    print(f"    Guessed strategy for this adopted position: Logic {guessed_strategy}")

    order_side = "buy" if side == "long" else "sell"

    if guessed_strategy == "B":
        # ---- Logic B formula: ATR-based stop (with the same
        # percentage floor used at normal entry), TP forced to a
        # fixed TP_RR_MULT reward:risk ratio off that distance. ----
        df_b = fetch_candles(sym, resolution="1m")
        stop = target = None
        entry_atr = None
        if df_b is not None and len(df_b) > ATR_PERIOD:
            df_b = compute_atr(df_b, period=ATR_PERIOD)
            entry_atr = float(df_b["atr"].iloc[-1])
            raw_dist_pct = (entry_atr * SL_ATR_MULT) / entry_price * 100
            stop_dist_pct = max(raw_dist_pct, MIN_STOP_DIST_PCT_B)
            sl_dist = entry_price * (stop_dist_pct / 100)
            tp_dist = sl_dist * TP_RR_MULT
            stop = entry_price - sl_dist if side == "long" else entry_price + sl_dist
            target = entry_price + tp_dist if side == "long" else entry_price - tp_dist
        if stop is None:
            # Couldn't fetch fresh ATR data — safe percentage fallback
            stop = entry_price * (0.99 if side == "long" else 1.01)
            target = entry_price * (1.02 if side == "long" else 0.98)
            entry_atr = 0.0

        bracket_active = False
        bracket_sl_order_id = None
        try:
            place_bracket_with_retry(product["id"], sym, order_side, abs(size), stop, target, side)
            bracket_active = True
            print(f"    Exchange-side bracket attached for adopted Logic B "
                  f"position: SL={stop:.6f} TP={target:.6f}")
            try:
                bracket_sl_order_id = get_bracket_stop_loss_order_id(product["id"])
            except Exception as e:
                print(f"    [WARN] Could not cache the bracket's stop-loss leg id "
                      f"yet ({e}) — will try again if/when trailing needs it.")
        except Exception as e:
            print(f"    [WARN] Could not attach bracket for adopted Logic B "
                  f"position ({e}) — will rely on the polling loop for this trade.")

        state["position"] = {
            "symbol": sym, "product_id": product["id"], "side": side,
            "size": abs(size), "original_size": abs(size),
            "entry_price": entry_price, "stop": stop, "target": target,
            "entry_time": str(now_ist()), "strategy": "B",
            "entry_order_id": None, "bracket_active": bracket_active,
            "bracket_sl_order_id": bracket_sl_order_id,
            "exchange_stop_synced": stop if bracket_active else None,
            "entry_atr": entry_atr, "extreme_price": entry_price,
            "r_basis": abs(entry_price - stop), "early_exit_done": False,
        }
    else:
        print(f"    Recomputing stop/target from current data since original "
              f"signal candle isn't available.")
        df = fetch_candles(sym)
        stop = target = None
        if df is not None and len(df) > SWING_LOOKBACK:
            if side == "long":
                candidate_stop = df["low"].iloc[-SWING_LOOKBACK:].min() * (1 - STOP_BUFFER_PCT / 100)
                candidate_dist_pct = (entry_price - candidate_stop) / entry_price * 100
            else:
                candidate_stop = df["high"].iloc[-SWING_LOOKBACK:].max() * (1 + STOP_BUFFER_PCT / 100)
                candidate_dist_pct = (candidate_stop - entry_price) / entry_price * 100

            # ---- Sanity check (BUG FIX) ----
            # If price has moved significantly since the original real
            # entry (e.g. rallied further before this restart/reconcile
            # ran), the recent swing low/high can end up on the WRONG
            # side of entry_price — e.g. a "swing low" that's actually
            # ABOVE the long's entry price. That silently produces a
            # NEGATIVE stop_dist_pct, which then flows into an INVERTED
            # target (below entry for a long, above entry for a short).
            # This actually happened in practice: the exchange safety
            # bracket correctly rejected it as "immediate_execution"
            # (since a stop already on the wrong side of price is
            # trivially triggerable), but our own polling-loop stop/
            # target check had no such guard and would have closed the
            # position almost immediately against the wrong level.
            # Fix: only use the swing-based level if it actually sits on
            # the valid side of entry; otherwise fall back to the same
            # safe percentage-based stop/target used when there isn't
            # enough candle data at all.
            if candidate_dist_pct > 0:
                stop = candidate_stop
                stop_dist_pct = candidate_dist_pct
                target = (entry_price * (1 + (stop_dist_pct * RISK_REWARD_MULT) / 100) if side == "long"
                          else entry_price * (1 - (stop_dist_pct * RISK_REWARD_MULT) / 100))
            else:
                print(f"    [WARN] Swing-based stop for {sym} came out on the WRONG side "
                      f"of entry (price has likely moved a lot since the real entry) — "
                      f"falling back to a safe percentage-based stop/target instead.")

        if stop is None:
            stop = entry_price * (0.99 if side == "long" else 1.01)
            target = entry_price * (1.02 if side == "long" else 0.98)

        exchange_safety_stop_active = False
        bracket_sl_order_id = None
        try:
            place_bracket_with_retry(product["id"], sym, order_side, abs(size), stop, target, side)
            exchange_safety_stop_active = True
            print(f"    Exchange-side safety-net bracket attached for adopted "
                  f"position: SL={stop:.6f} TP={target:.6f}")
            try:
                bracket_sl_order_id = get_bracket_stop_loss_order_id(product["id"])
            except Exception as e:
                print(f"    [WARN] Could not cache the bracket's stop-loss leg id "
                      f"yet ({e}) — will try again if/when the staircase needs it.")
        except Exception as e:
            print(f"    [WARN] Could not attach safety-net bracket for adopted "
                  f"position ({e}) — will rely on the polling loop for this trade.")

        state["position"] = {
            "symbol": sym, "product_id": product["id"], "side": side,
            "size": abs(size), "original_size": abs(size),
            "entry_price": entry_price, "stop": stop, "target": target,
            "entry_time": str(now_ist()), "milestones_locked": 0,
            "max_progress": 0.0, "strategy": "A",
            "entry_order_id": None,  # unknown for an adopted position — see note below
            "exchange_safety_stop_active": exchange_safety_stop_active,
            "exchange_stop_synced": stop if exchange_safety_stop_active else None,
            "bracket_sl_order_id": bracket_sl_order_id,
        }
        # NOTE: entry_order_id is None here because we never placed the
        # original entry order ourselves (it happened before this
        # restart). This means later staircase-trailing updates to this
        # particular bracket will be skipped (edit_bracket_order needs
        # that id) — the exchange-side stop will stay frozen at this
        # initial level rather than trailing. The bracket still fully
        # protects the ORIGINAL stop level either way; it just won't
        # ratchet tighter as profit builds, unlike bracketed positions
        # opened normally by this script. Acceptable trade-off for a
        # rare edge case (adopting a position after an unplanned restart).


# ============================================================
# Position management (partial TP, trailing SL, smart near-target exit)
# ============================================================

def compute_progress(position, current_price):
    entry = position["entry_price"]
    target = position["target"]
    if position["side"] == "long":
        total_distance = target - entry
        if total_distance <= 0:
            return 0.0
        return (current_price - entry) / total_distance
    else:
        total_distance = entry - target
        if total_distance <= 0:
            return 0.0
        return (entry - current_price) / total_distance


def cancel_all_orders_for_product(product_id):
    """
    ---- BUG FIX v4: stop using the /v2/orders/all bulk endpoint ----
    v3 (product_id as a query param, matching Delta's docs) fixed the
    "silently does nothing" problem, but uncovered a NEW issue: this
    specific bulk-DELETE endpoint returns "401 Signature Mismatch" for
    this account when product_id is passed as a query param — likely an
    account/endpoint-specific quirk in how the request signature is
    validated for this route. A 401 is even worse than v1/v2's problem,
    since it means we could NEVER confirm a clean slate through this path.

    Fix: stop relying on the bulk endpoint entirely. Instead, fetch this
    product's live open orders (a plain GET — proven reliable elsewhere
    in this file) and cancel each one individually via the same
    single-order cancel_order() call already used successfully (from a
    signature standpoint) in startup reconciliation. Raises on any
    failure so callers correctly skip this loop's entry rather than
    proceeding on an unconfirmed state.
    """
    live_orders = client.get_live_orders({"product_id": product_id})
    if isinstance(live_orders, dict):
        live_orders = live_orders.get("result", [])
    for o in live_orders:
        oid = o.get("id")
        if oid is not None:
            try:
                client.cancel_order(product_id, oid)
            except Exception as e:
                # ---- BUG FIX: 'open_order_not_found' means the order is
                # ALREADY gone — that's success for our purposes (nothing
                # left to cancel), not a failure. Without this, a single
                # stale/ghost order-id returned by get_live_orders (that
                # the exchange itself no longer recognizes when we try to
                # cancel it) caused an INFINITE false-skip loop: every
                # entry attempt saw this same "failure" forever, even
                # though Positions/Open Orders/Stop Orders on the exchange
                # were all completely empty (confirmed directly against
                # the exchange UI). Only re-raise for genuine uncertainty
                # (timeouts, other errors) where we truly can't confirm
                # the state — those should still cause a cautious skip.
                if "open_order_not_found" in str(e):
                    print(f"    [INFO] order {oid} on product {product_id} was already "
                          f"gone (open_order_not_found) — treating as already-cancelled.")
                    continue
                raise


def confirm_no_open_orders(product_id):
    """
    Actually VERIFIES (via a fresh GET) that no open orders remain for
    this product — rather than just trusting that the DELETE call above
    didn't raise an exception. That trust was exactly what let 5 stray
    orders pile up silently: the DELETE was hitting the wrong parameter
    location and "succeeding" while doing nothing. Returns True only if
    we positively confirmed the product is clean; False/None (fetch
    failed or a GENUINE order still present) means the caller should NOT
    proceed with a new entry this loop.

    ---- BUG FIX: don't just count the raw list length ----
    Found in practice: Delta's own GET /v2/orders kept returning the SAME
    order-id for ETHUSD in every single call, indefinitely — but actually
    trying to cancel that exact id always came back "open_order_not_found"
    (i.e. it doesn't really exist; it's a stale/ghost entry in the list
    response itself). A raw "len(live_orders) == 0" check can NEVER pass
    while that ghost entry keeps showing up, causing an infinite false
    "still has open orders" skip — even though the exchange's own
    Positions/Open-Orders/Stop-Orders UI showed completely empty. Now:
    for anything still listed, try to cancel it and tolerate
    "open_order_not_found" the same way cancel_all_orders_for_product
    does — only a GENUINE, cancelable (or otherwise-erroring) order
    counts as "not clean".
    """
    try:
        live_orders = client.get_live_orders({"product_id": product_id})
        if isinstance(live_orders, dict):
            live_orders = live_orders.get("result", [])
    except Exception as e:
        print(f"    [WARN] couldn't verify open-orders are clean ({e})")
        return False

    for o in live_orders:
        oid = o.get("id")
        if oid is None:
            continue
        try:
            client.cancel_order(product_id, oid)
        except Exception as e:
            if "open_order_not_found" in str(e):
                print(f"    [INFO] order {oid} on product {product_id} is a stale/ghost "
                      f"list-entry (open_order_not_found on cancel) — not a real blocker.")
                continue
            print(f"    [WARN] order {oid} on product {product_id} still couldn't be "
                  f"cancelled ({e}) — treating as genuinely not clean.")
            return False
    return True


def manage_bracket_position_b(state, pos, symbol_data):
    """
    Manages a Logic B position that has a native bracket order (SL+TP)
    already attached on the exchange. Two jobs each loop:
      1. Detect if the exchange-side bracket already closed the position
         (SL or TP triggered) — if so, log it and clear local tracking.
      2. If still open, check whether ATR has expanded significantly since
         entry (a sign of a strengthening/volatile trend) — if so, trail
         the stop behind the best price seen, replacing the bracket with
         a tighter SL (and a deliberately distant TP, so the trailing stop
         does the real exit management from here on), matching the
         reference bot's design.
    """
    sym = pos["symbol"]
    side = pos["side"]

    try:
        live_pos = client.get_position(pos["product_id"])
        size_now = float(live_pos.get("size", 0)) if isinstance(live_pos, dict) else 0
    except Exception as e:
        # ---- BUG FIX: don't just give up on the whole loop here. ----
        # This used to `return` immediately on any failure of this single
        # exchange call — which meant EVERYTHING below (external-close
        # detection, staircase progress, ATR-expansion trailing, the 1.5R
        # early-exit) was skipped too, not just this one status check.
        # Found in practice: a 27+ minute run of consecutive 504 Gateway
        # Timeouts on this call left a position sitting at ~53% progress
        # toward its target with ZERO staircase milestones locked, purely
        # because this early return kept blocking the staircase logic
        # further down from ever running — even though that logic doesn't
        # actually need this exchange call at all (it works off local
        # candle data). Now matches Logic A's identical fix: assume the
        # position is still open (its exchange-side bracket keeps
        # protecting it regardless) and fall through to the candle-based
        # management below, instead of freezing everything on one flaky
        # API call.
        print(f"  [WARN] {sym}: couldn't check live position status ({e}) — "
              f"falling back to local candle-based management for now (the "
              f"exchange-side bracket still protects the position either way).")
        size_now = pos["size"]  # assume still open this loop; try again next loop

    if size_now == 0:
        # ---- BUG FIX: require confirmation on a SECOND consecutive loop
        # before trusting this. Observed in practice: during an exchange
        # disruption event, get_position() returned a technically-valid
        # response with size=0 even though the position was still genuinely
        # open (confirmed afterward on the exchange's own UI) — Delta's own
        # backend was inconsistent during that outage. Acting on a single
        # reading caused local tracking to wrongly think the position was
        # flat, and the bot went on to try opening brand-new trades while
        # the real position sat unmonitored (still protected by its
        # bracket, luckily, but untracked locally). Now requires the SAME
        # zero-size reading on two consecutive loops (~20s apart) before
        # believing it — a small delay in detecting a genuine close, worth
        # it to avoid losing track of a real, still-open position. ----
        confirmations = pos.get("_zero_size_confirmations", 0) + 1
        if confirmations < 2:
            pos["_zero_size_confirmations"] = confirmations
            state["position"] = pos
            print(f"  [WARN] {sym}: exchange shows zero size — could be a genuine "
                  f"close, or a flaky/incomplete API response (seen during past "
                  f"exchange disruptions). Confirming again next loop before "
                  f"treating this as closed.")
            return
        # Bracket already closed this position on the exchange side. We
        # don't get an exact fill price/reason from this check alone, so
        # we log it using the current market price as a reasonable proxy.
        current_price = symbol_data[sym].iloc[-1]["close"] if sym in symbol_data else pos["entry_price"]
        entry_price = pos["entry_price"]
        if side == "long":
            approx_pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            approx_pnl_pct = (entry_price - current_price) / entry_price * 100
        net_pnl_pct = estimate_net_pnl_pct(approx_pnl_pct, "B")

        # ---- BUG FIX: try to get the REAL exit price + REAL fee from
        # Delta's own /v2/fills before falling back to the candle-price
        # proxy above (see get_real_fill_and_fee for the full story). ----
        real_exit_price, real_fee_amount = get_real_fill_and_fee(pos["product_id"])
        log_exit_price = real_exit_price if real_exit_price is not None else current_price
        print(f"  {sym}: bracket order already closed this position on the exchange "
              f"(SL or TP triggered). Exit ~{log_exit_price:.6f}"
              f"{' (real fill)' if real_exit_price is not None else ' (approx — real fill lookup failed)'}, "
              f"approx gross P&L: {approx_pnl_pct:+.3f}% "
              f"(approx net after fees: {net_pnl_pct:+.3f}%)")
        log_trade_event(
            time=str(now_ist()), symbol=sym, action="CLOSE",
            side=("sell" if side == "long" else "buy"), size=pos["size"],
            reason="bracket_closed", entry_price=entry_price, exit_price=current_price,
            approx_gross_pnl_pct=round(approx_pnl_pct, 4),
            approx_net_pnl_pct_after_fees=round(net_pnl_pct, 4), fill_method="bracket",
            strategy="B", real_exit_price=real_exit_price, real_fee_amount=real_fee_amount,
        )
        state["last_trade_close_time_b"] = time.time()
        state["position"] = None
        return
    else:
        pos["_zero_size_confirmations"] = 0

    if sym not in symbol_data:
        return
    latest = symbol_data[sym].iloc[-1]
    current_price = latest["close"]
    current_atr = latest["atr"]
    entry_atr = pos.get("entry_atr")

    if pd.isna(current_atr) or entry_atr is None or entry_atr <= 0:
        return

    # ---- NEW: Local safety-net fallback for persistent exchange-sync
    # failures. Track how long pos["stop"] (our locally-intended level)
    # has differed from pos["exchange_stop_synced"] (what's actually
    # confirmed live on the exchange). If that gap has persisted for
    # LOGIC_B_LOCAL_FALLBACK_SECONDS or more (observed in practice:
    # CloudFront 403s blocking the sync lookup for 30+ minutes straight)
    # AND price has now crossed the intended (tighter) stop, close the
    # position ourselves directly — don't keep waiting indefinitely for
    # the exchange sync to succeed. The OLD, wider exchange-side bracket
    # is still live as a backstop the whole time regardless; this adds
    # protection at the level we actually intended, closing the exact
    # gap Logic A already covered via its own local candle-based check
    # (which Logic B never had, having been designed to rely entirely on
    # the exchange-side bracket).
    stop_confirmed_synced = pos.get("exchange_stop_synced") == pos["stop"]
    if stop_confirmed_synced:
        pos["stop_sync_failing_since"] = None
    else:
        if pos.get("stop_sync_failing_since") is None:
            pos["stop_sync_failing_since"] = time.time()
        failing_for = time.time() - pos["stop_sync_failing_since"]
        stop_hit_locally = ((side == "long" and current_price <= pos["stop"]) or
                             (side == "short" and current_price >= pos["stop"]))
        if failing_for >= LOGIC_B_LOCAL_FALLBACK_SECONDS and stop_hit_locally:
            print(f"  [WARN] {sym}: exchange-side bracket sync has been failing for "
                  f"{failing_for:.0f}s AND price ({current_price:.6f}) has now crossed "
                  f"the intended stop ({pos['stop']:.6f}) — closing directly via a local "
                  f"market order instead of waiting any longer for the exchange sync.")
            try:
                close_resp, close_method = place_market_order_direct(
                    pos["product_id"], ("sell" if side == "long" else "buy"),
                    pos["size"], reduce_only=True)
                exit_price = _extract_fill_price(close_resp) or current_price
                entry_price = pos["entry_price"]
                if side == "long":
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price * 100
                net_pnl_pct = estimate_net_pnl_pct(pnl_pct, "B")
                log_trade_event(
                    time=str(now_ist()), symbol=sym, action="CLOSE",
                    side=("sell" if side == "long" else "buy"), size=pos["size"],
                    reason="local_fallback_stop", entry_price=entry_price, exit_price=exit_price,
                    approx_gross_pnl_pct=round(pnl_pct, 4),
                    approx_net_pnl_pct_after_fees=round(net_pnl_pct, 4),
                    fill_method=close_method, strategy="B",
                )
                print(f"  {sym}: closed via local fallback. Approx gross P&L: {pnl_pct:+.3f}% "
                      f"(net after fees: {net_pnl_pct:+.3f}%)")
                state["last_trade_close_time_b"] = time.time()
                state["position"] = None
                return
            except Exception as e:
                print(f"  [WARN] {sym}: local fallback close attempt failed too ({e}) — "
                      f"will retry next loop. The original exchange-side bracket is still "
                      f"live as a backstop in the meantime.")

    # Track the best (most favorable) price seen so far
    extreme = pos.get("extreme_price", pos["entry_price"])
    if side == "long":
        extreme = max(extreme, current_price)
    else:
        extreme = min(extreme, current_price)
    pos["extreme_price"] = extreme

    # ---- Staircase stop-loss ratcheting (ADDED — same system Logic A
    # uses: 20%->breakeven, 50%->20%, 75%->50%, 90%->75% of the distance
    # to target). This runs independently of the ATR-expansion trailing
    # below — it guarantees a baseline profit-lock purely from price
    # progress, even on quiet trades that never trigger the ATR-expansion
    # trailing at all. When ATR-expansion trailing ALSO triggers this loop,
    # whichever of the two gives the MORE protective (tighter, in the
    # favorable direction) stop is the one that's actually applied. ----
    entry = pos["entry_price"]
    target = pos["target"]
    progress = compute_progress(pos, current_price)
    pos["max_progress"] = max(pos.get("max_progress", 0.0), progress)
    staircase_stop = pos["stop"]
    locked_so_far = pos.get("milestones_locked", 0)
    for i in range(locked_so_far, len(STAIRCASE_TRIGGERS)):
        trigger = STAIRCASE_TRIGGERS[i]
        lock_level = STAIRCASE_LOCKS[i]
        if pos["max_progress"] < trigger:
            break
        candidate = (entry + lock_level * (target - entry) if side == "long"
                     else entry - lock_level * (entry - target))
        if (side == "long" and candidate > staircase_stop) or (side == "short" and candidate < staircase_stop):
            staircase_stop = candidate
        pos["milestones_locked"] = i + 1

    expansion_pct = ((current_atr - entry_atr) / entry_atr) * 100.0

    # ---- R-multiple early exit (matches reference bot design) ----
    # ---- BUG FIX: delayed from 1.0R to 1.5R (handoff-doc's known Logic-B
    # fix #3). At exactly 1R, the "modest guaranteed profit" being locked
    # in was often barely bigger than round-trip fees once slippage from
    # the market-order entry/exit was accounted for — so this early exit
    # was frequently taking a near-breakeven trade off the table instead
    # of letting genuinely-working setups run further. 1.5R gives the
    # trade more room to prove itself before the "lock in profit now"
    # decision, while still protecting against a full round-trip back to
    # breakeven/loss on setups that stall. ----
    # Once the trade reaches 1.5R profit (moved 1.5x its SL-distance in
    # its favor), take the modest guaranteed profit now — UNLESS ATR has
    # expanded significantly, which signals the move may have more room,
    # in which case we skip the early exit and let the ATR-expansion
    # trailing logic below manage it instead (riding toward the full TP).
    r_basis = pos.get("r_basis")
    if r_basis and r_basis > 0 and not pos.get("early_exit_done", False):
        if side == "long":
            r_multiple = (current_price - pos["entry_price"]) / r_basis
        else:
            r_multiple = (pos["entry_price"] - current_price) / r_basis

        if r_multiple >= EARLY_EXIT_R_MULTIPLE:
            if expansion_pct >= ATR_EXPANSION_TRIGGER_PCT:
                print(f"  {sym}: reached {EARLY_EXIT_R_MULTIPLE}R profit AND ATR expanded "
                      f"{expansion_pct:.1f}% — skipping early exit, letting the trailing "
                      f"stop manage this trade for potentially more than {EARLY_EXIT_R_MULTIPLE}R.")
                pos["early_exit_done"] = True  # don't re-check every loop once decided
            else:
                print(f"  {sym}: reached {EARLY_EXIT_R_MULTIPLE}R profit (r_multiple={r_multiple:.2f}), "
                      f"ATR has NOT expanded — taking the early exit now instead of risking a "
                      f"round-trip back to breakeven/loss.")
                try:
                    cancel_all_orders_for_product(pos["product_id"])
                except Exception as e:
                    print(f"  [WARN] {sym}: couldn't cancel bracket before early exit ({e})")
                try:
                    resp, method = place_market_order_direct(
                        pos["product_id"], ("sell" if side == "long" else "buy"),
                        pos["size"], reduce_only=True)
                    entry_price = pos["entry_price"]
                    pnl_pct = ((current_price - entry_price) / entry_price * 100 if side == "long"
                               else (entry_price - current_price) / entry_price * 100)
                    net_pnl_pct = estimate_net_pnl_pct(pnl_pct, "B")
                    log_trade_event(
                        time=str(now_ist()), symbol=sym, action="CLOSE",
                        side=("sell" if side == "long" else "buy"), size=pos["size"],
                        reason="early_exit_1_5R", entry_price=entry_price, exit_price=current_price,
                        approx_gross_pnl_pct=round(pnl_pct, 4),
                        approx_net_pnl_pct_after_fees=round(net_pnl_pct, 4),
                        fill_method=method, strategy="B",
                    )
                    print(f"  {sym}: CLOSED early at {EARLY_EXIT_R_MULTIPLE}R, approx gross P&L: {pnl_pct:+.3f}% "
                          f"(approx net after fees: {net_pnl_pct:+.3f}%)")
                    state["last_trade_close_time_b"] = time.time()
                    state["position"] = None
                    return
                except Exception as e:
                    print(f"  [WARN] {sym}: early-exit order failed ({e}) — will retry next loop")

    if expansion_pct < ATR_EXPANSION_TRIGGER_PCT:
        # ATR hasn't expanded enough for trailing-mode yet, but the
        # staircase ratchet above may still have produced a tighter stop
        # purely from price progress — apply that on its own if so,
        # keeping the ORIGINAL target unchanged (staircase only ever
        # tightens the stop, same as Logic A).
        new_stop = staircase_stop
        new_tp = pos["target"]
        trigger_desc = f"staircase progress reached {pos['max_progress']:.2f}"
    else:
        trail_dist = TRAIL_MULT_B * current_atr
        atr_stop = extreme - trail_dist if side == "long" else extreme + trail_dist
        # Whichever of the staircase-ratchet stop and the ATR-trail stop
        # is more protective (tighter, in the favorable direction) wins.
        new_stop = max(staircase_stop, atr_stop) if side == "long" else min(staircase_stop, atr_stop)
        # Matches reference bot's _far_take_profit(): far_dist = original
        # SL-distance (r_basis) * 20, anchored off ENTRY price — not the
        # current trailing distance/extreme, which would drift each update.
        r_basis = pos.get("r_basis", trail_dist)
        far_dist = r_basis * 20
        entry_price = pos["entry_price"]
        new_tp = entry_price + far_dist if side == "long" else entry_price - far_dist
        trigger_desc = f"ATR expanded {expansion_pct:.1f}% since entry (>= {ATR_EXPANSION_TRIGGER_PCT}%)"

    old_stop = pos["stop"]
    improved = (side == "long" and new_stop > old_stop) or (side == "short" and new_stop < old_stop)
    if improved:
        # ---- BUG FIX: update the LOCAL stop/target immediately, regardless
        # of whether the exchange-side push below succeeds. Previously
        # pos["stop"] was only assigned INSIDE the try block, AFTER the
        # exchange API calls — so if those calls raised (which they did
        # repeatedly in practice, from CloudFront 403s and similar), the
        # local stop silently never advanced either, even though
        # milestones_locked kept incrementing in the staircase loop above
        # (that counter isn't gated on push-success). This produced exactly
        # the mismatch seen in practice: the dashboard showing "3/4 steps
        # locked" while Stop still displayed the very first, un-tightened
        # value. Now the local value is authoritative and updates
        # immediately; only the EXCHANGE-side sync is retried below (and
        # again on later loops via exchange_stop_synced, same pattern
        # already used for Logic A). ----
        print(f"  {sym}: {trigger_desc} -> trailing stop from {old_stop:.6f} to {new_stop:.6f}")
        pos["stop"] = new_stop
        pos["target"] = new_tp

    if pos.get("exchange_stop_synced") != pos["stop"]:
        far_tp = pos["target"]
        try:
            bracket_sl_order_id = pos.get("bracket_sl_order_id")
            if bracket_sl_order_id is None:
                # Not cached yet (e.g. attach happened before this fix, or the
                # cache attempt failed earlier) — fall back to looking it up now.
                bracket_sl_order_id = get_bracket_stop_loss_order_id(pos["product_id"])
                pos["bracket_sl_order_id"] = bracket_sl_order_id

            if bracket_sl_order_id is not None:
                try:
                    # CORRECT approach: edit the EXISTING bracket via PUT, using the
                    # bracket's OWN stop-loss leg order id (a resting/pending order
                    # Delta created when the bracket attached) — not our original
                    # MARKET entry order's id, which is already 'closed' by the
                    # time we're trailing and gets rejected as 'open_order_not_found'.
                    edit_bracket_order(bracket_sl_order_id, pos["product_id"], sym, pos["stop"], far_tp)
                except Exception as e:
                    # The cached id might have gone stale (rare — e.g. if the
                    # bracket was somehow replaced). Refresh it once and retry
                    # before giving up, rather than immediately falling back to
                    # the disruptive cancel+recreate path.
                    print(f"  [WARN] {sym}: edit with cached bracket id failed ({e}) — "
                          f"refreshing the id and retrying once.")
                    bracket_sl_order_id = get_bracket_stop_loss_order_id(pos["product_id"])
                    pos["bracket_sl_order_id"] = bracket_sl_order_id
                    if bracket_sl_order_id is None:
                        raise
                    edit_bracket_order(bracket_sl_order_id, pos["product_id"], sym, pos["stop"], far_tp)
            else:
                # Couldn't find the bracket's stop-loss leg — fall back to the
                # cancel+recreate approach as a last resort.
                #
                # ---- BUG FIX: fetch the REAL, current size from the exchange
                # first, rather than trusting our own tracked pos["size"].
                # If this position ever merged with another trade opened on
                # the same symbol (Delta nets same-direction positions
                # together automatically — observed in practice: a Logic-B
                # entry combined with an existing untracked position into
                # one bigger position), our local pos["size"] could be
                # SMALLER than the real size. Recreating the bracket using
                # the stale, smaller size would leave the extra (merged-in)
                # portion of the position completely UNPROTECTED. ----
                print(f"  [WARN] {sym}: couldn't find the bracket's stop-loss leg order — "
                      f"falling back to cancel+recreate.")
                real_size = pos["size"]
                try:
                    live_pos = client.get_position(pos["product_id"])
                    if isinstance(live_pos, dict) and live_pos.get("size"):
                        real_size = abs(float(live_pos["size"]))
                        if real_size != pos["size"]:
                            print(f"  [INFO] {sym}: real exchange size ({real_size}) differs "
                                  f"from locally tracked size ({pos['size']}) — likely merged "
                                  f"with another position. Using the real size for the "
                                  f"recreated bracket so the FULL position stays protected.")
                            pos["size"] = real_size
                except Exception as e:
                    print(f"  [WARN] {sym}: couldn't verify real position size before "
                          f"recreating bracket ({e}) — using last-known local size "
                          f"({real_size}) as a fallback.")
                cancel_all_orders_for_product(pos["product_id"])
                place_bracket_with_retry(
                    pos["product_id"], sym, ("buy" if side == "long" else "sell"),
                    real_size, pos["stop"], far_tp, side)
            pos["exchange_stop_synced"] = pos["stop"]
            print(f"  {sym}: exchange-side bracket synced: SL -> {pos['stop']:.6f}")
        except Exception as e:
            print(f"  [WARN] {sym}: exchange-side bracket still out of sync with local "
                  f"stop ({pos['stop']:.6f}) — will keep retrying every loop until this "
                  f"succeeds ({e})")

    state["position"] = pos


def manage_open_position(state, symbol_data):
    pos = state["position"]
    sym = pos["symbol"]
    if sym not in symbol_data:
        print(f"  [WARN] no fresh data for open position {sym} this loop")
        return

    latest = symbol_data[sym].iloc[-1]
    price = latest["close"]
    side = pos["side"]

    # ---- Live unrealized P&L snapshot (for the dashboard corner widget) ----
    entry_price = pos["entry_price"]
    if side == "long":
        live_pnl_pct = (price - entry_price) / entry_price * 100
    else:
        live_pnl_pct = (entry_price - price) / entry_price * 100
    LATEST_STATE["position"] = pos
    LATEST_STATE["current_price"] = price
    LATEST_STATE["live_pnl_pct"] = live_pnl_pct

    # ---- Logic B: exchange handles the actual SL/TP execution natively.
    # Our job here is just to detect closure and manage ATR-expansion-
    # triggered trailing.
    #
    # ---- BUG FIX: this used to require bracket_active == True before even
    # calling manage_bracket_position_b — meaning if the INITIAL bracket
    # attach failed (e.g. 'bracket_order_exists' because a stale bracket
    # from an earlier trade was still attached), this Logic B position got
    # ZERO active management at all: no external-close-detection, no ATR-
    # trailing, no staircase-sync, nothing — it just silently fell through
    # to Logic A's code below with the wrong strategy-specific logic
    # applied. manage_bracket_position_b() doesn't actually depend on
    # bracket_active being True — it independently looks up whatever
    # bracket currently exists (ours or a leftover one) via
    # get_bracket_stop_loss_order_id, so it's safe to always call it for
    # any Logic B position. ----
    if pos.get("strategy") == "B":
        manage_bracket_position_b(state, pos, symbol_data)
        return

    # ---- Logic A with an active exchange-side safety bracket: check FIRST
    # whether that bracket already closed the position on the exchange
    # (SL or TP triggered there) before relying on our own candle-based
    # stop/target check below. This closes a real gap that was found in
    # practice: the safety bracket correctly stopped out a losing trade,
    # but this script's own local state kept showing the position as
    # still open (since the candle-based check alone never "saw" the
    # exchange-side close), leaving the dashboard stale and blocking new
    # Logic A entries until a manual restart forced a reconciliation. ----
    if pos.get("strategy") == "A" and pos.get("exchange_safety_stop_active"):
        try:
            live_pos = client.get_position(pos["product_id"])
            size_now = float(live_pos.get("size", 0)) if isinstance(live_pos, dict) else pos["size"]
        except Exception as e:
            print(f"  [WARN] {sym}: couldn't check live position status against the "
                  f"exchange ({e}) — will retry next loop, falling back to the "
                  f"local candle-based stop/target check for now.")
            size_now = pos["size"]  # assume still open this loop; try again next loop

        if size_now == 0:
            # ---- BUG FIX: require confirmation on a SECOND consecutive
            # loop before trusting this. Observed in practice during an
            # exchange disruption event: get_position() returned size=0
            # even though the position was still genuinely open (confirmed
            # afterward on the exchange's own UI, still showing the exact
            # same entry/SL/TP) — Delta's own backend was inconsistent
            # during that outage. Acting on the single reading wrongly
            # cleared local tracking, and the bot went on to try opening
            # brand-new trades while the real position sat untracked
            # locally (though still protected by its bracket on the
            # exchange, luckily). Now requires the SAME zero-size reading
            # on two consecutive loops (~20s apart) before believing it. ----
            confirmations = pos.get("_zero_size_confirmations", 0) + 1
            if confirmations < 2:
                pos["_zero_size_confirmations"] = confirmations
                state["position"] = pos
                print(f"  [WARN] {sym}: exchange shows zero size — could be a genuine "
                      f"close, or a flaky/incomplete API response (seen during past "
                      f"exchange disruptions). Confirming again next loop before "
                      f"treating this as closed.")
                return
            # The exchange-side safety bracket already closed this position
            # (SL or TP triggered there). We don't get an exact fill price
            # from this simple existence-check, so we log it using the
            # current close price as a reasonable proxy — same approach
            # already used for Logic B's equivalent bracket-closed case.
            approx_exit_price = price
            if side == "long":
                approx_pnl_pct = (approx_exit_price - entry_price) / entry_price * 100
            else:
                approx_pnl_pct = (entry_price - approx_exit_price) / entry_price * 100
            net_pnl_pct = estimate_net_pnl_pct(approx_pnl_pct, "A")

            # ---- BUG FIX: try to get the REAL exit price + REAL fee from
            # Delta's own /v2/fills before falling back to the candle-price
            # proxy above. This is the exact bug found in practice: a BTCUSD
            # bracket-close logged here at the candle price ~64643.0 while
            # the real exchange fill was 64601.50 (a $41.5 difference,
            # which alone was skewing that trade's logged P&L by ~$0.81). ----
            real_exit_price, real_fee_amount = get_real_fill_and_fee(pos["product_id"])
            log_exit_price = real_exit_price if real_exit_price is not None else approx_exit_price
            print(f"  {sym}: exchange-side safety bracket already closed this position "
                  f"(SL or TP triggered there before the local check caught it). "
                  f"Exit ~{log_exit_price:.6f}"
                  f"{' (real fill)' if real_exit_price is not None else ' (approx — real fill lookup failed)'}, "
                  f"approx gross P&L: {approx_pnl_pct:+.3f}% (approx net after fees: {net_pnl_pct:+.3f}%)")
            log_trade_event(
                time=str(now_ist()), symbol=sym, action="CLOSE",
                side=("sell" if side == "long" else "buy"), size=pos["size"],
                reason="exchange_bracket_closed", entry_price=entry_price,
                exit_price=approx_exit_price, approx_gross_pnl_pct=round(approx_pnl_pct, 4),
                approx_net_pnl_pct_after_fees=round(net_pnl_pct, 4), fill_method="bracket",
                strategy="A", real_exit_price=real_exit_price, real_fee_amount=real_fee_amount,
            )
            state["position"] = None
            return
        else:
            pos["_zero_size_confirmations"] = 0

    # ---- Hard stop-loss / full target check first ----
    hit_stop = hit_target = False
    if side == "long":
        if latest["low"] <= pos["stop"]:
            hit_stop = True
        elif latest["high"] >= pos["target"]:
            hit_target = True
    else:
        if latest["high"] >= pos["stop"]:
            hit_stop = True
        elif latest["low"] <= pos["target"]:
            hit_target = True

    if hit_stop or hit_target:
        _close_position(state, pos, price, "stop" if hit_stop else "target", pos["size"])
        return

    # ---- Progress tracking + staircase stop-loss ratcheting (now applies
    # to BOTH Logic A and Logic B — added after observing that fixed SL/TP
    # alone let a profitable-looking trade fully round-trip back to a loss) ----
    progress = compute_progress(pos, price)
    pos["max_progress"] = max(pos.get("max_progress", 0.0), progress)

    entry = pos["entry_price"]
    target = pos["target"]
    locked_so_far = pos.get("milestones_locked", 0)  # how many staircase steps already applied

    for i in range(locked_so_far, len(STAIRCASE_TRIGGERS)):
        trigger = STAIRCASE_TRIGGERS[i]
        lock_level = STAIRCASE_LOCKS[i]
        if pos["max_progress"] < trigger:
            break  # triggers are in increasing order, so no need to check further ones yet

        if side == "long":
            new_stop = entry + lock_level * (target - entry)
        else:
            new_stop = entry - lock_level * (entry - target)

        old_stop = pos["stop"]
        # Only ever move the stop in the favorable direction, never loosen it
        if (side == "long" and new_stop > old_stop) or (side == "short" and new_stop < old_stop):
            pos["stop"] = new_stop
            print(f"  {sym}: progress reached {pos['max_progress']:.2f} (>= {trigger}) -> "
                  f"stop moved to the {lock_level:.0%} level ({old_stop:.5f} -> {new_stop:.5f})")
        pos["milestones_locked"] = i + 1

    # ---- Keep the exchange-side safety-net bracket IN SYNC with the local
    # staircase stop (BUG FIX). Previously this push only happened once,
    # right when a milestone first triggered — if that single attempt
    # failed (e.g. 'market_disrupted_cancel_only_mode' during an exchange
    # outage, observed in practice), the exchange-side hard stop stayed
    # frozen at the OLD, wider level FOREVER, since nothing retried it
    # unless a LATER milestone happened to trigger again. Now this runs
    # every loop regardless of whether a new milestone triggered THIS
    # loop, comparing the local stop against the last level we know we
    # successfully pushed — and keeps retrying every loop until the
    # exchange actually confirms the update, so a temporary disruption
    # can no longer cause a permanently-stale exchange-side stop. ----
    # ---- BUG FIX: this used to require exchange_safety_stop_active == True
    # before even trying to sync — meaning if our OWN initial bracket-attach
    # failed (e.g. 'bracket_order_exists' because a stale bracket from an
    # earlier trade was still attached to this symbol), staircase updates
    # would NEVER be pushed to the exchange, no matter how many times the
    # local stop moved. The staircase would keep computing perfectly
    # correct tighter levels locally forever, with zero way to reach the
    # exchange — "visible in the dashboard but never actually protecting
    # anything for real." Now this always tries to find and edit WHATEVER
    # bracket currently exists on this position (ours or a leftover one),
    # regardless of whether our own attach succeeded earlier. ----
    if pos.get("strategy") == "A":
        if pos.get("exchange_stop_synced") != pos["stop"]:
            # ---- BUG FIX: use the bracket's OWN stop-loss leg id (same
            # correct pattern Logic B already uses), not entry_order_id —
            # the original entry order is already 'closed' the instant it
            # fills, so Delta rejects edits to it with 'open_order_not_found'
            # (observed in practice: every staircase move after entry was
            # silently failing to reach the exchange for exactly this
            # reason). Cache it once, refresh on failure, same as Logic B. ----
            bracket_sl_order_id = pos.get("bracket_sl_order_id")
            if bracket_sl_order_id is None:
                try:
                    bracket_sl_order_id = get_bracket_stop_loss_order_id(pos["product_id"])
                    pos["bracket_sl_order_id"] = bracket_sl_order_id
                except Exception as e:
                    bracket_sl_order_id = None
                    print(f"    [WARN] Couldn't look up the bracket's stop-loss leg id "
                          f"({e}) — will retry next loop.")
            if bracket_sl_order_id is not None:
                try:
                    edit_bracket_order(bracket_sl_order_id, pos["product_id"], sym, pos["stop"], target)
                    pos["exchange_stop_synced"] = pos["stop"]
                    # Now confirmed a real bracket exists and is manageable —
                    # this also unlocks the external-close-detection check
                    # above, which was gated on this same flag.
                    pos["exchange_safety_stop_active"] = True
                    print(f"    Exchange-side safety-net bracket synced: SL -> {pos['stop']:.6f}")
                except Exception as e:
                    # The cached id might have gone stale — refresh once and retry.
                    print(f"    [WARN] Edit with cached bracket id failed ({e}) — "
                          f"refreshing the id and retrying once.")
                    try:
                        bracket_sl_order_id = get_bracket_stop_loss_order_id(pos["product_id"])
                        pos["bracket_sl_order_id"] = bracket_sl_order_id
                        if bracket_sl_order_id is not None:
                            edit_bracket_order(bracket_sl_order_id, pos["product_id"], sym, pos["stop"], target)
                            pos["exchange_stop_synced"] = pos["stop"]
                            pos["exchange_safety_stop_active"] = True
                            print(f"    Exchange-side safety-net bracket synced: SL -> {pos['stop']:.6f}")
                    except Exception as e2:
                        print(f"    [WARN] Exchange-side bracket still out of sync with local "
                              f"stop ({pos['stop']:.6f}) — will keep retrying every loop until "
                              f"this succeeds ({e2})")

    state["position"] = pos  # persist progress-tracking updates


def _close_position(state, pos, price, reason, size):
    sym = pos["symbol"]
    side = pos["side"]
    close_side = "sell" if side == "long" else "buy"
    strategy = pos.get("strategy", "A")

    try:
        if strategy == "B":
            resp, method = place_market_order_direct(
                pos["product_id"], close_side, size, reduce_only=True)
        else:
            resp, method = place_order_with_fallback(
                pos["product_id"], close_side, size, price, reduce_only=True)
    except Exception as e:
        if "no_position_for_reduce_only" in str(e):
            print(f"  [INFO] {sym}: exchange says there's no position left to close "
                  f"(it must have already been closed — possibly by the exchange-side "
                  f"bracket triggering just before our own attempt, a duplicate running "
                  f"instance of this script, or manually). Clearing local tracking.")
            # ---- BUG FIX: this used to just clear state and return, WITHOUT
            # ever logging a trade-close-event — meaning any trade closed
            # this way (observed to happen often, since Logic B especially
            # relies on the exchange bracket for the real execution) was
            # completely invisible in our own trades-log (Logic A/B dashboard
            # pages), even though it genuinely happened and is recorded in
            # Delta's own order history. We don't know the EXACT fill price
            # here, so we log using the last-known reference price as a
            # reasonable approximation — better than the trade vanishing
            # from our records entirely. ----
            entry_price = pos["entry_price"]
            if side == "long":
                approx_pnl_pct = (price - entry_price) / entry_price * 100
            else:
                approx_pnl_pct = (entry_price - price) / entry_price * 100
            net_pnl_pct = estimate_net_pnl_pct(approx_pnl_pct, strategy)

            # ---- BUG FIX: try to get the REAL exit price + REAL fee from
            # Delta's own /v2/fills before falling back to the reference-
            # price proxy above (same fix as the bracket-closed paths). ----
            real_exit_price, real_fee_amount = get_real_fill_and_fee(pos["product_id"])
            log_trade_event(
                time=str(now_ist()), symbol=sym, action="CLOSE",
                side=close_side, size=size,
                reason="closed_before_our_order_approx", entry_price=entry_price,
                exit_price=price, approx_gross_pnl_pct=round(approx_pnl_pct, 4),
                approx_net_pnl_pct_after_fees=round(net_pnl_pct, 4),
                fill_method="unknown_exchange_side", strategy=strategy,
                real_exit_price=real_exit_price, real_fee_amount=real_fee_amount,
            )
            if strategy == "B":
                state["last_trade_close_time_b"] = time.time()
            state["position"] = None
            return
        else:
            raise  # unknown error — let the outer loop's error handler log it

    # ---- Abnormal-fill detection (BUG FIX — the ADAUSD incident) ----
    # A real fill can land far from the reference price on a genuinely
    # thin/broken order book (observed: a market close filled at $0.00001
    # when the real price was ~$0.155). Use the ACTUAL fill price for P&L
    # (more accurate than just assuming the reference price filled), and
    # if it's abnormally far from the reference, trip a dedicated breaker
    # so this is impossible to miss and no further trades happen until
    # it's manually reviewed.
    actual_fill_price = _extract_fill_price(resp) or price
    fill_dist_pct = abs(actual_fill_price - price) / price * 100 if price else 0.0
    if fill_dist_pct > MAX_CLOSE_SLIPPAGE_PCT * 2:
        print(f"  [CRITICAL — ABNORMAL FILL DETECTED] {sym} closed at "
              f"{actual_fill_price:.6f}, but the expected/reference price was "
              f"{price:.6f} ({fill_dist_pct:.1f}% away!). This looks like a "
              f"thin/broken order book, not a normal fill. Blocking all new "
              f"entries until this is manually reviewed and reset from the dashboard.")
        state["abnormal_fill_breaker_tripped"] = True
        state["abnormal_fill_detail"] = (
            f"{sym} closed at {actual_fill_price:.6f} vs expected {price:.6f} "
            f"({fill_dist_pct:.1f}% away) at {now_ist()}")

    entry_price = pos["entry_price"]
    if side == "long":
        gross_pnl_pct = (actual_fill_price - entry_price) / entry_price * 100
    else:
        gross_pnl_pct = (entry_price - actual_fill_price) / entry_price * 100

    # Note: daily-loss circuit breaker now compares REAL account balance
    # (checked at the top of each loop, see check_daily_loss_circuit_breaker)
    # instead of summing our own computed P&L here — more robust against
    # any internal calculation bugs.

    net_pnl_pct = estimate_net_pnl_pct(gross_pnl_pct, strategy)

    # ---- BUG FIX: exit_price here is already the real fill (from
    # _extract_fill_price above), but the FEE was still a flat assumed
    # round-trip % (estimate_net_pnl_pct) rather than Delta's real
    # per-fill fee — found in practice to be off by $0.06-$2.36/trade.
    # Fetch the real total fee from /v2/fills and use it if available;
    # falls back to the old %-estimate fee if the lookup fails. We pass
    # the already-correct actual_fill_price as real_exit_price too, so
    # log_trade_event's dollar-figure recompute stays internally consistent. ----
    _, real_fee_amount = get_real_fill_and_fee(pos["product_id"])
    log_trade_event(
        time=str(now_ist()), symbol=sym, action="CLOSE",
        side=close_side, size=size, reason=reason,
        entry_price=entry_price, exit_price=actual_fill_price,
        approx_gross_pnl_pct=round(gross_pnl_pct, 4),
        approx_net_pnl_pct_after_fees=round(net_pnl_pct, 4),
        fill_method=method, order_response=json.dumps(resp),
        strategy=pos.get("strategy", "A"),
        real_exit_price=actual_fill_price, real_fee_amount=real_fee_amount,
    )
    print(f"  CLOSED {sym} due to {reason} (filled via {method}), "
          f"approx gross P&L: {gross_pnl_pct:+.3f}% "
          f"(approx net after fees: {net_pnl_pct:+.3f}%)")
    if strategy == "B":
        state["last_trade_close_time_b"] = time.time()
    if strategy == "A" and pos.get("exchange_safety_stop_active"):
        # The bracket's child orders are normally auto-cancelled by Delta
        # once the position is fully flat, but clean up explicitly too —
        # matches the existing "cancel stray leftover orders" pattern used
        # elsewhere (harmless no-op via open_order_not_found if already gone).
        try:
            cancel_all_orders_for_product(pos["product_id"])
        except Exception as e:
            print(f"  [INFO] {sym}: cleanup of leftover safety-bracket orders "
                  f"failed/no-op ({e}) — usually harmless, they're likely "
                  f"already gone.")
    state["position"] = None


# ============================================================
# Entry logic
# ============================================================

def swing_low(df, idx, lookback=SWING_LOOKBACK):
    start = max(0, idx - lookback)
    return df["low"].iloc[start:idx + 1].min()


def swing_high(df, idx, lookback=SWING_LOOKBACK):
    start = max(0, idx - lookback)
    return df["high"].iloc[start:idx + 1].max()


def look_for_entry_b(state, symbol_data, products):
    """
    Logic B: fast EMA(5/13) crossover + RSI filter + ATR-based SL/TP.
    Only checked if Logic A found nothing this loop. Uses its OWN risk
    settings (LOGIC_B_RISK_PER_TRADE_PCT, LOGIC_B_LEVERAGE) — independent
    from Logic A's sizing.
    """
    last_close_ts = state.get("last_trade_close_time_b")
    if last_close_ts is not None:
        elapsed = time.time() - last_close_ts
        if elapsed < COOLDOWN_SECONDS:
            print(f"  [COOLDOWN] Logic B: {elapsed:.1f}s since last close, need "
                  f"{COOLDOWN_SECONDS}s — skipping entry scan this loop.")
            return False

    print("  --- Logic B entry scan detail (per coin) ---")
    for sym, df in symbol_data.items():
        i = len(df) - 1
        if i < 1:
            continue
        fast_col, slow_col = f"ema_{EMA_FAST}", f"ema_{EMA_SLOW}"
        prev_fast, prev_slow = df[fast_col].iloc[i - 1], df[slow_col].iloc[i - 1]
        curr_fast, curr_slow = df[fast_col].iloc[i], df[slow_col].iloc[i]
        rsi = df["rsi"].iloc[i]
        atr = df["atr"].iloc[i]
        price = df["close"].iloc[i]

        if pd.isna(prev_fast) or pd.isna(curr_slow) or pd.isna(rsi) or pd.isna(atr):
            print(f"  {sym}: Logic B indicators not ready yet")
            continue

        bullish_cross = prev_fast <= prev_slow and curr_fast > curr_slow
        bearish_cross = prev_fast >= prev_slow and curr_fast < curr_slow

        print(f"  {sym}: EMA{EMA_FAST}={curr_fast:.5f} EMA{EMA_SLOW}={curr_slow:.5f} "
              f"bullish_cross={bullish_cross} bearish_cross={bearish_cross} "
              f"RSI={rsi:.1f} ATR={atr:.5f} mode={STRATEGY_MODE_B}")

        side = None
        if STRATEGY_MODE_B == "trend":
            # Continuous re-entry: fires every loop where price/EMAs stay
            # aligned with the trend and RSI confirms — not just at the
            # moment of a fresh cross. Matches the reference bot's default.
            uptrend = price > curr_fast > curr_slow
            downtrend = price < curr_fast < curr_slow
            if uptrend and RSI_LONG_MIN <= rsi <= RSI_LONG_MAX:
                side = "long"
            elif downtrend and RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX:
                side = "short"
            else:
                continue
        else:
            # "crossover" mode — only a FRESH cross counts
            if bullish_cross and RSI_LONG_MIN <= rsi <= RSI_LONG_MAX:
                side = "long"
            elif bearish_cross and RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX:
                side = "short"
            else:
                if bullish_cross or bearish_cross:
                    print(f"    -> SKIP: crossover happened but RSI ({rsi:.1f}) outside "
                          f"the required band for that direction")
                continue

        # real-market "price" decided the direction and ATR above. Now fetch
        # the TESTNET account's own current price to anchor the initial
        # SL/TP/sizing estimate (reduces the gap vs the real fill later).
        exec_price = get_exec_price(sym, price)
        if abs(exec_price - price) / price * 100 > 0.02:
            print(f"    [INFO] Real-market price {price:.5f} vs testnet's own price "
                  f"{exec_price:.5f} — anchoring SL/TP/sizing to the testnet price.")

        stop = exec_price - SL_ATR_MULT * atr if side == "long" else exec_price + SL_ATR_MULT * atr
        target = exec_price + TP_ATR_MULT * atr if side == "long" else exec_price - TP_ATR_MULT * atr
        stop_dist_pct = abs(exec_price - stop) / exec_price * 100

        # NOTE: the fee-aware minimum-target filter was removed for a while
        # per an earlier request, then RE-ADDED below (after the floor-
        # widening, since that's what actually determines the final target).

        print(f"    -> Logic B CONDITIONS MET: {side.upper()} entry")

        # ---- Minimum stop-distance floor (WIDEN, don't skip) ----
        # When ATR is extremely tiny, the raw ATR-based stop can be far
        # tighter than MIN_STOP_DIST_PCT_B — instead of skipping the trade
        # (which the reference bot's code doesn't even check for, and
        # which just means fewer trades for us), we WIDEN the stop to the
        # floor and let position sizing shrink accordingly to keep the
        # same $ risk. This lets genuinely-signaled trades still execute
        # during low-volatility periods, while staying safe from
        # 'immediate_liquidation' rejections (which happen with razor-thin
        # stops forced to near-max leverage).
        if stop_dist_pct < MIN_STOP_DIST_PCT_B:
            print(f"    [INFO] ATR-based stop distance {stop_dist_pct:.3f}% is tinier than "
                  f"the {MIN_STOP_DIST_PCT_B}% floor — widening the stop to the floor "
                  f"(position size will shrink to keep the same $ risk) instead of skipping.")
            stop_dist_pct = MIN_STOP_DIST_PCT_B
            stop = exec_price * (1 - stop_dist_pct / 100) if side == "long" else exec_price * (1 + stop_dist_pct / 100)
            target = (exec_price * (1 + stop_dist_pct * TP_RR_MULT / 100) if side == "long"
                      else exec_price * (1 - stop_dist_pct * TP_RR_MULT / 100))

        # ---- Fee-aware minimum-target filter (RE-ADDED per user request) ----
        # Recompute the FINAL target-move % after the floor-widening above
        # (target is always stop_dist_pct * TP_RR_MULT away in Logic B, by
        # design) and skip this trade if it's smaller than what round-trip
        # taker fees would eat — same protective idea Logic A already has,
        # just using Logic B's taker-fee-based threshold instead of maker.
        final_target_move_pct = stop_dist_pct * TP_RR_MULT
        if final_target_move_pct < MIN_TARGET_PCT_B:
            print(f"    -> SKIP: target move ({final_target_move_pct:.3f}%) is smaller than "
                  f"the fee-aware minimum ({MIN_TARGET_PCT_B:.3f}%) — likely to lose money "
                  f"to fees even if the direction call is right.")
            continue

        if sym not in products:
            print(f"    -> SKIP: {sym} not available on testnet account")
            continue

        available = get_wallet_available_balance()
        if FIXED_SIZE and FIXED_SIZE > 0:
            size = FIXED_SIZE
        else:
            risk_amount = available * (LOGIC_B_RISK_PER_TRADE_PCT / 100)
            max_notional = available * LOGIC_B_LEVERAGE * MARGIN_SAFETY_FACTOR
            notional = min(risk_amount / (stop_dist_pct / 100), max_notional)
            product = products[sym]
            contract_value = float(product.get("contract_value", 1))
            size = max(1, round(notional / (contract_value * exec_price)))

        product = products[sym]

        # ---- BUG FIX: sweep stray leftover orders before entering ----
        # Found in practice: when the create_order call for a NEW entry
        # times out on the network side (client.create_order raising a
        # read-timeout), this script has no way to know whether the
        # order actually landed on the exchange before the timeout — so
        # the NEXT loop just tries again with a brand-new order. Observed
        # result: FIVE separate stray limit orders piling up unfilled on
        # the exchange at the same price (visible in "Open Orders"), each
        # one eating into available margin, until entries started failing
        # with "insufficient_margin" — even though this script's own
        # state showed "no position" the whole time. Cancelling any
        # leftover open orders on this exact product right before placing
        # a fresh one caps the damage at one stray order at a time
        # instead of letting them accumulate indefinitely across loops.
        # ---- BUG FIX v2: if the sweep itself fails, DO NOT proceed ----
        # v1 of this fix (see comment above) still placed a fresh order
        # even when the cancel-sweep failed — but in practice CANCEL
        # requests time out on this testnet just as often as CREATE
        # requests do, so "proceed anyway" meant the sweep was silently
        # doing nothing useful most of the time. Result: 10 stray orders
        # stacked up in a single ~6-minute run despite v1 being deployed.
        # Now: if we can't CONFIRM the slate is clean, skip this symbol
        # for this loop entirely rather than gambling on a duplicate
        # order — the next loop will try the sweep again before any new
        # entry is attempted, so a temporarily-flaky exchange just delays
        # entries instead of stacking orders.
        try:
            cancel_all_orders_for_product(product["id"])
        except Exception as e:
            print(f"    -> SKIP: couldn't confirm no stray orders remain on {sym} "
                  f"({e}) — NOT placing a new order this loop to avoid stacking "
                  f"duplicates. Will retry the sweep next loop.")
            continue

        # ---- BUG FIX: actually VERIFY the sweep worked ----
        # The DELETE call above can "succeed" (no exception) without
        # actually cancelling anything (see its own comment) — so trusting
        # that alone let stray orders keep accumulating. Explicitly
        # re-check via GET before proceeding.
        if not confirm_no_open_orders(product["id"]):
            print(f"    -> SKIP: {sym} still has open orders resting after the sweep "
                  f"— NOT placing a new order this loop to avoid stacking duplicates.")
            continue

        order_side = "buy" if side == "long" else "sell"

        # ---- BUG FIX: limit-first entry (was: instant market order) ----
        # Handoff-doc's known Logic-B fix #1. Logic B previously always used
        # an instant market order for entries, unlike Logic A's "limit-
        # first, market-fallback" approach — paying the (usually pricier)
        # taker fee and taking on slippage risk on every single entry, even
        # though a small limit-offset would often fill just fine within the
        # timeout window. Switched to the same place_order_with_fallback()
        # Logic A uses: try a LIMIT order near the current price first
        # (cheaper maker fee, no slippage), and only fall back to a market
        # order if it doesn't fill within LIMIT_ORDER_TIMEOUT_SECONDS.
        limit_price = (exec_price * (1 - LIMIT_OFFSET_PCT / 100) if side == "long"
                       else exec_price * (1 + LIMIT_OFFSET_PCT / 100))
        # ---- BUG FIX: adopt the trade if create_order times out but the
        # order actually landed (and filled) on the exchange anyway ----
        # Found in practice: client.create_order can raise a client-side
        # read-timeout while the order still gets created (and sometimes
        # filled) server-side. Previously this exception just propagated
        # up to the [LOOP ERROR] handler and the whole entry attempt was
        # forgotten — the next loop's stray-order sweep would eventually
        # cancel an unfilled resting order, but if it had ALREADY FILLED
        # into a real position, that position was left completely
        # unmanaged by this script (protected only by luck, not by any
        # bracket/staircase logic here). Now: on a timeout, immediately
        # check the real exchange position for this product — if one
        # exists, ADOPT it (same logic used at startup) and move on,
        # instead of silently losing track of a live trade.
        try:
            resp, method = place_order_with_fallback(product["id"], order_side, size, limit_price)
        except Exception as e:
            print(f"    [WARN] create_order for {sym} raised an error/timeout ({e}) — "
                  f"checking whether it actually landed on the exchange anyway...")
            try:
                real_pos = client.get_position(product["id"])
                real_size = float(real_pos.get("size", 0)) if isinstance(real_pos, dict) else 0
            except Exception as e2:
                print(f"    [WARN] couldn't verify position for {sym} either ({e2}) — "
                      f"will re-check next loop.")
                continue
            if real_size != 0:
                print(f"    [INFO] Found a REAL position on {sym} (size={real_size}) despite "
                      f"the timeout — the order filled anyway. Adopting it now instead of "
                      f"losing track of it.")
                adopt_real_position(state, sym, product, real_pos, real_size)
            else:
                print(f"    [INFO] No real position found on {sym} — the order likely "
                      f"didn't fill (or is still resting; next loop's sweep will handle it).")
            continue
        entry_order_id = resp.get("id") if isinstance(resp, dict) else None
        if entry_order_id is None:
            # try the nested "result" shape defensively too
            if isinstance(resp, dict) and isinstance(resp.get("result"), dict):
                entry_order_id = resp["result"].get("id")

        # Get the AUTHORITATIVE entry price from the exchange's own position
        # record (single source of truth) — do NOT second-guess it based on
        # a slippage-percentage threshold, since a genuinely large slippage
        # fill was previously discarded as "corrupted data" this way,
        # leaving our internal tracking disconnected from the real position.
        price_for_calc, real_size = get_authoritative_entry_price(product["id"])
        if price_for_calc is None:
            # Fall back to whatever the order response itself reports, and
            # failing that, the testnet exec_price (much closer to reality
            # than the original real-market signal price) — logged clearly
            # so it's visible if this path is ever hit.
            fallback = _extract_fill_price(resp)
            price_for_calc = fallback if fallback is not None else exec_price
            print(f"    [WARN] Couldn't confirm entry price from the exchange's position "
                  f"record — using {price_for_calc:.5f} as a fallback (verify manually).")
        else:
            if abs(price_for_calc - exec_price) / exec_price * 100 > 0.02:
                print(f"    [INFO] Testnet exec_price estimate was {exec_price:.5f}, exchange "
                      f"confirms REAL entry was {price_for_calc:.5f} (verified via position "
                      f"record) — recalculating stop/target from this real entry price.")
            if real_size is not None and real_size != size:
                print(f"    [INFO] Requested/expected size was {size}, but the real resulting "
                      f"position size is {real_size} (likely merged with a small pre-existing "
                      f"residual on this symbol) — using the real size for tracking so P&L "
                      f"and future closes stay accurate.")
                size = real_size

        # ---- Recompute SL/TP from the REAL entry price, with TP forced to
        # a fixed risk:reward ratio off the SL distance (matches reference
        # bot design). Uses stop_dist_pct (which may already have been
        # widened to the MIN_STOP_DIST_PCT_B floor above) rather than raw
        # ATR again, so the floor-widening isn't silently undone here. ----
        sl_dist = price_for_calc * (stop_dist_pct / 100)
        tp_dist = sl_dist * TP_RR_MULT
        if side == "long":
            stop = price_for_calc - sl_dist
            target = price_for_calc + tp_dist
        else:
            stop = price_for_calc + sl_dist
            target = price_for_calc - tp_dist

        # ---- Attach a native bracket order (SL+TP in one exchange-side
        # order) instead of managing exits via our own polling loop ----
        bracket_active = False
        bracket_sl_order_id = None
        try:
            bracket_resp = place_bracket_with_retry(
                product["id"], sym, order_side, size, stop, target, side)
            bracket_active = True
            print(f"    Bracket order attached: SL={stop:.6f} TP={target:.6f}")
            # Look up the bracket's own stop-loss leg id ONCE, right now,
            # and cache it — this is the id trailing-stop updates need
            # later. Fetching it fresh on every trailing update (instead of
            # once here) meant a GET /v2/orders call every ~20s for the
            # life of the trade, which was observed hitting a CloudFront
            # 403 block during a volatile stretch and silently breaking
            # trailing-stop updates for the rest of the trade. Caching it
            # once here removes nearly all of that repeated traffic.
            try:
                bracket_sl_order_id = get_bracket_stop_loss_order_id(product["id"])
            except Exception as e:
                print(f"    [WARN] Could not cache the bracket's stop-loss leg id yet "
                      f"({e}) — will try again if/when trailing needs to update it.")
        except Exception as e:
            print(f"    [WARN] Could not attach bracket order ({e}) — falling back to "
                  f"our own polling-based SL/TP management for this trade instead.")

        state["position"] = {
            "symbol": sym, "product_id": product["id"], "side": side,
            "size": size, "original_size": size,
            "entry_price": price_for_calc, "stop": stop, "target": target,
            "entry_time": str(now_ist()), "strategy": "B",
            "entry_order_id": entry_order_id,
            "milestones_locked": 0, "max_progress": 0.0,
            "bracket_active": bracket_active,
            "bracket_sl_order_id": bracket_sl_order_id,
            "exchange_stop_synced": stop if bracket_active else None,
            "entry_atr": atr, "extreme_price": price_for_calc,
            "r_basis": sl_dist,  # original SL distance in price terms — the "R" unit
            "early_exit_done": False,
        }
        log_trade_event(
            time=str(now_ist()), symbol=sym, action="OPEN",
            side=order_side, size=size, entry_price=price_for_calc,
            stop=stop, target=target, fill_method=method,
            order_response=json.dumps(resp), strategy="B",
        )
        print(f"  OPENED {sym} {side} size={size} @ ~{price:.4f} (filled via {method}) [Logic B]")
        return True

    return False


def check_daily_loss_circuit_breaker(state):
    """
    REAL-BALANCE-BASED kill switch (matches the reference bot's approach) —
    compares actual account balance now vs at the start of the day, instead
    of summing our own computed trade P&Ls. This is more robust: even if a
    corrupted fill-price or a calculation bug produces a wrong P&L number
    internally, the REAL balance is unaffected by our bugs, so the breaker
    can't trip on a fake loss the way the old self-summed version could.
    Resets automatically at UTC day change.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        current_balance = get_wallet_total_balance()
        LATEST_STATE["wallet_balance"] = current_balance
    except Exception as e:
        print(f"  [WARN] Couldn't fetch balance for circuit-breaker check: {e}")
        return True  # fail-open rather than fail-closed on a transient API hiccup

    if state.get("daily_loss_date") != today or state.get("start_of_day_balance") is None:
        state["daily_loss_date"] = today
        state["start_of_day_balance"] = current_balance
        state["daily_breaker_tripped"] = False
        print(f"  [INFO] New trading day (UTC) — start-of-day balance recorded: ${current_balance:.2f}")

    start_balance = state.get("start_of_day_balance", current_balance)
    if start_balance > 0:
        loss_pct = (start_balance - current_balance) / start_balance * 100
    else:
        loss_pct = 0.0

    if loss_pct >= MAX_DAILY_LOSS_PCT:
        if not state.get("daily_breaker_tripped", False):
            print(f"  [CIRCUIT BREAKER TRIPPED] Real balance dropped {loss_pct:.2f}% today "
                  f"(${start_balance:.2f} -> ${current_balance:.2f}), limit is {MAX_DAILY_LOSS_PCT}%.")
        state["daily_breaker_tripped"] = True

    if state.get("daily_breaker_tripped", False):
        print(f"  [CIRCUIT BREAKER] Daily loss limit ({MAX_DAILY_LOSS_PCT}%) hit today "
              f"(start ${start_balance:.2f} -> now ${current_balance:.2f}) — no new entries until tomorrow.")
        return False
    return True


def check_minimum_balance_floor(state):
    """
    Second, independent safety net beyond the daily circuit breaker. The
    daily breaker resets every UTC day, so a slow multi-day erosion (e.g.
    losing 4% several days in a row, never quite tripping the 5% daily
    limit any single day) could still quietly drain the account with
    nothing ever stopping it. This checks the CURRENT balance against the
    balance the very first time this bot ever ran (captured once, persisted
    in the state file, never overwritten again) — if it ever falls below
    MIN_BALANCE_FLOOR_PCT of that original amount, block ALL new entries
    (both logics) until manually reset via the dashboard, same pattern as
    the daily breaker's reset button.
    """
    try:
        current_balance = get_wallet_total_balance()
    except Exception as e:
        print(f"  [WARN] Couldn't fetch balance for minimum-balance floor check: {e}")
        return True  # fail-open on a transient API hiccup, same as the daily breaker

    if state.get("all_time_starting_balance") is None:
        state["all_time_starting_balance"] = current_balance
        print(f"  [INFO] Recorded all-time starting balance: ${current_balance:.2f} "
              f"(new trades will be blocked if balance ever falls below "
              f"{MIN_BALANCE_FLOOR_PCT}% of this, i.e. ${current_balance * MIN_BALANCE_FLOOR_PCT / 100:.2f})")
        return True

    floor_balance = state["all_time_starting_balance"] * (MIN_BALANCE_FLOOR_PCT / 100)
    if current_balance < floor_balance:
        if not state.get("min_balance_breaker_tripped", False):
            print(f"  [MIN-BALANCE BREAKER TRIPPED] Balance (${current_balance:.2f}) has fallen "
                  f"below {MIN_BALANCE_FLOOR_PCT}% of the all-time starting balance "
                  f"(${state['all_time_starting_balance']:.2f}) — floor is ${floor_balance:.2f}.")
        state["min_balance_breaker_tripped"] = True

    if state.get("min_balance_breaker_tripped", False):
        print(f"  [MIN-BALANCE BREAKER] No new entries until this is manually reset from "
              f"the dashboard — current ${current_balance:.2f} is below the ${floor_balance:.2f} floor.")
        return False
    return True


def look_for_entry_a(state, symbol_data, products):
    """Logic A: 200 EMA + VWAP + CVD confluence (existing strategy)."""
    print("  --- Entry scan detail (per coin) ---")
    for sym, df in symbol_data.items():
        i = len(df) - 1
        row = df.iloc[i]
        price = row["close"]
        ema = row[f"ema_{EMA_PERIOD}"]
        vwap = row["vwap"]
        if pd.isna(ema) or pd.isna(vwap):
            print(f"  {sym}: EMA/VWAP not ready yet (still warming up)")
            continue

        bullish = price > ema
        bearish = price < ema
        dist_from_vwap_pct = abs(price - vwap) / price * 100
        near_vwap = dist_from_vwap_pct <= VWAP_PROXIMITY_PCT
        trend_label = "BULLISH (price>EMA200)" if bullish else "BEARISH (price<EMA200)"
        cvd_now = df["cvd"].iloc[i]
        cvd_rising_flag = cvd_rising(df, i, CVD_LOOKBACK)
        cvd_falling_flag = cvd_falling(df, i, CVD_LOOKBACK)

        print(f"  {sym}: price={price:.5f} EMA200={ema:.5f} ({trend_label}) "
              f"VWAP={vwap:.5f} (dist={dist_from_vwap_pct:.3f}%, need<={VWAP_PROXIMITY_PCT}%) "
              f"CVD_now={cvd_now:.1f} rising={cvd_rising_flag} falling={cvd_falling_flag}")

        if not near_vwap:
            print(f"    -> SKIP: price not close enough to VWAP")
            continue

        side = None
        if bullish and cvd_rising_flag:
            side = "long"
            stop = swing_low(df, i) * (1 - STOP_BUFFER_PCT / 100)
            stop_dist_pct = (price - stop) / price * 100
        elif bearish and cvd_falling_flag:
            side = "short"
            stop = swing_high(df, i) * (1 + STOP_BUFFER_PCT / 100)
            stop_dist_pct = (stop - price) / price * 100
        else:
            print(f"    -> SKIP: trend direction and CVD momentum don't agree "
                  f"(need bullish+CVD_rising OR bearish+CVD_falling)")
            continue

        if stop_dist_pct <= 0:
            print(f"    -> SKIP: invalid stop distance")
            continue

        # ---- Minimum stop-distance floor (WIDEN, don't skip) ----
        # See MIN_STOP_DIST_PCT_A definition above for the full story.
        # Widening here (rather than skipping) keeps every genuinely-
        # signaled setup tradeable, same as Logic B's identical fix —
        # just with position size shrinking afterward to keep the same
        # $ risk, instead of the size ballooning toward max-leverage on
        # a razor-tight, noise-level stop.
        if stop_dist_pct < MIN_STOP_DIST_PCT_A:
            print(f"    [INFO] swing-based stop distance {stop_dist_pct:.3f}% is tinier than "
                  f"the {MIN_STOP_DIST_PCT_A}% floor — widening the stop to the floor "
                  f"(position size will shrink to keep the same $ risk) instead of skipping.")
            stop_dist_pct = MIN_STOP_DIST_PCT_A
            stop = (price * (1 - stop_dist_pct / 100) if side == "long"
                    else price * (1 + stop_dist_pct / 100))

        target_move_pct = stop_dist_pct * RISK_REWARD_MULT
        if target_move_pct < MIN_TARGET_PCT:
            print(f"    -> SKIP: target move {target_move_pct:.3f}% < fee-aware "
                  f"minimum {MIN_TARGET_PCT:.3f}%")
            continue

        print(f"    -> ALL 3 CONDITIONS MET: trend={trend_label}, "
              f"near_vwap=True, cvd_confirms=True. Proceeding with {side.upper()} entry.")

        if sym not in products:
            print(f"    -> SKIP: {sym} has a valid signal but isn't available on "
                  f"the TESTNET account (production-only symbol) — can't place an "
                  f"order for it here. Consider removing it from SYMBOLS_TO_WATCH.")
            continue

        # ---- Liquidation-distance safety sanity check ----
        approx_liq_dist_pct, actual_stop_dist_pct = get_liquidation_distance_pct(
            price, stop, side, LEVERAGE)
        print(f"    Safety check: stop is {actual_stop_dist_pct:.2f}% away, "
              f"approx liquidation is ~{approx_liq_dist_pct:.2f}% away at {LEVERAGE}x leverage.")
        if actual_stop_dist_pct >= approx_liq_dist_pct * 0.7:
            print(f"    -> SKIP: stop-loss is too close to estimated liquidation distance "
                  f"(within 70%) — too risky at this leverage, skipping this setup.")
            continue

        target = (price * (1 + target_move_pct / 100) if side == "long"
                  else price * (1 - target_move_pct / 100))

        available = get_wallet_available_balance()
        product = products[sym]
        if FIXED_SIZE and FIXED_SIZE > 0:
            size = FIXED_SIZE
        else:
            risk_amount = available * (RISK_PER_TRADE_PCT / 100)
            max_notional = available * LEVERAGE * MARGIN_SAFETY_FACTOR
            notional = min(risk_amount / (stop_dist_pct / 100), max_notional)

            contract_value = float(product.get("contract_value", 1))
            size = max(1, round(notional / (contract_value * price)))

        order_side = "buy" if side == "long" else "sell"
        limit_price = (price * (1 - LIMIT_OFFSET_PCT / 100) if side == "long"
                       else price * (1 + LIMIT_OFFSET_PCT / 100))

        # Remember the stop_dist_pct that position SIZE was actually sized
        # for (via risk_amount / stop_dist_pct above) — needed after order
        # placement to detect if a large slippage fill silently made the
        # REAL risk much bigger than what this size was computed for.
        intended_stop_dist_pct = stop_dist_pct

        # ---- BUG FIX: sweep stray leftover orders before entering ----
        # Same fix as Logic B — see its identical comment for the full
        # story (a create_order timeout can leave an order actually
        # resting on the exchange while this script believes it failed;
        # without this sweep, retries stack up multiple orphaned orders
        # and eventually exhaust available margin).
        # ---- BUG FIX v2: if the sweep fails, DO NOT proceed ----
        # Same fix as Logic B — see its identical comment for the full
        # story. v1 ("proceed anyway" on sweep failure) still let orders
        # stack up, because CANCEL requests fail on this testnet just as
        # often as CREATE requests do — "proceed anyway" made the sweep
        # a no-op exactly when it mattered most.
        try:
            cancel_all_orders_for_product(product["id"])
        except Exception as e:
            print(f"    -> SKIP: couldn't confirm no stray orders remain on {sym} "
                  f"({e}) — NOT placing a new order this loop to avoid stacking "
                  f"duplicates. Will retry the sweep next loop.")
            continue

        # ---- BUG FIX: actually VERIFY the sweep worked (see Logic B's
        # identical comment for the full story — the DELETE call could
        # silently "succeed" without cancelling anything). ----
        if not confirm_no_open_orders(product["id"]):
            print(f"    -> SKIP: {sym} still has open orders resting after the sweep "
                  f"— NOT placing a new order this loop to avoid stacking duplicates.")
            continue

        # ---- BUG FIX: adopt the trade if create_order times out but the
        # order actually landed (and filled) on the exchange anyway ----
        # Same fix as Logic B — see its identical comment for the full
        # story. Without this, a real filled position from a timed-out
        # create_order call was silently forgotten by this script.
        try:
            resp, method = place_order_with_fallback(product["id"], order_side, size, limit_price)
        except Exception as e:
            print(f"    [WARN] create_order for {sym} raised an error/timeout ({e}) — "
                  f"checking whether it actually landed on the exchange anyway...")
            try:
                real_pos = client.get_position(product["id"])
                real_size = float(real_pos.get("size", 0)) if isinstance(real_pos, dict) else 0
            except Exception as e2:
                print(f"    [WARN] couldn't verify position for {sym} either ({e2}) — "
                      f"will re-check next loop.")
                continue
            if real_size != 0:
                print(f"    [INFO] Found a REAL position on {sym} (size={real_size}) despite "
                      f"the timeout — the order filled anyway. Adopting it now instead of "
                      f"losing track of it.")
                adopt_real_position(state, sym, product, real_pos, real_size)
            else:
                print(f"    [INFO] No real position found on {sym} — the order likely "
                      f"didn't fill (or is still resting; next loop's sweep will handle it).")
            continue

        entry_order_id = resp.get("id") if isinstance(resp, dict) else None
        if entry_order_id is None:
            try:
                entry_order_id = resp["result"].get("id")
            except Exception:
                entry_order_id = None

        # Get the AUTHORITATIVE entry price from the exchange's own position
        # record rather than guessing based on a slippage-percentage
        # threshold (see Logic B's identical fix for why — a genuinely
        # large slippage fill was previously discarded as "bad data" this
        # way, leaving our tracking disconnected from the real position).
        price_for_calc, real_size = get_authoritative_entry_price(product["id"])
        if price_for_calc is None:
            fallback = _extract_fill_price(resp)
            price_for_calc = fallback if fallback is not None else price
            print(f"    [WARN] Couldn't confirm entry price from the exchange's position "
                  f"record — using {price_for_calc:.5f} as a fallback (verify manually).")
        else:
            if real_size is not None and real_size != size:
                print(f"    [INFO] Requested/expected size was {size}, but the real resulting "
                      f"position size is {real_size} (likely merged with a small pre-existing "
                      f"residual on this symbol) — using the real size for tracking so P&L "
                      f"and future closes stay accurate.")
                size = real_size
            if abs(price_for_calc - price) / price * 100 > 0.02:
                print(f"    [INFO] Signal price was {price:.5f}, exchange confirms REAL entry "
                      f"was {price_for_calc:.5f} (verified via position record, not guessed) "
                      f"— recalculating stop/target from this real entry price.")
                if side == "long":
                    stop = swing_low(df, i) * (1 - STOP_BUFFER_PCT / 100)
                    stop_dist_pct = (price_for_calc - stop) / price_for_calc * 100
                    target = price_for_calc * (1 + (stop_dist_pct * RISK_REWARD_MULT) / 100)
                else:
                    stop = swing_high(df, i) * (1 + STOP_BUFFER_PCT / 100)
                    stop_dist_pct = (stop - price_for_calc) / price_for_calc * 100
                    target = price_for_calc * (1 - (stop_dist_pct * RISK_REWARD_MULT) / 100)

                    # ---- Slippage risk-ballooning safety check (BUG FIX) ----
                # Position SIZE above was computed assuming `intended_stop_dist_pct`
                # (a small, tight risk % relative to the SIGNAL price). The swing-
                # based stop is an ABSOLUTE price level that doesn't move just
                # because the real fill came in far from the signal price — so if
                # a large slippage/gap happens on entry (testnet price anomalies
                # have been observed at 1-3%+), the REAL stop_dist_pct relative to
                # the ACTUAL entry can balloon to several times what was intended,
                # meaning the real dollar-risk on this trade is now much bigger
                # than RISK_PER_TRADE_PCT of the account was ever meant to allow —
                # this actually happened in practice (0.51% intended -> 3.41%
                # realized, a 6.7x risk blowup, contributing to a real ~10%
                # single-trade account loss that tripped the daily circuit
                # breaker). Rather than holding a position whose real risk no
                # longer matches what it was sized for, close it immediately here.
                if stop_dist_pct > intended_stop_dist_pct * MAX_SLIPPAGE_RISK_MULT:
                    print(f"    [WARN] Real stop distance ({stop_dist_pct:.2f}%) is "
                          f"{stop_dist_pct / intended_stop_dist_pct:.1f}x the intended "
                          f"({intended_stop_dist_pct:.2f}%) this position's size was "
                          f"calculated for — the slippage on this fill made the real "
                          f"risk far bigger than sized for. Closing immediately instead "
                          f"of holding an oversized-risk position.")
                    close_side = "sell" if side == "long" else "buy"
                    try:
                        close_resp, close_method = place_order_with_fallback(
                            product["id"], close_side, size, price_for_calc, reduce_only=True)
                        exit_price = _extract_fill_price(close_resp) or price_for_calc
                    except Exception as e:
                        print(f"    [WARN] Emergency close order failed ({e}) — falling back "
                              f"to normal position tracking with the wide stop instead; "
                              f"monitor this trade manually.")
                        exit_price = None
                    if exit_price is not None:
                        if side == "long":
                            emergency_pnl_pct = (exit_price - price_for_calc) / price_for_calc * 100
                        else:
                            emergency_pnl_pct = (price_for_calc - exit_price) / price_for_calc * 100
                        net_pnl_pct = estimate_net_pnl_pct(emergency_pnl_pct, "A")
                        log_trade_event(
                            time=str(now_ist()), symbol=sym, action="CLOSE",
                            side=close_side, size=size, reason="aborted_slippage_risk",
                            entry_price=price_for_calc, exit_price=exit_price,
                            approx_gross_pnl_pct=round(emergency_pnl_pct, 4),
                            approx_net_pnl_pct_after_fees=round(net_pnl_pct, 4),
                            fill_method=close_method, strategy="A",
                        )
                        print(f"    Closed immediately due to slippage risk. Approx P&L: "
                              f"{emergency_pnl_pct:+.3f}% (net after fees: {net_pnl_pct:+.3f}%)")
                        continue  # skip attaching a bracket / opening state["position"] below

        # ---- Safety-net exchange-side stop (NEW) ----
        # Logic A's staircase-trailing stop previously lived ONLY inside
        # this script's own polling loop — if the process/service is ever
        # down (deploy restart, Render free-tier sleep, crash) while a
        # position is open, nothing would enforce the stop until the loop
        # comes back. Attaching a bracket order here puts a hard stop on
        # Delta's own server as a safety net, independent of this script
        # staying alive. We reuse Logic B's proven bracket-order helpers.
        # NOTE: unlike Logic B, we do NOT set "bracket_active" (that flag
        # routes position management to manage_bracket_position_b(), which
        # has ATR-trailing logic specific to Logic B) — this is purely a
        # background safety net; our own staircase-trailing loop still
        # drives the "real" stop level, and pushes updates to this
        # exchange-side bracket via edit_bracket_order() whenever it moves.
        exchange_safety_stop_active = False
        bracket_sl_order_id = None
        try:
            place_bracket_with_retry(product["id"], sym, order_side, size, stop, target, side)
            exchange_safety_stop_active = True
            print(f"    Exchange-side safety-net bracket attached: SL={stop:.6f} TP={target:.6f}")
            # ---- BUG FIX: cache the bracket's OWN stop-loss leg id now,
            # same as Logic B already does. Without this, later staircase
            # updates tried to edit the bracket using entry_order_id (the
            # original MARKET/LIMIT entry order) — which becomes 'closed'
            # the moment it fills, so Delta rejects edits to it with
            # 'open_order_not_found' (observed in practice: every staircase
            # move after entry silently failed to reach the exchange). ----
            try:
                bracket_sl_order_id = get_bracket_stop_loss_order_id(product["id"])
            except Exception as e:
                print(f"    [WARN] Could not cache the bracket's stop-loss leg id yet "
                      f"({e}) — will try again if/when the staircase needs to update it.")
        except Exception as e:
            # ---- BUG FIX (was a real incident) ----
            # Previously this just printed a warning and let the trade run
            # with NO exchange-side protection at all, relying solely on
            # this script's own polling loop. That combined with a
            # separate close-order bug on one occasion to leave a position
            # completely unprotected for an extended period — before that
            # second bug is even considered, holding any position with zero
            # exchange-side stop whenever this attach fails is too risky to
            # accept silently. Close the position immediately instead.
            print(f"    [WARN] Could not attach safety-net bracket ({e}) — this position "
                  f"would have NO exchange-side stop protection at all if held. "
                  f"Closing it immediately instead of accepting that risk.")
            close_side = "sell" if side == "long" else "buy"
            try:
                close_resp, close_method = place_order_with_fallback(
                    product["id"], close_side, size, price_for_calc, reduce_only=True)
                exit_price = _extract_fill_price(close_resp) or price_for_calc
            except Exception as close_e:
                print(f"    [WARN] Emergency close after failed bracket attach also "
                      f"failed ({close_e}) — this position is now unprotected and "
                      f"UNTRACKED locally; check the exchange manually. Falling back "
                      f"to normal tracking with polling-only stop as a last resort.")
                exit_price = None
            if exit_price is not None:
                if side == "long":
                    abort_pnl_pct = (exit_price - price_for_calc) / price_for_calc * 100
                else:
                    abort_pnl_pct = (price_for_calc - exit_price) / price_for_calc * 100
                net_pnl_pct = estimate_net_pnl_pct(abort_pnl_pct, "A")
                log_trade_event(
                    time=str(now_ist()), symbol=sym, action="CLOSE",
                    side=close_side, size=size, reason="aborted_no_bracket_protection",
                    entry_price=price_for_calc, exit_price=exit_price,
                    approx_gross_pnl_pct=round(abort_pnl_pct, 4),
                    approx_net_pnl_pct_after_fees=round(net_pnl_pct, 4),
                    fill_method=close_method, strategy="A",
                )
                print(f"    Closed immediately (no bracket protection available). "
                      f"Approx P&L: {abort_pnl_pct:+.3f}% (net after fees: {net_pnl_pct:+.3f}%)")
                return True  # took action this loop; skip opening state["position"] below

        state["position"] = {
            "symbol": sym, "product_id": product["id"], "side": side,
            "size": size, "original_size": size,
            "entry_price": price_for_calc, "stop": stop, "target": target,
            "entry_time": str(now_ist()), "milestones_locked": 0,
            "max_progress": 0.0, "strategy": "A",
            "entry_order_id": entry_order_id,
            "exchange_safety_stop_active": exchange_safety_stop_active,
            "exchange_stop_synced": stop if exchange_safety_stop_active else None,
            "bracket_sl_order_id": bracket_sl_order_id,
        }
        log_trade_event(
            time=str(now_ist()), symbol=sym, action="OPEN",
            side=order_side, size=size, entry_price=price_for_calc,
            stop=stop, target=target, fill_method=method,
            order_response=json.dumps(resp), strategy="A",
        )
        print(f"  OPENED {sym} {side} size={size} @ ~{price:.4f} (filled via {method}) [Logic A]")
        return True  # only ONE trade at a time

    return False


# ============================================================
# Main loop — wrapped so it NEVER stops on its own
# ============================================================

def compute_market_regime(symbol_data_a):
    """
    ---- NEW FEATURE: market-regime indicator ----
    Surfaces, every loop, which strategy CURRENT market conditions favor
    right now — Logic A (mean-reversion, wants price close to VWAP) or
    Logic B (trend-following, thrives when price is running away from
    VWAP with real momentum) — regardless of which one is actually
    running. The algo already computes VWAP-distance every loop for its
    own entry-scan; this just surfaces that same number to the dashboard
    instead of leaving it buried in the console log ("algo se kuch
    chhupa nahi hai" — nothing the algo already knows should stay hidden
    from the person watching it).

    Uses the SAME VWAP_PROXIMITY_PCT threshold Logic A's own entry-check
    already uses: price close to VWAP = range-bound = Logic A's natural
    environment. Price running far from VWAP = a real directional trend
    — the exact condition observed in practice keeping Logic A flat all
    day, while being exactly the kind of move Logic B's EMA-crossover is
    built to catch. This is a simple, cheap heuristic (not a guarantee)
    — it's meant to explain WHY Logic A is or isn't finding setups, not
    to replace either strategy's own entry logic.
    """
    dists = {}
    for sym in SYMBOLS_TO_WATCH:
        df = symbol_data_a.get(sym)
        if df is None or len(df) == 0:
            continue
        latest = df.iloc[-1]
        price, vwap = latest["close"], latest["vwap"]
        if price:
            dists[sym] = abs(price - vwap) / price * 100

    if not dists:
        LATEST_STATE["market_regime"] = None
        return

    avg_dist = sum(dists.values()) / len(dists)

    if avg_dist <= VWAP_PROXIMITY_PCT:
        favors, label = "A", "Range-bound / price near VWAP"
    elif avg_dist >= VWAP_PROXIMITY_PCT * 2:
        favors, label = "B", "Strong trend / price far from VWAP"
    else:
        favors, label = "NEUTRAL", "Mixed / transitioning"

    LATEST_STATE["market_regime"] = {
        "favors": favors, "label": label,
        "avg_vwap_dist_pct": round(avg_dist, 3),
        "threshold_pct": VWAP_PROXIMITY_PCT,
        "per_symbol": {s: round(d, 3) for s, d in dists.items()},
    }


def run_one_loop_iteration(state, products):
    symbol_data_a = {}  # Logic A: 15-min candles
    symbol_data_b = {}  # Logic B: 1-min candles

    mode = LOGIC_MODE["active"]
    active_strategy = state["position"].get("strategy") if state["position"] else None
    # ---- Logic A data is now ALWAYS fetched (cheap, public, read-only
    # endpoint), regardless of active mode — needed for the market-regime
    # indicator above, which shows whether conditions favor A or B no
    # matter which is actually running. This does NOT change which logic
    # actually takes trades — that stays strictly gated by `mode` further
    # down in this function. ----
    need_a = True
    need_b = mode in ("B", "BOTH") or active_strategy == "B"

    # ---- Logic A data (15-min) — watches all of SYMBOLS_TO_WATCH ----
    if need_a:
        for sym in SYMBOLS_TO_WATCH:
            try:
                df_a = fetch_candles(sym, hours=LOGIC_A_LOOKBACK_HOURS, resolution=LOGIC_A_RESOLUTION)
            except Exception as e:
                print(f"  [ERROR] {sym}: Logic A (15m) fetch failed ({e}) — skipping")
                df_a = None
            if df_a is not None and len(df_a) >= EMA_PERIOD + 5:
                df_a = compute_ema(df_a, period=EMA_PERIOD)
                df_a = compute_vwap(df_a)
                df_a = compute_cvd(df_a)
                symbol_data_a[sym] = df_a

    compute_market_regime(symbol_data_a)

    # ---- Logic B data (1-min) — watches only LOGIC_B_SYMBOLS (BTC/ETH) ----
    if need_b:
        for sym in LOGIC_B_SYMBOLS:
            try:
                df_b = fetch_candles(sym, hours=LOOKBACK_HOURS, resolution="1m")
            except Exception as e:
                print(f"  [ERROR] {sym}: Logic B (1m) fetch failed ({e}) — skipping")
                df_b = None
            min_len_b = max(SWING_LOOKBACK, CVD_LOOKBACK, EMA_SLOW, RSI_PERIOD, ATR_PERIOD) + 5
            if df_b is not None and len(df_b) >= min_len_b:
                df_b = compute_ema(df_b, period=EMA_FAST)
                df_b = compute_ema(df_b, period=EMA_SLOW)
                df_b = compute_rsi(df_b, period=RSI_PERIOD)
                df_b = compute_atr(df_b, period=ATR_PERIOD)
                symbol_data_b[sym] = df_b

    # ---- Always manage an already-open position first, regardless of the
    # daily circuit breaker. (BUG FIX: this used to be gated behind the
    # circuit-breaker check below, which meant that if the breaker tripped
    # while a position was open, local monitoring — staircase/ATR trailing,
    # stop/target checks, and the exchange-bracket-external-close detection
    # — would ALL silently stop until the breaker reset. The exchange-side
    # bracket still protects the position during that gap, but trailing
    # stops stop tightening and a real external close wouldn't be noticed
    # locally until the breaker reset. New entries are the only thing that
    # should ever be blocked by the circuit breaker.) ----
    if state["position"] is not None:
        active_strategy = state["position"].get("strategy", "A")
        symbol_data_for_management = symbol_data_a if active_strategy == "A" else symbol_data_b
        manage_open_position(state, symbol_data_for_management)

    if not check_daily_loss_circuit_breaker(state):
        save_state(state)
        return

    if not check_minimum_balance_floor(state):
        save_state(state)
        return

    if state.get("abnormal_fill_breaker_tripped", False):
        print(f"  [ABNORMAL-FILL BREAKER] No new entries — an abnormal fill was "
              f"detected ({state.get('abnormal_fill_detail', 'see earlier log')}). "
              f"Reset manually from the dashboard after reviewing what happened.")
        save_state(state)
        return

    # ============================================================
    # GUARANTEE: only ONE trade is ever open system-wide, across BOTH
    # Logic A and Logic B combined. This works because:
    #   1. There is a single state["position"] slot (not two separate ones)
    #   2. Entries are only attempted when state["position"] is None
    #   3. Logic B is only checked if Logic A found NOTHING this loop
    #      (look_for_entry_b only runs when took_trade is False from A)
    #   4. If either logic opens a trade, it returns True immediately,
    #      so the other logic is never even checked that same loop
    # ============================================================
    if state["position"] is None:
        mode = LOGIC_MODE["active"]
        took_trade = False
        if mode in ("A", "BOTH"):
            took_trade = look_for_entry_a(state, symbol_data_a, products)
        if not took_trade and mode in ("B", "BOTH"):
            took_trade = look_for_entry_b(state, symbol_data_b, products)
        if not took_trade:
            print(f"  No tradeable setup this loop (mode={mode}) — staying flat.")

    save_state(state)


LATEST_STATE = {"position": None, "equity_note": None, "market_regime": None}


def main_loop(stop_event=None):
    def should_stop():
        return stop_event is not None and stop_event.is_set()

    state = load_state()

    # ---- BUG FIX: restore the last-chosen Active Logic mode (A/B/BOTH)
    # from the saved state. Previously LOGIC_MODE only lived in memory,
    # so any restart silently reverted it to "BOTH" — meaning a user who
    # explicitly selected "Logic A Only" could have Logic B quietly start
    # trading again after any restart, without them re-selecting it. ----
    saved_mode = state.get("logic_mode")
    if saved_mode in ("A", "B", "BOTH"):
        LOGIC_MODE["active"] = saved_mode
        print(f"Restored Active Logic mode from saved state: {saved_mode}")

    products = get_product_map()
    print(f"Loaded products: {list(products.keys())}")
    print(f"Fee-aware minimum target: {MIN_TARGET_PCT:.3f}% "
          f"(maker round-trip {ROUND_TRIP_FEE_PCT:.3f}% x {SAFETY_MARGIN_MULT} safety margin)")

    print(f"\nSetting leverage to {LEVERAGE}x for all watched products...")
    set_leverage_for_all(products)

    state = reconcile_with_exchange(state, products)
    save_state(state)
    LATEST_STATE["position"] = state["position"]

    while not should_stop():
        print(f"\n[{now_ist()}] Loop start. Position: {state['position']}")
        try:
            run_one_loop_iteration(state, products)
        except Exception as e:
            # CRITICAL: never let one bad loop kill the whole script.
            print(f"  [LOOP ERROR] {e}")
            traceback.print_exc()
            print("  Continuing to next loop despite the error above "
                  "(script only stops on manual Stop/Ctrl+C).")

        LATEST_STATE["position"] = state["position"]
        if state["position"] is None:
            LATEST_STATE["current_price"] = None
            LATEST_STATE["live_pnl_pct"] = None

        for _ in range(LOOP_INTERVAL_SECONDS):
            if should_stop():
                break
            time.sleep(1)

    print("\n[Algo stopped by user]")


if __name__ == "__main__":
    while True:
        try:
            main_loop()
            break  # main_loop only returns normally if stop_event was set (not used in CLI mode)
        except KeyboardInterrupt:
            print("\nStopped by user (Ctrl+C). State saved.")
            break
        except Exception as e:
            # Even a crash OUTSIDE the per-loop try/except (e.g. during
            # startup reconciliation) will restart the whole script rather
            # than exiting, per the "never stop until I stop it" requirement.
            print(f"\n[FATAL ERROR, RESTARTING IN 10s] {e}")
            traceback.print_exc()
            time.sleep(10)
