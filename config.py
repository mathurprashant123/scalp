"""
config.py — SAFE to commit to GitHub. Contains NO secrets.

API keys are read from environment variables instead of being hardcoded:
  - On your LOCAL PC: create a file named ".env" (see .env.example below)
    in this same folder, and it will be loaded automatically.
  - On RENDER: set DELTA_API_KEY and DELTA_API_SECRET in the dashboard's
    "Environment" tab (Settings -> Environment) -- Render injects them as
    real environment variables, no .env file needed there.

This is why config.py itself can safely be pushed to GitHub -- the actual
secret values never live in this file or in version control.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads a local .env file if present (does nothing on Render,
                # since Render provides env vars directly -- load_dotenv()
                # simply finds no .env file there and continues normally)

API_KEY = os.getenv("DELTA_API_KEY", "")
API_SECRET = os.getenv("DELTA_API_SECRET", "")

# Testnet-India base URL -- used for ORDER PLACEMENT, balance, leverage
# (fake money, safe to place real orders against)
BASE_URL = os.getenv("DELTA_BASE_URL", "https://cdn-ind.testnet.deltaex.org")

# REAL (production) exchange base URL -- used ONLY for fetching live market
# data (candles), since testnet's price feed doesn't reflect real market
# movement. This is a PUBLIC, read-only endpoint -- no API key needed and
# no real money risk, since we never place orders here.
REAL_DATA_BASE_URL = os.getenv("DELTA_REAL_DATA_BASE_URL", "https://api.india.delta.exchange")

if not API_KEY or not API_SECRET:
    print("[WARN] DELTA_API_KEY / DELTA_API_SECRET are not set. "
          "Locally: create a .env file (see .env.example). "
          "On Render: set them under Settings -> Environment.")
