"""Quick connectivity test — run before using paper_trader.py"""
import os, sys
from pathlib import Path

# Load .env
env_path = Path(__file__).resolve().parents[2] / '.env'
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        k, _, v = line.partition('=')
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from alpaca.trading.client import TradingClient

key    = os.environ['ALPACA_API_KEY']
secret = os.environ['ALPACA_SECRET_KEY']
client = TradingClient(key, secret, paper=True)

acct = client.get_account()
print(f"Account status : {acct.status}")
print(f"Equity         : ${float(acct.equity):,.2f}")
print(f"Cash           : ${float(acct.cash):,.2f}")
print(f"Buying power   : ${float(acct.buying_power):,.2f}")

positions = client.get_all_positions()
print(f"Open positions : {len(positions)}")
for p in positions:
    print(f"  {p.symbol}  qty={p.qty}  avg_entry={p.avg_entry_price}  "
          f"unrealized_PnL=${float(p.unrealized_pl):,.2f}")

if not positions:
    print("  (none)")
