#!/usr/bin/env bash
# =============================================================================
#  GCP VM Deployment Script — v18 Paper Trading
# =============================================================================
#  Usage (run ON the GCP VM):
#    1. SSH into your VM:   gcloud compute ssh <VM_NAME> --zone <ZONE>
#    2. Upload this repo:   git clone https://github.com/boriso-ctrl/Financial-Algorithms.git
#    3. Run this script:    cd Financial-Algorithms && bash scripts/deploy_gcp.sh
#    4. Add your API keys:  nano ~/Financial-Algorithms/.env
#
#  This script:
#    - Installs Python 3.11+ and pip
#    - Creates a venv and installs dependencies
#    - Sets up a daily cron job (16:35 ET / 20:35 UTC)
#    - Creates a runner wrapper with logging
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$REPO_DIR/.venv"
LOG_DIR="$REPO_DIR/logs"
RUNNER="$REPO_DIR/scripts/run_v18_daily.sh"

echo "============================================"
echo "  v18 Paper Trading — GCP VM Setup"
echo "============================================"
echo "  Repo: $REPO_DIR"
echo ""

# --- 1. System packages ---
echo "[1/5] Installing system dependencies ..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip python3-venv cron tzdata > /dev/null 2>&1
echo "  Done."

# --- 2. Set timezone to US/Eastern ---
echo "[2/5] Setting timezone to US/Eastern ..."
sudo timedatectl set-timezone America/New_York 2>/dev/null || \
  sudo ln -sf /usr/share/zoneinfo/America/New_York /etc/localtime
echo "  Timezone: $(date +%Z)"

# --- 3. Python venv ---
echo "[3/5] Creating Python venv ..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install numpy pandas yfinance alpaca-py python-dotenv -q
echo "  Installed: numpy, pandas, yfinance, alpaca-py, python-dotenv"

# --- 4. Create daily runner script ---
echo "[4/5] Creating daily runner script ..."
mkdir -p "$LOG_DIR"

cat > "$RUNNER" << 'RUNNER_EOF'
#!/usr/bin/env bash
# Daily runner for v18 paper trading — called by cron
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$REPO_DIR/.venv"
LOG_DIR="$REPO_DIR/logs"
DATE=$(date +%Y-%m-%d)
LOG_FILE="$LOG_DIR/v18_${DATE}.log"

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

echo "=== v18 run @ $(date) ===" >> "$LOG_FILE"
python scripts/paper_trading_v18.py --execute >> "$LOG_FILE" 2>&1
EXIT_CODE=$?
echo "=== exit code: $EXIT_CODE ===" >> "$LOG_FILE"

# Keep only last 90 days of logs
find "$LOG_DIR" -name "v18_*.log" -mtime +90 -delete 2>/dev/null || true
RUNNER_EOF

chmod +x "$RUNNER"
echo "  Runner: $RUNNER"

# --- 5. Install cron job ---
echo "[5/5] Installing cron job (16:35 ET daily, Mon-Fri) ..."

# Remove any existing v18 cron entry, then add new one
CRON_LINE="35 16 * * 1-5 $RUNNER"
(crontab -l 2>/dev/null | grep -v "run_v18_daily" ; echo "$CRON_LINE") | crontab -
echo "  Cron: $CRON_LINE"

# --- 6. Remind about .env ---
echo ""
echo "============================================"
echo "  SETUP COMPLETE"
echo "============================================"
echo ""
if [ ! -f "$REPO_DIR/.env" ]; then
    cp "$REPO_DIR/.env.example" "$REPO_DIR/.env" 2>/dev/null || true
    echo "  ACTION REQUIRED: Edit .env with your Alpaca keys:"
    echo "    nano $REPO_DIR/.env"
else
    echo "  .env found. Verify your Alpaca keys are set:"
    echo "    grep ALPACA $REPO_DIR/.env"
fi
echo ""
echo "  Test it now:"
echo "    source $VENV_DIR/bin/activate"
echo "    python scripts/paper_trading_v18.py --dry-run"
echo ""
echo "  Cron will run Mon-Fri at 16:35 ET (after market close)."
echo "  Logs: $LOG_DIR/v18_YYYY-MM-DD.log"
echo ""
