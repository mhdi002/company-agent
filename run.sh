#!/usr/bin/env bash
# ProposalAgent — one-command install + launch (Linux / macOS).
#
#   ./run.sh                 install what is missing, start the app, open the browser
#   ./run.sh --port 9000     custom port            ./run.sh --host 0.0.0.0   listen on all interfaces
#   ./run.sh --offline       demo mode (bundled fictional companies, no internet needed)
#   ./run.sh --background    run detached (logs/app.out, PID in logs/app.pid)   ./run.sh --stop
#   ./run.sh --update        git pull first         ./run.sh --test          run the test suite after install
#   ./run.sh --smoke-train   also train the tiny model end-to-end on CPU (~15 min) so the brain is "model"
#   ./run.sh --no-browser
#
# Then: open the page → Telegram card → paste your bot token → press Start in the bot chat → press Start.
set -euo pipefail
cd "$(dirname "$0")"

HOST="" PORT="" OFFLINE=0 BACKGROUND=0 UPDATE=0 TEST=0 SMOKE=0 BROWSER=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --offline) OFFLINE=1; shift ;;
    --background) BACKGROUND=1; shift ;;
    --update) UPDATE=1; shift ;;
    --test) TEST=1; shift ;;
    --smoke-train) SMOKE=1; shift ;;
    --no-browser) BROWSER=0; shift ;;
    --stop)
      if [[ -f logs/app.pid ]]; then kill "$(cat logs/app.pid)" 2>/dev/null && echo "stopped" || echo "not running"; rm -f logs/app.pid
      else echo "not running"; fi
      exit 0 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*"; }

[[ $UPDATE == 1 ]] && { say "git pull"; git pull --ff-only || warn "git pull failed; continuing with local copy"; }

# ---------------------------------------------------------------- Python ≥ 3.10
PY=""
for c in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [[ -z "$PY" ]]; then
  say "Python 3.10+ not found — trying to install it"
  if command -v apt-get >/dev/null; then sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
  elif command -v dnf >/dev/null; then sudo dnf install -y python3 python3-pip
  elif command -v brew >/dev/null; then brew install python@3.11
  else echo "Please install Python 3.10+ (https://www.python.org/downloads/) and re-run."; exit 1; fi
  PY=python3
fi
say "using $($PY --version)"

# ---------------------------------------------------------------- virtualenv + packages
if [[ ! -x .venv/bin/python ]]; then
  say "creating virtual environment .venv"
  "$PY" -m venv .venv || { warn "venv failed (Debian/Ubuntu: sudo apt-get install python3-venv)"; exit 1; }
fi
VPY=.venv/bin/python
HASH=$( (cat requirements.txt; command -v nvidia-smi >/dev/null && echo gpu) | sha256sum 2>/dev/null | cut -c1-16 || shasum -a 256 requirements.txt | cut -c1-16)
if [[ "$(cat .venv/.installed 2>/dev/null)" != "$HASH" ]]; then
  say "installing Python packages (first run takes a few minutes)"
  "$VPY" -m pip install --upgrade pip wheel >/dev/null
  if ! "$VPY" -c "import torch" 2>/dev/null; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      say "NVIDIA GPU detected → installing CUDA build of PyTorch"
      "$VPY" -m pip install torch
    elif [[ "$(uname)" == "Darwin" ]]; then
      "$VPY" -m pip install torch
    else
      say "no GPU → installing CPU build of PyTorch"
      "$VPY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu || "$VPY" -m pip install torch
    fi
  fi
  "$VPY" -m pip install -r requirements.txt
  echo "$HASH" > .venv/.installed
else
  say "Python packages up to date"
fi

# ---------------------------------------------------------------- Node + docx-js (optional)
if command -v npm >/dev/null 2>&1; then
  if [[ ! -d node_modules/docx ]]; then say "installing docx-js (npm)"; npm install --no-audit --no-fund --silent || warn "npm install failed; python-docx will be used"; fi
else
  warn "Node.js not found — Word files will use the python-docx backend (install Node 18+ for docx-js)"
fi

# ---------------------------------------------------------------- config
[[ -f .env ]] || { cp .env.example .env; chmod 600 .env; say "created .env (connect the Telegram bot from the web page)"; }
mkdir -p logs outputs/proposals outputs/records

if [[ $OFFLINE == 1 ]]; then
  export PA__SEARCH__PROVIDER=offline PA__RESEARCH__PROVIDER=offline
  say "offline demo mode (bundled fictional companies)"
fi

[[ $TEST == 1 ]] && { say "running tests"; "$VPY" -m pytest -q; }
[[ $SMOKE == 1 ]] && { say "training the tiny model end-to-end (CPU)"; PATH="$PWD/.venv/bin:$PATH" ./scripts/smoke_pipeline.sh
  export PA__AGENT__CHECKPOINT=training/checkpoints/sft-tiny/latest.pt; }

# ---------------------------------------------------------------- launch
HOST=${HOST:-$("$VPY" -c "from core.config import load_config; print(load_config()['app']['host'])")}
PORT=${PORT:-$("$VPY" -c "from core.config import load_config; print(load_config()['app']['port'])")}
URL="http://$([[ $HOST == 0.0.0.0 ]] && echo 127.0.0.1 || echo "$HOST"):$PORT/"

open_browser() {
  [[ $BROWSER == 1 ]] || return 0
  for _ in $(seq 1 60); do
    if "$VPY" -c "import urllib.request,sys; urllib.request.urlopen('$URL', timeout=1)" 2>/dev/null; then break; fi
    sleep 0.5
  done
  if command -v xdg-open >/dev/null; then xdg-open "$URL" >/dev/null 2>&1 || true
  elif command -v open >/dev/null; then open "$URL" || true; fi
}

say "ProposalAgent → $URL"
echo "    1) Telegram card: paste your bot token (from @BotFather) → open the bot → press Start"
echo "    2) Press Start in the app. The heartbeat keeps it running until you press Stop."
if [[ $BACKGROUND == 1 ]]; then
  nohup "$VPY" -m app --host "$HOST" --port "$PORT" > logs/app.out 2>&1 &
  echo $! > logs/app.pid
  open_browser
  say "running in background (PID $(cat logs/app.pid)); stop with ./run.sh --stop"
else
  open_browser &
  exec "$VPY" -m app --host "$HOST" --port "$PORT"
fi
