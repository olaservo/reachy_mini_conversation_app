#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# Deploy conversation app to Reachy Mini Wireless and start it via the daemon.
#
# Rsyncs your local source code to the robot's site-packages, then
# starts the app through the daemon API (same as clicking "Start"
# on the dashboard). Logs stream live to your terminal.
#
# PREREQUISITES:
#   1. Passwordless SSH access to the robot:
#        ssh-copy-id pollen@reachy-mini.local    (password: root)
#
#   2. The reachy-mini-daemon must be running on the robot.
#      It starts automatically on boot. If stopped:
#        ssh pollen@reachy-mini.local 'systemctl restart reachy-mini-daemon.service'
#
#   3. The robot must be powered on and on the same network as your laptop.
#      If reachy-mini.local doesn't resolve, pass the IP address instead.
#
# PLATFORM:
#   Requires bash, ssh, rsync, and curl.
#   Works on Linux and macOS natively. On Windows, use Git Bash or WSL.
#
# WIRELESS ONLY:
#   This script targets the Wireless version's shared venv at
#   /venvs/apps_venv/ with Python 3.12. The Lite version uses a
#   different venv layout — this script will NOT work on Lite as-is.
#
# HOW IT WORKS:
#   1. Checks the daemon is running (systemctl is-active)
#   2. Rsyncs reachy_mini_conversation_app/ into site-packages (~1 second)
#   3. Stops any currently running app (POST /api/apps/stop-current-app)
#   4. Starts conversation app via daemon (POST /api/apps/start-app/reachy_mini_conversation_app)
#   5. Tails journalctl for live log output
#
#   Ctrl+C detaches from logs — the app keeps running on the robot.
#   You can also stop it from the dashboard, or with the daemon API.
#
# USAGE:
#   ./deploy_wireless.sh                  # default: reachy-mini.local
#   ./deploy_wireless.sh 192.168.1.42    # custom IP
#   ./deploy_wireless.sh --sync-only     # sync without starting
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

SYNC_ONLY=false
HOST="reachy-mini.local"

for arg in "$@"; do
    case "$arg" in
        --sync-only) SYNC_ONLY=true ;;
        *)           HOST="$arg" ;;
    esac
done

USER="pollen"
SITE_PACKAGES="/venvs/apps_venv/lib/python3.12/site-packages"
REMOTE="${USER}@${HOST}"
TARGET_APP="${REMOTE}:${SITE_PACKAGES}/reachy_mini_conversation_app/"
TARGET_DATA="${REMOTE}:${SITE_PACKAGES}/reachy_talk_data/"
DAEMON_API="http://127.0.0.1:8000/api/apps"

# Resolve the project root (one level up from scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_APP="${SCRIPT_DIR}/src/reachy_mini_conversation_app/"
SOURCE_DATA="${SCRIPT_DIR}/src/reachy_talk_data/"
SOURCE_PROFILES="${SCRIPT_DIR}/profiles/"

# ── 1. Check daemon is running ───────────────────────────────────────
echo "==> Checking daemon on ${HOST}..."
if ! ssh -o ConnectTimeout=5 "${REMOTE}" \
    "systemctl is-active --quiet reachy-mini-daemon" 2>/dev/null; then
    echo "    ERROR: reachy-mini-daemon is not running on ${HOST}."
    echo "    Start it with: ssh ${REMOTE} 'sudo systemctl start reachy-mini-daemon'"
    exit 1
fi
echo "    Daemon is running."

# ── 2. Rsync source to site-packages ─────────────────────────────────
# This overwrites the installed package with your local source.
# The dashboard can still manage the app normally afterwards.
echo "==> Syncing reachy_mini_conversation_app/ to ${HOST}..."
rsync -az --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    "$SOURCE_APP" "$TARGET_APP"

# Profiles live in reachy_talk_data (moved from inside the app package).
# setup.py copies profiles/ into this package at build time; rsync does
# the same for deploy.
echo "==> Syncing reachy_talk_data/ + profiles/ to ${HOST}..."
rsync -az --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    "$SOURCE_DATA" "$TARGET_DATA"
rsync -az --delete \
    "$SOURCE_PROFILES" "${REMOTE}:${SITE_PACKAGES}/reachy_talk_data/profiles/"
echo "    Sync complete."

if $SYNC_ONLY; then
    echo "Done (sync only)."
    exit 0
fi

# ── 3. Stop any currently running app ─────────────────────────────────
# Uses the same daemon API as the dashboard's "Stop" button.
echo "==> Stopping current app (if any)..."
ssh "${REMOTE}" "curl -sf -X POST ${DAEMON_API}/stop-current-app >/dev/null 2>&1 || true"
sleep 1

# ── 4. Start conversation app via daemon API ──────────────────────────
echo "==> Starting reachy_mini_conversation_app via daemon..."
RESP=$(ssh "${REMOTE}" "curl -sf -X POST ${DAEMON_API}/start-app/reachy_mini_conversation_app 2>/dev/null")
if [ $? -ne 0 ]; then
    echo "    ERROR: Failed to start conversation app via daemon API."
    echo "    Response: ${RESP}"
    exit 1
fi
echo "    Started."

echo ""
echo "Web UI: http://${HOST}:8042"
echo ""

# ── 5. Tail logs live ─────────────────────────────────────────────────
echo "==> Tailing logs (Ctrl+C to detach — app keeps running)..."
echo ""
ssh "${REMOTE}" "journalctl -u reachy-mini-daemon -f --no-hostname -o cat" 2>/dev/null || true
