#!/bin/bash
# auto_push.sh — watches this repo for local changes and pushes them to
# GitHub automatically. Polling-based (no extra dependencies like fswatch
# needed) so it runs on a plain macOS Python/Homebrew-less setup.
#
# Usage:
#   ./auto_push.sh [interval_seconds]
#
# Logs to auto_push.log in the same folder. Ctrl+C to stop when run in
# the foreground; see run_auto_push_background.sh / the LaunchAgent
# setup for running it unattended.

set -uo pipefail
cd "$(dirname "$0")"

INTERVAL="${1:-15}"
LOG_FILE="./auto_push.log"
BRANCH_REMOTE="main"   # the branch name on GitHub to push to

if [ ! -d .git ]; then
  echo "Not a git repository: $(pwd)" | tee -a "$LOG_FILE"
  exit 1
fi

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG_FILE"
}

log "=== auto-push agent started (interval=${INTERVAL}s, repo=$(pwd)) ==="

while true; do
  if [ -n "$(git status --porcelain)" ]; then
    git add -A
    MSG="auto: local changes $(date '+%Y-%m-%d %H:%M:%S')"
    if git commit -m "$MSG" >> "$LOG_FILE" 2>&1; then
      log "committed: $MSG"
      if git push origin "HEAD:${BRANCH_REMOTE}" >> "$LOG_FILE" 2>&1; then
        log "pushed OK"
      else
        log "PUSH FAILED (will retry once new changes trigger another commit, or next time this succeeds — the commit is safe locally either way)"
      fi
    else
      log "nothing to commit (or commit failed) — see above"
    fi
  fi
  sleep "$INTERVAL"
done
