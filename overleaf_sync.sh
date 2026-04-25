#!/usr/bin/env bash
# Sync the QF manuscript package to an Overleaf Git project.
#
# Usage examples:
#   bash overleaf_sync.sh --remote "https://git.overleaf.com/<project_id>"
#   bash overleaf_sync.sh --message "Update after referee revisions"
#   bash overleaf_sync.sh --no-push
#
# Notes:
# - The script syncs only manuscript-relevant files into ./overleaf-sync:
#     main_qf.tex
#     merged.bib
#     qf/figures/**
#     qf/results/**/*.tex
# - The first run should provide --remote unless overleaf remote already exists.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SYNC_DIR="$ROOT/overleaf-sync"
REMOTE_NAME="overleaf"
REMOTE_URL=""
COMMIT_MESSAGE="Sync manuscript package to Overleaf"
NO_PUSH=0

usage() {
  cat <<'EOF'
Usage:
  bash overleaf_sync.sh [options]

Options:
  --remote <url>       Overleaf Git URL (e.g., https://git.overleaf.com/<id>)
  --sync-dir <path>    Local sync directory (default: ./overleaf-sync)
  --message <text>     Commit message
  --no-push            Prepare sync + commit, but do not push
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)
      REMOTE_URL="${2:-}"
      shift 2
      ;;
    --sync-dir)
      SYNC_DIR="${2:-}"
      shift 2
      ;;
    --message)
      COMMIT_MESSAGE="${2:-}"
      shift 2
      ;;
    --no-push)
      NO_PUSH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[overleaf] Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

mkdir -p "$SYNC_DIR"

EXTRA_EXCLUDES=()
if [[ "$SYNC_DIR" == "$ROOT/"* ]]; then
  REL_SYNC_DIR="${SYNC_DIR#"$ROOT"/}"
  EXTRA_EXCLUDES+=(--exclude="/$REL_SYNC_DIR/***")
fi

echo "[overleaf] Syncing manuscript files into: $SYNC_DIR"
rsync -a --delete --prune-empty-dirs \
  "${EXTRA_EXCLUDES[@]}" \
  --include="/main_qf.tex" \
  --include="/merged.bib" \
  --include="/qf/" \
  --include="/qf/figures/***" \
  --include="/qf/results/" \
  --include="/qf/results/*.tex" \
  --include="/qf/results/**/*.tex" \
  --exclude="*" \
  "$ROOT/" "$SYNC_DIR/"

if [[ ! -d "$SYNC_DIR/.git" ]]; then
  echo "[overleaf] Initializing git repo in sync directory"
  git -C "$SYNC_DIR" init
fi

if [[ -n "$REMOTE_URL" ]]; then
  if git -C "$SYNC_DIR" remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
    git -C "$SYNC_DIR" remote set-url "$REMOTE_NAME" "$REMOTE_URL"
  else
    git -C "$SYNC_DIR" remote add "$REMOTE_NAME" "$REMOTE_URL"
  fi
fi

git -C "$SYNC_DIR" add -A
if git -C "$SYNC_DIR" diff --cached --quiet; then
  echo "[overleaf] No changes to commit."
else
  git -C "$SYNC_DIR" commit -m "$COMMIT_MESSAGE"
  echo "[overleaf] Committed changes."
fi

if [[ "$NO_PUSH" -eq 1 ]]; then
  echo "[overleaf] --no-push set. Skipping push."
  exit 0
fi

if ! git -C "$SYNC_DIR" remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
  echo "[overleaf] Remote '$REMOTE_NAME' is not configured." >&2
  echo "[overleaf] Re-run with: --remote https://git.overleaf.com/<project_id>" >&2
  exit 1
fi

echo "[overleaf] Pushing to $REMOTE_NAME (HEAD -> master)"
git -C "$SYNC_DIR" push "$REMOTE_NAME" HEAD:master
echo "[overleaf] Done."
