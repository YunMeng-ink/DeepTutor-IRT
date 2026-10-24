#!/usr/bin/env bash
# One-click launcher for DeepTutor (backend + frontend together).
#
# Usage:
#   ./start.sh                  # production mode: builds web/, then starts
#   ./start.sh --dev            # Next.js dev server with hot reload
#   ./start.sh --init           # run the config wizard first, then start
#   ./start.sh --check          # verify the environment and exit (no start)
#
# Extra flags are passed through to `deeptutor start`
# (--detach, --no-browser, --home ...). Stop the app with Ctrl+C
# (or `deeptutor stop` after --detach). Default ports: frontend 3782,
# backend 8001 - your configured ports may differ; the startup banner
# prints the actual URLs.
#
# Interpreter resolution: $DEEPTUTOR_PYTHON if set, else the first of
# python / python3 / .venv / the conda env named "deeptutor" that can import
# deeptutor. If none qualifies, a fresh .venv is bootstrapped with
# `pip install -e .` (first run only, needs Python 3.11+).

set -euo pipefail

cd "$(dirname "$0")"

log() { printf '[start] %s\n' "$*"; }
warn() { printf '[start] WARNING: %s\n' "$*" >&2; }
die() { printf '[start] ERROR: %s\n' "$*" >&2; exit 1; }

# ── locate a Python that has DeepTutor importable ──────────────────────────

_venv_python() {
  if [ -x ".venv/bin/python" ]; then echo ".venv/bin/python"
  elif [ -x ".venv/Scripts/python.exe" ]; then echo ".venv/Scripts/python.exe"
  fi
}

_conda_deeptutor_python() {
  command -v conda >/dev/null 2>&1 || return 0
  local env_path
  env_path="$(conda env list 2>/dev/null | awk '$1 == "deeptutor" {print $NF; exit}')" || true
  [ -n "${env_path:-}" ] || return 0
  # conda prints Windows paths with backslashes; translate for Git Bash.
  if command -v cygpath >/dev/null 2>&1; then
    env_path="$(cygpath -u "$env_path" 2>/dev/null || echo "$env_path")"
  fi
  if [ -x "$env_path/bin/python" ]; then echo "$env_path/bin/python"
  elif [ -x "$env_path/python.exe" ]; then echo "$env_path/python.exe"
  fi
}

# Isolated mode (-I): the repo root is the cwd here, and any Python could
# "import deeptutor" straight from the checkout — only a real install counts.
_can_import() { "$1" -I -c "import deeptutor" >/dev/null 2>&1; }

find_python() {
  local candidates=()
  if [ -n "${DEEPTUTOR_PYTHON:-}" ]; then
    candidates+=("$DEEPTUTOR_PYTHON")
  fi
  local venv_py conda_py
  venv_py="$(_venv_python)"
  if [ -n "$venv_py" ]; then
    candidates+=("$venv_py")
  fi
  conda_py="$(_conda_deeptutor_python)"
  if [ -n "$conda_py" ]; then
    candidates+=("$conda_py")
  fi
  candidates+=("python" "python3")

  local candidate
  for candidate in "${candidates[@]}"; do
    command -v "$candidate" >/dev/null 2>&1 || [ -x "$candidate" ] || continue
    if _can_import "$candidate"; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

PY="$(find_python || true)"

if [ -z "$PY" ]; then
  log "no Python with DeepTutor found - bootstrapping .venv (first run only)"
  BOOT_PY="${DEEPTUTOR_PYTHON:-python}"
  command -v "$BOOT_PY" >/dev/null 2>&1 || die "Python not found. Install Python 3.11+ and re-run."
  "$BOOT_PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
    || die "Python 3.11+ required (found $($BOOT_PY --version))."
  "$BOOT_PY" -m venv .venv
  PY="$(_venv_python)"
  [ -n "$PY" ] || die "failed to create .venv"
  log "installing DeepTutor into .venv (pip install -e .) - this can take a few minutes"
  "$PY" -m pip install -e .
  _can_import "$PY" || die "installation failed - see the pip output above"
fi

"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
  || die "Python 3.11+ required (found $($PY --version))."

# ── flag handling ──────────────────────────────────────────────────────────

RUN_INIT=0
PASS_THROUGH=()
for arg in "$@"; do
  case "$arg" in
    --init) RUN_INIT=1 ;;
    *) PASS_THROUGH+=("$arg") ;;
  esac
done

# ── environment report / gate ──────────────────────────────────────────────

log "Python: $PY ($("$PY" --version 2>&1))"

if [ -d web ] && [ ! -f deeptutor_web/server.js ]; then
  if command -v node >/dev/null 2>&1; then
    log "Node.js: $(node --version)"
    [ -d web/node_modules ] || log "first start will run 'npm ci' + build the web app (one time)"
  else
    warn "Node.js 20+ not found - the frontend needs it (https://nodejs.org)."
  fi
fi

if [ ! -f data/user/settings/agents.yaml ]; then
  warn "no LLM provider configured yet."
  warn "run './start.sh --init' for the setup wizard, or configure later in Web Settings."
fi

if [ "${1:-}" = "--check" ]; then
  log "environment check passed"
  exit 0
fi

# ── launch ─────────────────────────────────────────────────────────────────

if [ "$RUN_INIT" = "1" ]; then
  log "running the setup wizard (deeptutor init)"
  "$PY" -m deeptutor_cli init
fi

log "starting DeepTutor (Ctrl+C stops both servers)..."
exec "$PY" -m deeptutor_cli start "${PASS_THROUGH[@]}"
