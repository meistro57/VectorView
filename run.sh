#!/usr/bin/env bash
# Build VectorView and launch it with the project's Python venv on PATH.
#
#   ./run.sh              # create venv if needed, rebuild, start
#   ./run.sh --no-build   # skip the Go rebuild
#   NO_BUILD=1 ./run.sh   # same, via environment
#
# Extra arguments are forwarded to the binary.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV_DIR="${VECTORVIEW_VENV_DIR:-$ROOT/venv}"
PYTHON_BIN="${VECTORVIEW_PYTHON:-$VENV_DIR/bin/python3}"
BUILD_BINARY="$ROOT/vectorview"

log() { printf '\033[1;36m[run]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[run]\033[0m %s\n' "$*" >&2; exit 1; }

if [[ ! -x "$PYTHON_BIN" ]]; then
	log "Creating Python virtualenv at $VENV_DIR"
	command -v python3 >/dev/null 2>&1 || die "python3 not found on PATH"
	python3 -m venv "$VENV_DIR" || die "failed to create virtualenv (is python3-venv installed?)"
	PYTHON_BIN="$VENV_DIR/bin/python3"
fi

if [[ -f "$ROOT/requirements.txt" ]]; then
	log "Installing Python requirements"
	"$PYTHON_BIN" -m pip install --quiet --disable-pip-version-check --upgrade pip >/dev/null
	"$PYTHON_BIN" -m pip install --quiet --disable-pip-version-check -r "$ROOT/requirements.txt"
fi

command -v go >/dev/null 2>&1 || die "go not found on PATH"

BUILD=1
ARGS=()
for arg in "$@"; do
	case "$arg" in
	--no-build) BUILD=0 ;;
	*) ARGS+=("$arg") ;;
	esac
done
if [[ "${NO_BUILD:-0}" == "1" ]]; then BUILD=0; fi

if [[ "$BUILD" == "1" ]]; then
	log "Building ./vectorview"
	go build -o "$BUILD_BINARY" .
fi

[[ -x "$BUILD_BINARY" ]] || die "missing $BUILD_BINARY (run without --no-build first)"

export VECTORVIEW_PYTHON="$PYTHON_BIN"
export PATH="$VENV_DIR/bin:$PATH"

PORT="${VECTORVIEW_PORT:-7433}"
log "Starting VectorView on http://localhost:$PORT"
exec "$BUILD_BINARY" "${ARGS[@]}"
