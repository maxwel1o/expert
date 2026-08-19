#!/usr/bin/env bash
set -Eeuo pipefail

EXPECTED_VERSION="0.17.0"
MODE="${1:---apply}"
case "$MODE" in
  --dry-run|--apply) ;;
  *) echo "usage: $0 [--dry-run|--apply]" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
SOURCE_DIR="$REPO_ROOT/vendor/hermes-agent"
UPSTREAM_INSTALLER="$SOURCE_DIR/setup-hermes.sh"

test -s "$SOURCE_DIR/pyproject.toml" || {
  echo "ERROR: vendored Hermes source is missing: $SOURCE_DIR" >&2
  exit 1
}
test -s "$SOURCE_DIR/LICENSE" || {
  echo "ERROR: vendored Hermes MIT license is missing" >&2
  exit 1
}
test -s "$UPSTREAM_INSTALLER" || {
  echo "ERROR: vendored Hermes installer is missing: $UPSTREAM_INSTALLER" >&2
  exit 1
}

detect_hermes() {
  if test -n "${HERMES_BIN:-}"; then
    printf '%s\n' "$HERMES_BIN"
  elif command -v hermes >/dev/null 2>&1; then
    command -v hermes
  elif test -x "$SOURCE_DIR/venv/bin/hermes"; then
    printf '%s\n' "$SOURCE_DIR/venv/bin/hermes"
  fi
}

read_version() {
  "$1" --version 2>/dev/null | sed -nE 's/.*v([0-9]+\.[0-9]+\.[0-9]+).*/\1/p' | head -n 1
}

CURRENT_BIN="$(detect_hermes || true)"
if test -n "$CURRENT_BIN"; then
  test -x "$CURRENT_BIN" || {
    echo "ERROR: HERMES_BIN is not executable: $CURRENT_BIN" >&2
    exit 1
  }
  CURRENT_VERSION="$(read_version "$CURRENT_BIN")"
  test "$CURRENT_VERSION" = "$EXPECTED_VERSION" || {
    echo "ERROR: Hermes version mismatch: expected $EXPECTED_VERSION, found ${CURRENT_VERSION:-unknown} at $CURRENT_BIN" >&2
    echo "Set HERMES_BIN to a Hermes $EXPECTED_VERSION executable or use an isolated environment." >&2
    exit 1
  }
  echo "KEEP hermes=$CURRENT_BIN version=$CURRENT_VERSION"
  exit 0
fi

if test "$MODE" = --dry-run; then
  echo "DRY_RUN action=install-hermes version=$EXPECTED_VERSION"
  echo "DRY_RUN source=$SOURCE_DIR"
  echo "DRY_RUN installer=$UPSTREAM_INSTALLER"
  echo "DRY_RUN api=configure-after-install-with-hermes-setup"
  exit 0
fi

echo "INSTALL hermes-version=$EXPECTED_VERSION source=$SOURCE_DIR"
echo "NOTICE: Hermes dependencies are rebuilt locally and are not stored in this repository."
echo "NOTICE: The upstream installer may offer to run 'hermes setup'; API credentials remain local to this user."
(
  cd "$SOURCE_DIR"
  bash "$UPSTREAM_INSTALLER"
)

CURRENT_BIN="$(detect_hermes || true)"
test -n "$CURRENT_BIN" && test -x "$CURRENT_BIN" || {
  echo "ERROR: Hermes installer finished but no executable was found." >&2
  echo "Reload your shell or set HERMES_BIN=$SOURCE_DIR/venv/bin/hermes" >&2
  exit 1
}
CURRENT_VERSION="$(read_version "$CURRENT_BIN")"
test "$CURRENT_VERSION" = "$EXPECTED_VERSION" || {
  echo "ERROR: installed Hermes version is ${CURRENT_VERSION:-unknown}; expected $EXPECTED_VERSION" >&2
  exit 1
}

echo "HERMES_READY bin=$CURRENT_BIN version=$CURRENT_VERSION"
echo "NEXT: run '$CURRENT_BIN setup' if you did not configure your model/API in the installer."
