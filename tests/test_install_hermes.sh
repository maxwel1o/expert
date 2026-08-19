#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf -- "$TMP_DIR"' EXIT

cat >"$TMP_DIR/hermes-ok" <<'EOF'
#!/usr/bin/env bash
echo 'Hermes Agent v0.17.0 (test fixture)'
EOF
chmod +x "$TMP_DIR/hermes-ok"

ok_output="$(HERMES_BIN="$TMP_DIR/hermes-ok" "$ROOT/scripts/install-hermes.sh" --dry-run)"
grep -q 'KEEP hermes=' <<<"$ok_output"
grep -q 'version=0.17.0' <<<"$ok_output"

top_output="$(HERMES_BIN="$TMP_DIR/hermes-ok" "$ROOT/install.sh" --install-hermes)"
grep -q 'version=0.17.0' <<<"$top_output"

cat >"$TMP_DIR/hermes-wrong" <<'EOF'
#!/usr/bin/env bash
echo 'Hermes Agent v0.16.0 (test fixture)'
EOF
chmod +x "$TMP_DIR/hermes-wrong"

if HERMES_BIN="$TMP_DIR/hermes-wrong" "$ROOT/scripts/install-hermes.sh" --dry-run \
  >"$TMP_DIR/wrong.out" 2>"$TMP_DIR/wrong.err"; then
  echo "expected version mismatch to fail" >&2
  exit 1
fi
grep -q 'expected 0.17.0, found 0.16.0' "$TMP_DIR/wrong.err"

echo "INSTALL_HERMES_TEST_OK"
