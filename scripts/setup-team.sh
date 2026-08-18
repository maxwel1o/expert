#!/usr/bin/env bash
set -Eeuo pipefail

# Configure the default Hermes Agent as Leader and create four independent
# Worker profiles. API/provider settings must already exist in the default
# Hermes configuration; this script never embeds credentials.

MODE="${1:---apply}"
case "$MODE" in --dry-run|--apply) ;; *) echo "usage: $0 [--dry-run|--apply]" >&2; exit 2 ;; esac

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
HERMES_BIN="${HERMES_BIN:-$(command -v hermes || true)}"
test -n "$HERMES_BIN" && test -x "$HERMES_BIN" || {
  echo "ERROR: hermes executable not found; set HERMES_BIN" >&2; exit 1;
}

CONFIG_PATH="$($HERMES_BIN config path)"
HERMES_HOME_DIR="$(dirname "$CONFIG_PATH")"
PROFILE_ROOT="$HERMES_HOME_DIR/profiles"
ENV_PATH="$HERMES_HOME_DIR/.env"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
BACKUP_ROOT="$HERMES_HOME_DIR/team-change-backups/$STAMP-team-setup"
WORKERS=(deployer tester profiler analyst)

for role in leader "${WORKERS[@]}"; do
  test -s "$REPO_ROOT/roles/$role/SOUL.md" || {
    echo "ERROR: missing roles/$role/SOUL.md" >&2; exit 1;
  }
done
test -s "$CONFIG_PATH" || { echo "ERROR: configure Hermes API/provider first" >&2; exit 1; }

if [ "$MODE" = --dry-run ]; then
  echo "DRY_RUN hermes=$HERMES_BIN"
  echo "DRY_RUN hermes_home=$HERMES_HOME_DIR"
  echo "DRY_RUN leader=default workers=${WORKERS[*]}"
  echo "DRY_RUN backup=$BACKUP_ROOT"
  exit 0
fi

mkdir -p "$BACKUP_ROOT/profiles" "$PROFILE_ROOT"
cp -p "$CONFIG_PATH" "$BACKUP_ROOT/default-config.yaml"
if test -f "$HERMES_HOME_DIR/SOUL.md"; then
  cp -p "$HERMES_HOME_DIR/SOUL.md" "$BACKUP_ROOT/default-SOUL.md"
else
  : >"$BACKUP_ROOT/default-SOUL.md.absent"
fi

remove_worker_kanban_toolset() {
  local config="$1"
  sed -i '/^toolsets:$/,/^[^ ]/ { /^  - kanban$/d; }' "$config"
}

ensure_worker() {
  local role="$1" description="$2" profile_dir="$PROFILE_ROOT/$1"
  if "$HERMES_BIN" profile show "$role" >/dev/null 2>&1; then
    echo "KEEP profile=$role"
  else
    "$HERMES_BIN" profile create "$role" --no-skills --description "$description"
    install -m 600 "$CONFIG_PATH" "$profile_dir/config.yaml"
    # A local user's already-configured provider settings may be inherited by
    # Workers. No credential is read into, or written by, this repository.
    test ! -f "$ENV_PATH" || install -m 600 "$ENV_PATH" "$profile_dir/.env"
    echo "CREATED profile=$role"
  fi
  mkdir -p "$BACKUP_ROOT/profiles/$role"
  test ! -f "$profile_dir/SOUL.md" || cp -p "$profile_dir/SOUL.md" "$BACKUP_ROOT/profiles/$role/SOUL.md"
  test ! -f "$profile_dir/config.yaml" || cp -p "$profile_dir/config.yaml" "$BACKUP_ROOT/profiles/$role/config.yaml"
  test -f "$profile_dir/config.yaml" || install -m 600 "$CONFIG_PATH" "$profile_dir/config.yaml"
  remove_worker_kanban_toolset "$profile_dir/config.yaml"
  install -m 644 "$REPO_ROOT/roles/$role/SOUL.md" "$profile_dir/SOUL.md"
  mkdir -p "$profile_dir/skills"
}

ensure_worker deployer "Ascend NPU model deployment and service lifecycle Worker"
ensure_worker tester "Ascend NPU functional, precision, performance and stability testing Worker"
ensure_worker profiler "Ascend NPU profiling collection and evidence integrity Worker"
ensure_worker analyst "Ascend NPU performance evidence analysis and optimization Worker"

install -m 644 "$REPO_ROOT/roles/leader/SOUL.md" "$HERMES_HOME_DIR/SOUL.md"

# Kanban is a top-level toolset in Hermes v0.17.x, not an entry accepted by
# `hermes tools enable`. Only the default Agent receives the orchestrator set.
block="$(sed -n '/^toolsets:$/,/^[^ ]/p' "$CONFIG_PATH")"
if printf '%s\n' "$block" | grep -q '^  - kanban$'; then
  :
elif grep -q '^toolsets:$' "$CONFIG_PATH"; then
  sed -i '/^toolsets:$/a\  - kanban' "$CONFIG_PATH"
elif grep -q '^toolsets: \[\]$' "$CONFIG_PATH"; then
  sed -i 's/^toolsets: \[\]$/toolsets:\n  - kanban/' "$CONFIG_PATH"
else
  sed -i '1i toolsets:\n  - kanban' "$CONFIG_PATH"
fi

"$HERMES_BIN" kanban init
"$HERMES_BIN" config set kanban.enabled true
"$HERMES_BIN" config set kanban.auto_decompose false
"$HERMES_BIN" config set kanban.dispatch_in_gateway false
sed -i '/^  orchestrator_profile:/d; /^  default_assignee:/d' "$CONFIG_PATH"

if test "$(id -u)" -eq 0 && id hermes >/dev/null 2>&1; then
  chown -R hermes:hermes "$HERMES_HOME_DIR/SOUL.md" "$PROFILE_ROOT"
fi
"$HERMES_BIN" config check
echo "TEAM_READY leader=default workers=${WORKERS[*]} backup=$BACKUP_ROOT"
echo "NOTICE: restart the Hermes Gateway only if your deployment requires it."
