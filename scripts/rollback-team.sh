#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  rollback-team.sh setup BACKUP_DIR
  rollback-team.sh skills STAMP
  rollback-team.sh progress BACKUP_DIR
EOF
  exit 2
}

MODE="${1:-}"
VALUE="${2:-}"
test -n "$MODE" && test -n "$VALUE" || usage
HERMES_BIN="${HERMES_BIN:-$(command -v hermes || true)}"
test -n "$HERMES_BIN" && test -x "$HERMES_BIN"
CONFIG_PATH="$($HERMES_BIN config path)"
HERMES_HOME="$(dirname "$CONFIG_PATH")"

case "$MODE" in
  progress)
    exec "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/install-team-progress.sh" --rollback "$VALUE"
    ;;
  setup)
    backup="$(realpath -e "$VALUE")"
    case "$backup" in "$HERMES_HOME/team-change-backups/"*) ;; *) echo "ERROR: backup outside Hermes backup root" >&2; exit 2 ;; esac
    test -f "$backup/default-config.yaml"
    cp -p "$backup/default-config.yaml" "$CONFIG_PATH"
    if test -f "$backup/default-SOUL.md"; then
      cp -p "$backup/default-SOUL.md" "$HERMES_HOME/SOUL.md"
    elif test -f "$backup/default-SOUL.md.absent"; then
      rm -f -- "$HERMES_HOME/SOUL.md"
    fi
    for role in deployer tester profiler analyst; do
      src="$backup/profiles/$role"
      dst="$HERMES_HOME/profiles/$role"
      test ! -f "$src/SOUL.md" || cp -p "$src/SOUL.md" "$dst/SOUL.md"
      test ! -f "$src/config.yaml" || cp -p "$src/config.yaml" "$dst/config.yaml"
    done
    "$HERMES_BIN" config check
    echo "ROLLED_BACK setup=$backup"
    ;;
  skills)
    stamp="$VALUE"
    case "$stamp" in *[!A-Za-z0-9_.-]*|'') echo "ERROR: unsafe stamp" >&2; exit 2 ;; esac
    record="$HERMES_HOME/skill-install-records/$stamp-installed.tsv"
    backup="$HERMES_HOME/skill-backups/$stamp"
    test -s "$record"
    tac "$record" | while IFS=$'\t' read -r role skill; do
      case "$role" in
        leader) parent="$HERMES_HOME/skills" ;;
        deployer|tester|profiler|analyst) parent="$HERMES_HOME/profiles/$role/skills" ;;
        *) echo "ERROR: invalid role in record" >&2; exit 2 ;;
      esac
      case "$skill" in *[!A-Za-z0-9_.-]*|'') echo "ERROR: unsafe skill in record" >&2; exit 2 ;; esac
      target="$parent/$skill"
      test ! -e "$target" || rm -rf -- "$target"
      test ! -e "$backup/$role/$skill" || mv "$backup/$role/$skill" "$target"
    done
    echo "ROLLED_BACK skills=$stamp"
    ;;
  *) usage ;;
esac
