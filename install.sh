#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
WITH_PROGRESS=0
case "${1:-}" in
  "") ;;
  --with-progress) WITH_PROGRESS=1 ;;
  *) echo "usage: $0 [--with-progress]" >&2; exit 2 ;;
esac

"$SCRIPT_DIR/scripts/setup-team.sh" --apply
"$SCRIPT_DIR/scripts/install-skills.sh"
if [ "$WITH_PROGRESS" -eq 1 ]; then
  "$SCRIPT_DIR/scripts/install-team-progress.sh" --apply
fi
"$SCRIPT_DIR/scripts/verify-team.sh"
