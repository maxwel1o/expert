#!/usr/bin/env bash
set -Eeuo pipefail

HERMES_BIN="${HERMES_BIN:-$(command -v hermes || true)}"
test -n "$HERMES_BIN" && test -x "$HERMES_BIN"
HERMES_HOME="$($HERMES_BIN config path)"
HERMES_HOME="$(dirname "$HERMES_HOME")"

"$HERMES_BIN" config check
"$HERMES_BIN" profile list
"$HERMES_BIN" kanban assignees

test -s "$HERMES_HOME/SOUL.md"
for role in deployer tester profiler analyst; do
  test -s "$HERMES_HOME/profiles/$role/SOUL.md"
  test -s "$HERMES_HOME/profiles/$role/config.yaml"
done

count_skills() {
  local role="$1" expected="$2" parent
  if [ "$role" = leader ]; then parent="$HERMES_HOME/skills"; else parent="$HERMES_HOME/profiles/$role/skills"; fi
  local actual
  actual="$(find "$parent" -mindepth 2 -maxdepth 2 -type f -name SKILL.md | wc -l)"
  test "$actual" -ge "$expected" || { echo "ERROR: $role Skill count $actual < $expected" >&2; exit 1; }
  echo "skills $role=$actual required=$expected"
}

count_skills leader 1
count_skills deployer 127
count_skills tester 19
count_skills profiler 5
count_skills analyst 49

echo "VERIFY_OK leader=default workers=4 bundled_skills=201"
