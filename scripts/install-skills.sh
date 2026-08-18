#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
BUNDLE_ROOT="$REPO_ROOT/skills"
MANIFEST_SOURCE="$REPO_ROOT/manifests/skills.csv"
STAMP="${1:-$(date -u +%Y%m%d-%H%M%S)}"
HERMES_HOME="${HERMES_HOME:-/opt/data}"

case "$STAMP" in
  *[!A-Za-z0-9_.-]*|'')
    printf 'ERROR: unsafe stamp: %s\n' "$STAMP" >&2
    exit 2
    ;;
esac

test "$(id -u)" -eq 0
test -d "$BUNDLE_ROOT"
test -f "$MANIFEST_SOURCE"
test -d "$HERMES_HOME/skills"

declare -A EXPECTED=(
  [leader]=1
  [deployer]=127
  [tester]=19
  [profiler]=5
  [analyst]=49
)

target_parent() {
  case "$1" in
    leader) printf '%s\n' "$HERMES_HOME/skills" ;;
    deployer|tester|profiler|analyst) printf '%s\n' "$HERMES_HOME/profiles/$1/skills" ;;
    *) return 2 ;;
  esac
}

for role in leader deployer tester profiler analyst; do
  source_role="$BUNDLE_ROOT/$role"
  target_role="$(target_parent "$role")"
  test -d "$source_role"
  test -d "$target_role"

  actual="$(find "$source_role" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  if [ "$actual" -ne "${EXPECTED[$role]}" ]; then
    printf 'ERROR: %s expected %s packages, got %s\n' "$role" "${EXPECTED[$role]}" "$actual" >&2
    exit 3
  fi

  while IFS= read -r source_dir; do
    skill_name="$(basename "$source_dir")"
    case "$skill_name" in
      *[!A-Za-z0-9_.-]*|'')
        printf 'ERROR: unsafe skill name: %s\n' "$skill_name" >&2
        exit 3
        ;;
    esac
    test -s "$source_dir/SKILL.md"
  done < <(find "$source_role" -mindepth 1 -maxdepth 1 -type d | sort)
done

if find "$BUNDLE_ROOT" -type l -print -quit | grep -q .; then
  printf 'ERROR: bundle contains symbolic links\n' >&2
  exit 3
fi

if find "$BUNDLE_ROOT" -mindepth 4 -name SKILL.md -type f ! -path '*/references/*' -print -quit | grep -q .; then
  printf 'ERROR: bundle contains an unexpected nested discoverable SKILL.md\n' >&2
  exit 3
fi

total="$(find "$BUNDLE_ROOT" -mindepth 3 -maxdepth 3 -name SKILL.md -type f | wc -l)"
if [ "$total" -ne 201 ]; then
  printf 'ERROR: expected 201 package SKILL.md files, got %s\n' "$total" >&2
  exit 3
fi

STAGE_ROOT="$HERMES_HOME/.skill-install-stage-$STAMP"
BACKUP_ROOT="$HERMES_HOME/skill-backups/$STAMP"
RECORD_ROOT="$HERMES_HOME/skill-install-records"

if [ -e "$STAGE_ROOT" ] || [ -e "$BACKUP_ROOT" ]; then
  printf 'ERROR: stage or backup path already exists for stamp %s\n' "$STAMP" >&2
  exit 4
fi

install -d -m 700 "$STAGE_ROOT" "$BACKUP_ROOT" "$RECORD_ROOT"
cp "$MANIFEST_SOURCE" "$STAGE_ROOT/manifest.csv"
: > "$STAGE_ROOT/installed.tsv"
: > "$STAGE_ROOT/backed-up.tsv"

for role in leader deployer tester profiler analyst; do
  install -d -m 755 "$STAGE_ROOT/$role"
  while IFS= read -r source_dir; do
    skill_name="$(basename "$source_dir")"
    cp -a "$source_dir" "$STAGE_ROOT/$role/$skill_name"
  done < <(find "$BUNDLE_ROOT/$role" -mindepth 1 -maxdepth 1 -type d | sort)
done

chown -R hermes:hermes "$STAGE_ROOT"
chmod -R go-w "$STAGE_ROOT"

rollback() {
  rc="$?"
  trap - ERR
  printf 'ERROR: installation failed; rolling back committed targets\n' >&2
  if [ -f "$STAGE_ROOT/installed.tsv" ]; then
    tac "$STAGE_ROOT/installed.tsv" | while IFS=$'\t' read -r role skill_name; do
      [ -n "$role" ] || continue
      target="$(target_parent "$role")/$skill_name"
      rollback_dir="$STAGE_ROOT/rollback-new/$role"
      install -d -m 700 "$rollback_dir"
      if [ -e "$target" ]; then
        mv "$target" "$rollback_dir/$skill_name"
      fi
      backup="$BACKUP_ROOT/$role/$skill_name"
      if [ -e "$backup" ]; then
        mv "$backup" "$target"
      fi
    done
  fi
  exit "$rc"
}
trap rollback ERR

for role in leader deployer tester profiler analyst; do
  target_role="$(target_parent "$role")"
  while IFS= read -r staged_dir; do
    skill_name="$(basename "$staged_dir")"
    target="$target_role/$skill_name"
    if [ -e "$target" ]; then
      install -d -m 700 "$BACKUP_ROOT/$role"
      mv "$target" "$BACKUP_ROOT/$role/$skill_name"
      printf '%s\t%s\n' "$role" "$skill_name" >> "$STAGE_ROOT/backed-up.tsv"
    fi
    mv "$staged_dir" "$target"
    printf '%s\t%s\n' "$role" "$skill_name" >> "$STAGE_ROOT/installed.tsv"
  done < <(find "$STAGE_ROOT/$role" -mindepth 1 -maxdepth 1 -type d | sort)
done

trap - ERR

cp "$STAGE_ROOT/manifest.csv" "$RECORD_ROOT/$STAMP-manifest.csv"
cp "$STAGE_ROOT/installed.tsv" "$RECORD_ROOT/$STAMP-installed.tsv"
cp "$STAGE_ROOT/backed-up.tsv" "$RECORD_ROOT/$STAMP-backed-up.tsv"
chown hermes:hermes "$RECORD_ROOT/$STAMP-manifest.csv" "$RECORD_ROOT/$STAMP-installed.tsv" "$RECORD_ROOT/$STAMP-backed-up.tsv"

printf 'INSTALLED stamp=%s leader=%s deployer=%s tester=%s profiler=%s analyst=%s total=201\n' \
  "$STAMP" "${EXPECTED[leader]}" "${EXPECTED[deployer]}" "${EXPECTED[tester]}" "${EXPECTED[profiler]}" "${EXPECTED[analyst]}"
