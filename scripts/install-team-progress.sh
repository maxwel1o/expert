#!/bin/sh
set -eu

INSTALL_ROOT="/opt/data/team-progress"
CLI_PATH="/usr/local/bin/team-progress"
BACKUP_ROOT="/opt/data/team-change-backups"
S6_SETUIDGID="${S6_SETUIDGID:-/package/admin/s6-2.15.0.0/command/s6-setuidgid}"
MARKER="BEGIN TEAM-PROGRESS-PROTOCOL v1"
HOOK_COMMAND="/opt/data/team-progress/bin/hermes-kanban-progress-hook"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SOURCE_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PACKAGE_SOURCE="$SOURCE_ROOT/team_progress"
ASSET_SOURCE="$SOURCE_ROOT/team_progress/assets"
MERGER_SOURCE="$SCRIPT_DIR/merge-hermes-hooks.py"

DEFAULT_SOUL="/opt/data/SOUL.md"
DEFAULT_CONFIG="/opt/data/config.yaml"
DEFAULT_ALLOWLIST="/opt/data/shell-hooks-allowlist.json"

worker_ids() {
    printf '%s\n' deployer tester profiler analyst
}

profile_home() {
    printf '/opt/data/profiles/%s' "$1"
}

usage() {
    echo "usage: $0 --dry-run|--apply|--verify|--rollback BACKUP_DIR" >&2
    exit 2
}

require_sources() {
    test -x "$S6_SETUIDGID"
    test -f "$PACKAGE_SOURCE/cli.py"
    test -f "$ASSET_SOURCE/team-progress"
    test -f "$ASSET_SOURCE/hermes-kanban-progress-hook"
    test -f "$ASSET_SOURCE/leader-appendix.md"
    test -f "$ASSET_SOURCE/worker-appendix.md"
    test -f "$MERGER_SOURCE"
    python3 -c "import yaml" >/dev/null
}

require_profile_homes() {
    test -f "$DEFAULT_SOUL"
    test -f "$DEFAULT_CONFIG"
    for role in $(worker_ids); do
        home=$(profile_home "$role")
        test -f "$home/SOUL.md"
        test -f "$home/config.yaml"
    done
}

validate_configs() {
    python3 "$MERGER_SOURCE" check "$DEFAULT_CONFIG" >/dev/null
    for role in $(worker_ids); do
        python3 "$MERGER_SOURCE" check "$(profile_home "$role")/config.yaml" >/dev/null
    done
}

verify_souls() {
    count=$(grep -c "$MARKER" "$DEFAULT_SOUL")
    test "$count" -eq 1
    for role in $(worker_ids); do
        file="$(profile_home "$role")/SOUL.md"
        count=$(grep -c "$MARKER" "$file")
        test "$count" -eq 1
    done
}

verify_hook_home() {
    home=$1
    config="$home/config.yaml"
    allowlist="$home/shell-hooks-allowlist.json"
    python3 "$MERGER_SOURCE" verify "$config" >/dev/null
    test -f "$allowlist"
    "$S6_SETUIDGID" hermes env \
        HERMES_HOME="$home" HOME="$home" \
        hermes hooks doctor >/dev/null
}

verify_install() {
    test -x "$CLI_PATH"
    test -x "$HOOK_COMMAND"
    test -f "$INSTALL_ROOT/lib/team_progress/cli.py"
    test -d "$INSTALL_ROOT/state"
    owner=$(stat -c '%U:%G' "$INSTALL_ROOT/state")
    test "$owner" = "hermes:hermes"
    verify_souls
    "$S6_SETUIDGID" hermes "$CLI_PATH" --help >/dev/null
    "$S6_SETUIDGID" hermes "$CLI_PATH" reconcile --help >/dev/null
    "$S6_SETUIDGID" hermes "$CLI_PATH" wait-final --help >/dev/null
    "$S6_SETUIDGID" hermes "$CLI_PATH" status --all --json >/dev/null
    schema=$(
        "$S6_SETUIDGID" hermes python3 -c \
            "import sqlite3; c=sqlite3.connect('$INSTALL_ROOT/state/progress.db'); print(c.execute(\"select value from meta where key='schema_version'\").fetchone()[0]); c.close()"
    )
    version=$(
        "$S6_SETUIDGID" hermes env \
            PYTHONPATH="$INSTALL_ROOT/lib" \
            python3 -c \
            "import team_progress; print(team_progress.__version__)"
    )
    test "$schema" = "1"
    test "$version" = "1.2.0"
    verify_hook_home "/opt/data"
    for role in $(worker_ids); do
        verify_hook_home "$(profile_home "$role")"
    done
    echo "VERIFIED install_root=$INSTALL_ROOT schema=$schema version=$version active_profiles=5 workers=4"
}

patch_soul() {
    target=$1
    appendix=$2
    python3 - "$target" "$appendix" <<'PY'
import os
import re
import sys
import tempfile
from pathlib import Path

target = Path(sys.argv[1])
appendix = Path(sys.argv[2]).read_text(encoding="utf-8").strip()
text = target.read_text(encoding="utf-8").rstrip()
pattern = re.compile(
    r"\n*<!-- BEGIN TEAM-PROGRESS-PROTOCOL v1 -->.*?"
    r"<!-- END TEAM-PROGRESS-PROTOCOL v1 -->\n*",
    re.DOTALL,
)
text = pattern.sub("\n", text).rstrip() + "\n\n" + appendix + "\n"
mode = target.stat().st_mode & 0o777
owner = target.stat().st_uid
group = target.stat().st_gid
fd, temp_name = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temp_name, mode)
    os.chown(temp_name, owner, group)
    os.replace(temp_name, target)
finally:
    if os.path.exists(temp_name):
        os.unlink(temp_name)
PY
}

backup_optional() {
    source=$1
    destination=$2
    if test -e "$source"; then
        cp -p "$source" "$destination"
    else
        : >"$destination.absent"
    fi
}

restore_optional() {
    destination=$1
    saved=$2
    failed_dir=$3
    if test -f "$saved"; then
        cp -p "$saved" "$destination"
    elif test -f "$saved.absent" && test -e "$destination"; then
        name=$(basename "$destination")
        mv "$destination" "$failed_dir/$name.created-by-team-progress"
    fi
}

backup_sqlite() {
    source=$1
    destination=$2
    if test ! -f "$source"; then
        : >"$destination.absent"
        return
    fi
    python3 - "$source" "$destination" <<'PY'
import sqlite3
import sys

source = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True, timeout=5)
destination = sqlite3.connect(sys.argv[2])
try:
    source.backup(destination)
finally:
    destination.close()
    source.close()
PY
}

approve_hook_home() {
    home=$1
    "$S6_SETUIDGID" hermes env \
        HERMES_HOME="$home" HOME="$home" \
        python3 -c "from hermes_cli.config import load_config; from agent.shell_hooks import register_from_config; register_from_config(load_config(), accept_hooks=True)" >/dev/null
    "$S6_SETUIDGID" hermes env \
        HERMES_HOME="$home" HOME="$home" \
        hermes hooks test kanban_task_completed >/dev/null
    "$S6_SETUIDGID" hermes env \
        HERMES_HOME="$home" HOME="$home" \
        hermes hooks test kanban_task_blocked >/dev/null
    "$S6_SETUIDGID" hermes env \
        HERMES_HOME="$home" HOME="$home" \
        hermes hooks doctor >/dev/null
}

apply_install() {
    require_sources
    require_profile_homes
    validate_configs
    timestamp=$(date -u +%Y%m%d-%H%M%S)
    backup="$BACKUP_ROOT/$timestamp-team-progress"
    new_root="/opt/data/.team-progress-new-$timestamp"

    test ! -e "$backup"
    test ! -e "$new_root"
    mkdir -p \
        "$backup/souls" \
        "$backup/configs" \
        "$backup/allowlists" \
        "$backup/stale-state-files" \
        "$new_root/lib" \
        "$new_root/bin" \
        "$new_root/state"

    cp -p "$DEFAULT_SOUL" "$backup/souls/default.SOUL.md"
    backup_optional "$DEFAULT_CONFIG" "$backup/configs/default.config.yaml"
    backup_optional "$DEFAULT_ALLOWLIST" "$backup/allowlists/default.json"
    for role in $(worker_ids); do
        home=$(profile_home "$role")
        cp -p "$home/SOUL.md" "$backup/souls/$role.SOUL.md"
        backup_optional "$home/config.yaml" "$backup/configs/$role.config.yaml"
        backup_optional "$home/shell-hooks-allowlist.json" "$backup/allowlists/$role.json"
    done

    if test -d "$INSTALL_ROOT/state"; then
        backup_sqlite \
            "$INSTALL_ROOT/state/progress.db" \
            "$backup/progress.db"
    else
        : >"$backup/progress.db.absent"
    fi

    {
        echo "install_root=$INSTALL_ROOT"
        echo "cli_path=$CLI_PATH"
        echo "hook_command=$HOOK_COMMAND"
        echo "default_soul=$DEFAULT_SOUL"
        echo "active_profiles=5"
        echo "workers=4"
        echo "workers=$(worker_ids | tr '\n' ',' | sed 's/,$//')"
        echo "configs=5"
        echo "allowlists=5"
    } >"$backup/manifest.txt"
    chmod 0600 "$backup/manifest.txt"

    if test -e "$INSTALL_ROOT"; then
        mv "$INSTALL_ROOT" "$backup/previous-install"
        if test -d "$backup/previous-install/state"; then
            cp -a "$backup/previous-install/state/." "$new_root/state/"
        fi
        if test -e "$new_root/state/progress.db-wal"; then
            mv \
                "$new_root/state/progress.db-wal" \
                "$backup/stale-state-files/progress.db-wal"
        fi
        if test -e "$new_root/state/progress.db-shm"; then
            mv \
                "$new_root/state/progress.db-shm" \
                "$backup/stale-state-files/progress.db-shm"
        fi
        if test -f "$backup/progress.db"; then
            cp -p "$backup/progress.db" "$new_root/state/progress.db"
        fi
    fi
    if test -e "$CLI_PATH"; then
        cp -p "$CLI_PATH" "$backup/team-progress-wrapper"
        : >"$backup/had-wrapper"
    fi

    cp -a "$PACKAGE_SOURCE" "$new_root/lib/team_progress"
    install \
        -o root -g root -m 0755 \
        "$ASSET_SOURCE/hermes-kanban-progress-hook" \
        "$new_root/bin/hermes-kanban-progress-hook"
    install \
        -o root -g root -m 0755 \
        "$MERGER_SOURCE" \
        "$new_root/bin/merge-hermes-hooks.py"
    chown -R root:root "$new_root/lib" "$new_root/bin"
    chmod -R a-w "$new_root/lib"
    chown -R hermes:hermes "$new_root/state"
    chmod 0700 "$new_root/state"
    mv "$new_root" "$INSTALL_ROOT"
    chown root:root "$INSTALL_ROOT"
    chmod 0755 "$INSTALL_ROOT" "$INSTALL_ROOT/lib" "$INSTALL_ROOT/bin"
    install -o root -g root -m 0755 "$ASSET_SOURCE/team-progress" "$CLI_PATH"

    patch_soul "$DEFAULT_SOUL" "$ASSET_SOURCE/leader-appendix.md"
    for role in $(worker_ids); do
        patch_soul "$(profile_home "$role")/SOUL.md" "$ASSET_SOURCE/worker-appendix.md"
    done

    python3 "$INSTALL_ROOT/bin/merge-hermes-hooks.py" apply "$DEFAULT_CONFIG"
    for role in $(worker_ids); do
        python3 "$INSTALL_ROOT/bin/merge-hermes-hooks.py" apply "$(profile_home "$role")/config.yaml"
    done

    "$S6_SETUIDGID" hermes "$CLI_PATH" init >/dev/null
    approve_hook_home "/opt/data"
    for role in $(worker_ids); do
        approve_hook_home "$(profile_home "$role")"
    done

    verify_install
    echo "APPLIED backup=$backup"
}

rollback_install() {
    requested=$1
    resolved=$(realpath -e "$requested")
    case "$resolved" in
        "$BACKUP_ROOT"/*) ;;
        *) echo "refusing rollback path outside $BACKUP_ROOT" >&2; exit 2 ;;
    esac
    test -f "$resolved/manifest.txt"
    timestamp=$(date -u +%Y%m%d-%H%M%S)
    failed_root="/opt/data/team-progress.failed.$timestamp"
    failed_config="$resolved/rollback-created-files.$timestamp"
    mkdir -p "$failed_config"

    if test -f "$INSTALL_ROOT/state/progress.db"; then
        backup_sqlite \
            "$INSTALL_ROOT/state/progress.db" \
            "$resolved/progress-current-before-rollback-$timestamp.db"
    fi
    if test -e "$INSTALL_ROOT"; then
        mv "$INSTALL_ROOT" "$failed_root"
    fi
    if test -d "$resolved/previous-install"; then
        mv "$resolved/previous-install" "$INSTALL_ROOT"
        current_snapshot="$resolved/progress-current-before-rollback-$timestamp.db"
        if test -f "$current_snapshot"; then
            cp -p "$current_snapshot" "$INSTALL_ROOT/state/progress.db"
            chown hermes:hermes "$INSTALL_ROOT/state/progress.db"
        fi
    fi

    cp -p "$resolved/souls/default.SOUL.md" "$DEFAULT_SOUL"
    restore_optional "$DEFAULT_CONFIG" "$resolved/configs/default.config.yaml" "$failed_config"
    restore_optional "$DEFAULT_ALLOWLIST" "$resolved/allowlists/default.json" "$failed_config"
    for role in $(worker_ids); do
        home=$(profile_home "$role")
        cp -p "$resolved/souls/$role.SOUL.md" "$home/SOUL.md"
        restore_optional "$home/config.yaml" "$resolved/configs/$role.config.yaml" "$failed_config"
        restore_optional "$home/shell-hooks-allowlist.json" "$resolved/allowlists/$role.json" "$failed_config"
    done

    if test -f "$resolved/had-wrapper"; then
        cp -p "$resolved/team-progress-wrapper" "$CLI_PATH"
    elif test -e "$CLI_PATH"; then
        mv "$CLI_PATH" "$failed_config/team-progress-wrapper"
    fi
    echo "ROLLED_BACK backup=$resolved failed_install=${failed_root:-none}"
}

case "${1:-}" in
    --dry-run)
        require_sources
        require_profile_homes
        validate_configs
        echo "DRY_RUN source=$SOURCE_ROOT"
        echo "DRY_RUN install_root=$INSTALL_ROOT"
        echo "DRY_RUN cli_path=$CLI_PATH"
        echo "DRY_RUN hook_command=$HOOK_COMMAND"
        echo "DRY_RUN active_profiles=5 workers=4"
        echo "DRY_RUN backup_root=$BACKUP_ROOT"
        ;;
    --apply)
        test "$#" -eq 1 || usage
        apply_install
        ;;
    --verify)
        test "$#" -eq 1 || usage
        verify_install
        ;;
    --rollback)
        test "$#" -eq 2 || usage
        rollback_install "$2"
        ;;
    *)
        usage
        ;;
esac
