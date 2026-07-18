#!/usr/bin/env bash

# One-time pre-upgrade helper for rigs on keryx-miner-rdna3 <= v0.3.12, where downloaded
# models still live INSIDE the package dir (/hive/miners/custom/keryx-miner-rdna3/models).
# Changing the flightsheet Install URL makes HiveOS delete that dir before extracting the
# new package — taking the multi-GB GGUFs with it. Run this FIRST to move them into the
# shared cache (/hive/miners/custom/models), which v0.3.13+ uses natively and upgrades
# never touch:
#
#   wget -qO- https://raw.githubusercontent.com/shmutalov/keryx-miner-rdna3/rdna3/integrations/hiveos/pre-hive-upgrade.sh | bash
#
# Safe to re-run; v0.3.13+ also merges any leftover in-package cache on every start
# (h-run.sh), so forgetting this only costs a one-time re-download.

set -uo pipefail

SHARED_DIR="/hive/miners/custom/models"

# Standalone-friendly: old packages do not ship this script, so when it is piped from wget
# there is no manifest next to it — fall back to the fork's fixed install dir.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo /)"
if [[ -f "$SCRIPT_DIR/h-manifest.conf" ]]; then
    . "$SCRIPT_DIR/h-manifest.conf"
    MINER_DIR="${CUSTOM_MINER_DIR:-/hive/miners/custom/keryx-miner-rdna3}"
    MINER_BIN="${CUSTOM_MINERBIN:-keryx-miner-rdna3}"
else
    MINER_DIR="/hive/miners/custom/keryx-miner-rdna3"
    MINER_BIN="keryx-miner-rdna3"
fi

# Stop the miner so no download/inference is mid-write while models move.
pkill -f "$MINER_DIR/$MINER_BIN" 2>/dev/null || true
pkill -f "./$MINER_BIN" 2>/dev/null || true

cleanup_legacy_models() {
    local legacy_dir="$MINER_DIR/models"

    [[ ! -d "$legacy_dir" ]] && return 0
    [[ "$legacy_dir" == "$SHARED_DIR" ]] && return 0

    if [[ ! -d "$SHARED_DIR" ]]; then
        mv "$legacy_dir" "$SHARED_DIR" || true
        return 0
    fi

    # Merge only entries that do not yet exist in the shared cache.
    for entry in "$legacy_dir"/*; do
        [[ -e "$entry" ]] || break
        local name
        name="$(basename "$entry")"
        [[ -e "$SHARED_DIR/$name" ]] && continue
        mv "$entry" "$SHARED_DIR/$name" || true
    done

    # Remove empty directories left behind after merge.
    find "$legacy_dir" -depth -type d -empty -delete 2>/dev/null || true

    if rmdir "$legacy_dir" 2>/dev/null; then
        echo "[keryx] Legacy model cache moved to: $SHARED_DIR"
        return 0
    fi

    if [[ "${KERYX_PURGE_LEGACY_MODELS:-0}" == "1" ]]; then
        rm -rf "$legacy_dir" || true
        echo "[keryx] WARNING: force-purged legacy model cache at $legacy_dir (KERYX_PURGE_LEGACY_MODELS=1)."
    else
        echo "[keryx] WARNING: some entries remain at $legacy_dir (already present in $SHARED_DIR). They will be deleted with the package dir on upgrade — set KERYX_PURGE_LEGACY_MODELS=1 and re-run to remove them now."
    fi
}

if [[ ! -d "$MINER_DIR" ]]; then
    echo "[keryx] No install found at $MINER_DIR — nothing to migrate."
else
    cleanup_legacy_models
fi

echo "[keryx] Pre-upgrade preparation complete."
echo "[keryx] It is now safe to change the Install URL in the HiveOS custom miner config and apply."
