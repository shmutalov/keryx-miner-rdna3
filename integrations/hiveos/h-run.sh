#!/usr/bin/env bash

cd `dirname $0`

[ -t 1 ] && . colors

. h-manifest.conf

[[ -z $CUSTOM_LOG_BASENAME ]] && echo -e "${RED}No CUSTOM_LOG_BASENAME is set${NOCOLOR}" && exit 1
[[ -z $CUSTOM_CONFIG_FILENAME ]] && echo -e "${RED}No CUSTOM_CONFIG_FILENAME is set${NOCOLOR}" && exit 1
[[ ! -f $CUSTOM_CONFIG_FILENAME ]] && echo -e "${RED}Custom config ${YELLOW}$CUSTOM_CONFIG_FILENAME${RED} is not found${NOCOLOR}" && exit 1

# RDNA3/Vulkan fork: the packaged binary is the zero-dup build — OPoI inference runs
# in-process (llama.cpp/Vulkan linked in) and the PoM walk shares its weight buffers, so
# there is no llama-server child and no CUDA. Runtime needs only the system Vulkan loader
# (libvulkan1, ensured by h-config.sh) and the amdgpu Vulkan driver (RADV/amdvlk).
# The package dir goes FIRST so the bundled Vulkan loader (libvulkan.so.1, shipped in the .tgz) is
# used even on HiveOS images without libvulkan1.
export LD_LIBRARY_PATH="$(dirname $0):${LD_LIBRARY_PATH:-}:/usr/lib/x86_64-linux-gnu"

# Shared, stable model cache OUTSIDE the package dir: a custom-miner upgrade deletes
# $CUSTOM_MINER_DIR, so the multi-GB GGUFs must not live inside it. The miner reads
# KERYX_MODELS_DIR (see slm::model_dir); a flightsheet env override wins over this default.
# Same path the official keryx-miner uses, so switching between miners re-uses the cache.
export KERYX_MODELS_DIR="${KERYX_MODELS_DIR:-/hive/miners/custom/models}"

# One-time migration: merge a legacy in-package cache (pre-v0.3.13 layout) into the shared
# cache so already-downloaded models are not re-fetched. Per-model dirs are moved only when
# absent from the shared cache; a non-empty leftover is warned about (set
# KERYX_PURGE_LEGACY_MODELS=1 in the flightsheet env to force-remove it).
cleanup_legacy_models() {
  local legacy_dir="$CUSTOM_MINER_DIR/models"
  local shared_dir="$KERYX_MODELS_DIR"

  [[ ! -d "$legacy_dir" ]] && return 0
  [[ "$legacy_dir" == "$shared_dir" ]] && return 0

  if [[ ! -d "$shared_dir" ]]; then
    mv "$legacy_dir" "$shared_dir" || true
    return 0
  fi

  # Merge only entries that do not yet exist in the shared cache.
  for entry in "$legacy_dir"/*; do
    [[ -e "$entry" ]] || break
    local name
    name="$(basename "$entry")"
    [[ -e "$shared_dir/$name" ]] && continue
    mv "$entry" "$shared_dir/$name" || true
  done

  # Remove empty directories left behind after merge.
  find "$legacy_dir" -depth -type d -empty -delete 2>/dev/null || true

  if rmdir "$legacy_dir" 2>/dev/null; then
    echo "[keryx] Legacy model cache cleaned up: $legacy_dir" >&2
    return 0
  fi

  if [[ "${KERYX_PURGE_LEGACY_MODELS:-0}" == "1" ]]; then
    rm -rf "$legacy_dir" || true
    echo "[keryx] WARNING: force-purged legacy model cache at $legacy_dir (KERYX_PURGE_LEGACY_MODELS=1)." >&2
  else
    echo "[keryx] WARNING: legacy model cache still exists at $legacy_dir while shared cache exists at $shared_dir (possible duplicate disk usage). Set KERYX_PURGE_LEGACY_MODELS=1 to remove it automatically." >&2
  fi
}

mkdir -p "$KERYX_MODELS_DIR"
cleanup_legacy_models

# Some HiveOS images don't pre-create /var/log/miner — without this `tee` fails and the miner's
# output (including any crash reason) is lost.
mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME")"

./$CUSTOM_MINERBIN $(< $CUSTOM_CONFIG_FILENAME) $@ 2>&1 | tee $CUSTOM_LOG_BASENAME.log
