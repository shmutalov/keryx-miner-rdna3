#!/usr/bin/env bash

# Self-locate the manifest from THIS script's own directory, so the package works under any folder
# name (versioned or not) with no hardcoded /hive/miners/custom/keryx-miner path and no symlink.
# No cd / no exit here: HiveOS may source this file.
__MD="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")" && pwd)"
. "$__MD/h-manifest.conf"

# The zero-dup binary talks to the GPU via Vulkan (mining AND in-process inference). The package
# BUNDLES the Vulkan loader (libvulkan.so.1, found via LD_LIBRARY_PATH in h-run.sh), so it starts
# even without libvulkan1. What it still needs from the system is an ICD *driver* — RADV
# (mesa-vulkan-drivers) for AMD; without one the miner finds 0 Vulkan devices and its probe exits.
if ! ls /usr/share/vulkan/icd.d/*.json >/dev/null 2>&1; then
    echo "keryx-miner-rdna3: no Vulkan ICD driver found — installing mesa-vulkan-drivers (RADV)"
    apt-get update -qq && apt-get install -y -qq mesa-vulkan-drivers libvulkan1 || \
        echo "WARNING: could not install a Vulkan driver; install one for your GPU (e.g. mesa-vulkan-drivers) or the miner will find 0 devices"
fi

conf=""
conf+=" -s $CUSTOM_URL --mining-address $CUSTOM_TEMPLATE"

[[ ! -z $CUSTOM_USER_CONFIG ]] && conf+=" $CUSTOM_USER_CONFIG"

echo "$conf"
echo "$conf" > $CUSTOM_CONFIG_FILENAME
