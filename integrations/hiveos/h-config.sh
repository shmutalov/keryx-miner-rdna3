#!/usr/bin/env bash

# Self-locate the manifest from THIS script's own directory, so the package works under any folder
# name (versioned or not) with no hardcoded /hive/miners/custom/keryx-miner path and no symlink.
# No cd / no exit here: HiveOS may source this file.
__MD="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")" && pwd)"
. "$__MD/h-manifest.conf"

# The zero-dup binary talks to the GPU via Vulkan (mining AND in-process inference).
# Most current HiveOS images ship libvulkan1; install it if this one doesn't.
if ! ldconfig -p 2>/dev/null | grep -q libvulkan.so.1; then
    echo "keryx-miner: libvulkan1 not found — installing"
    apt-get update -qq && apt-get install -y -qq libvulkan1 || \
        echo "WARNING: could not install libvulkan1; the miner will not start without it"
fi

conf=""
conf+=" -s $CUSTOM_URL --mining-address $CUSTOM_TEMPLATE"

[[ ! -z $CUSTOM_USER_CONFIG ]] && conf+=" $CUSTOM_USER_CONFIG"

echo "$conf"
echo "$conf" > $CUSTOM_CONFIG_FILENAME
