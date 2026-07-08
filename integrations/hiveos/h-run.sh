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

# Some HiveOS images don't pre-create /var/log/miner — without this `tee` fails and the miner's
# output (including any crash reason) is lost.
mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME")"

./$CUSTOM_MINERBIN $(< $CUSTOM_CONFIG_FILENAME) $@ 2>&1 | tee $CUSTOM_LOG_BASENAME.log
