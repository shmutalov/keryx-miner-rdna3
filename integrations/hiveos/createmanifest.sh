if [ "$#" -ne "2" ]
  then
    echo "No arguments supplied. Call using createmanifest.sh <VERSION_NUMBER> <MINER BINARY NAME>"
    exit
fi
cat > h-manifest.conf << EOF
# The name of the miner
CUSTOM_NAME=keryx-miner

# Optional version of your custom miner package
CUSTOM_VERSION=$1
CUSTOM_BUILD=0
CUSTOM_MINERBIN=$2

# Resolve the miner's ACTUAL install dir from this manifest's own location. This lets the package
# work under any folder name (e.g. the versioned "keryx-miner-v0.3.9-rdna3") WITHOUT a
# /hive/miners/custom/keryx-miner symlink. BASH_SOURCE[0] is this file's path even when sourced.
CUSTOM_MINER_DIR="\$(cd "\$(dirname "\$(readlink -f "\${BASH_SOURCE[0]:-\$0}")")" && pwd)"

# Full path to miner config file (inside the actual install dir, not a hardcoded /keryx-miner/)
CUSTOM_CONFIG_FILENAME="\$CUSTOM_MINER_DIR/config.ini"

# Full path to log file basename (without .log extension)
CUSTOM_LOG_BASENAME=/var/log/miner/\$CUSTOM_NAME

WEB_PORT=3338
EOF
