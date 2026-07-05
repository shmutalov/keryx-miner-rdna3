#!/usr/bin/env bash

# Self-locate the manifest from THIS script's own directory (works under a versioned folder, no
# symlink). No cd / no exit: HiveOS SOURCES this file and reads $khs / $stats from it afterwards,
# so changing the caller's cwd or calling exit would break the HiveOS agent.
__MD="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")" && pwd)"
. "$__MD/h-manifest.conf"

# Read the tail of the log ONCE and derive everything below from this in-memory copy, instead of
# re-reading the whole log file once per GPU. This keeps the script cheap on big rigs (12, 20+ GPUs).
# `tr -d '\000'` is a cheap guard in case the miner ever writes a stray NUL into the log.
log=`tail -n 4000 "$CUSTOM_LOG_BASENAME.log" 2>/dev/null | tr -d '\000'`

stats_raw=`grep "Current hashrate is" <<< "$log" | tail -n 1`

maxDelay=120
time_now=`date +%s`

# The miner logs with env_logger, whose default line starts "[2026-06-24T19:11:32Z INFO ...]"
# (ISO-8601 UTC, leading '['). Older builds logged "2026-06-24 19:11:32.000+02:00 [INFO ]".
# Pull the timestamp anywhere on the line (bracket/position independent) and let GNU date parse it
# (it understands both the T...Z form and the "date time+offset" form natively).
ts_field=`echo "$stats_raw" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2}([.][0-9]+)?(Z|[+-][0-9]{2}:?[0-9]{2})?' | head -1`
time_rep=`date -d "$ts_field" +%s 2>/dev/null || echo 0`
diffTime=`echo $((time_now-time_rep)) | tr -d '-'`

if [ "$diffTime" -lt "$maxDelay" ]; then
        # Value is second-to-last field (before unit), unit is last field.
        # The miner logs the rate with 2 decimals; dropping the dot then appending one 0 yields
        # rate*1000 (e.g. 3.83 -> "383" -> "3830"). NB: do NOT use `cut --output-delimiter=''` to
        # drop the dot — an empty output delimiter makes cut emit a NUL byte, which then trips
        # bash's "command substitution: ignored null byte" warning. `tr -d '.'` is clean.
        # HiveOS expects kilohashes (khs): Ghash/s = rate*1e6 khs = (rate*1000)*1e3, etc.
        total_hashrate=`echo $stats_raw | awk 'NF>=2{print $(NF-1)}' | tr -d '.' | sed 's/$/0/'`
        # Force base 10: a sub-1.0 rate yields a leading zero (e.g. 0.48 -> "0480") which bash would
        # otherwise parse as octal and reject ("value too great for base").
        total_hashrate=$((10#${total_hashrate:-0}))
        if [[ $stats_raw == *"Thash"* ]]; then
                total_hashrate=$(($total_hashrate*1000000))
        elif [[ $stats_raw == *"Ghash"* ]]; then
                total_hashrate=$(($total_hashrate*1000))
        elif [[ $stats_raw == *"Mhash"* ]]; then
                : # Mhash/s = rate*1e3 khs = rate*1000 already, no multiplier needed
        fi

        # GPU status
        readarray -t gpu_stats < <( jq --slurp -r -c '.[] | .busids, .brand, .temp, .fan | join(" ")' $GPU_STATS_JSON 2>/dev/null)
        busids=(${gpu_stats[0]})
        brands=(${gpu_stats[1]})
        temps=(${gpu_stats[2]})
        fans=(${gpu_stats[3]})
        gpu_count=${#busids[@]}

        hash_arr=()
        busid_arr=()
        fan_arr=()
        temp_arr=()

        if [ $(gpu-detect NVIDIA) -gt 0 ]; then
                BRAND_MINER="nvidia"
        elif [ $(gpu-detect AMD) -gt 0 ]; then
                BRAND_MINER="amd"
        fi

        # The miner keys its workers "Vulkan #N" by Vulkan device index, mining GPUs only.
        # HiveOS's busid list can also contain devices the miner never mines on — e.g. an
        # onboard iGPU. Using the raw loop index `i` as the miner device number then desyncs the
        # moment such a device is skipped (every later card reads one slot too high, the last one
        # falls off the end -> 0). Keep a SEPARATE counter that advances only for mining-brand
        # cards, so it tracks the miner's own numbering (an amd iGPU occupies one index in BOTH
        # lists, so the numberings stay aligned and its own slot truthfully reads 0).
        # No iGPU -> miner_dev == i.
        miner_dev=0
        for(( i=0; i < gpu_count; i++ )); do
                [[ "${brands[i]}" != $BRAND_MINER ]] && continue
                [[ "${busids[i]}" =~ ^([A-Fa-f0-9]+): ]]
                busid_arr+=($((16#${BASH_REMATCH[1]})))
                temp_arr+=(${temps[i]})
                fan_arr+=(${fans[i]})
                # Per-device line: "... Device Vulkan #N: 5.23 Ghash/s" — the worker id is
                # "Vulkan #N", so the trailing colon pins the number (no "#1" vs "#10" ambiguity).
                gpu_raw=`grep "Device Vulkan #$miner_dev:" <<< "$log" | tail -n 1`
                hashrate=`echo $gpu_raw | awk 'NF>=2{print $(NF-1)}' | tr -d '.' | sed 's/$/0/'`
                # Force base 10 (sub-1.0 rates yield a leading zero that bash would parse as octal).
                hashrate=$((10#${hashrate:-0}))
                if [[ $gpu_raw == *"Thash"* ]]; then
                        hashrate=$(($hashrate*1000000))
                elif [[ $gpu_raw == *"Ghash"* ]]; then
                        hashrate=$(($hashrate*1000))
                elif [[ $gpu_raw == *"Mhash"* ]]; then
                        : # Mhash/s = rate*1e3 khs = rate*1000 already, no multiplier needed
                fi
                hash_arr+=($hashrate)
                miner_dev=$((miner_dev+1))
        done

        hash_json=`printf '%s\n' "${hash_arr[@]}" | jq -cs '.'`
        bus_numbers=`printf '%s\n' "${busid_arr[@]}" | jq -cs '.'`
        fan_json=`printf '%s\n' "${fan_arr[@]}" | jq -cs '.'`
        temp_json=`printf '%s\n' "${temp_arr[@]}" | jq -cs '.'`

        uptime=$(( `date +%s` - `stat -c %Y $CUSTOM_CONFIG_FILENAME` ))

        stats=$(jq -nc \
                --argjson hs "$hash_json" \
                --arg ver "$CUSTOM_VERSION" \
                --arg ths "$total_hashrate" \
                --argjson bus_numbers "$bus_numbers" \
                --argjson fan "$fan_json" \
                --argjson temp "$temp_json" \
                --arg uptime "$uptime" \
                '{ hs: $hs, hs_units: "khs", algo: "keryxhash", ver: $ver, $uptime, $bus_numbers, $temp, $fan }')
        khs=$total_hashrate
else
        khs=0
        stats="null"
fi

echo "Log file : $CUSTOM_LOG_BASENAME.log"
echo "Time since last log entry : $diffTime"
echo "Raw stats : $stats_raw"
echo "KHS : $khs"
echo "Output : $stats"

[[ -z $khs ]] && khs=0
[[ -z $stats ]] && stats="null"
