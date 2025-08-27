#!/usr/bin/env bash

set -euo pipefail

OUTFILE="benchmark.csv"
TARGET_DIR="/data/embeddings/"

source "./.venv/bin/activate"

if [ ! -f "$OUTFILE" ]; then
    echo "version,chunksize,target,cores,bs,sys,user,real" > "$OUTFILE"
fi

run_benchmark() {
    local version=$1
    local chunksize=$2
    local target=$3
    local cores=$4
    local batchsize=$5

    rm -rf "$TARGET_DIR"

    export CHUNKSIZE="$chunksize"
    export BVERSION="$version"

    local real user sys
    LC_NUMERIC=C
    TIMEFORMAT="%lR %lU %lS"
    { time python -m birdnet_analyzer.embeddings -i "$target" -db /data/embeddings -t "$cores" -b "$batchsize" 2> python_stderr.log ; } 2>timing.tmp

    read real user sys <timing.tmp

    echo "$version,$chunksize,$target,$cores,$batchsize,$sys,$user,$real" >> "$OUTFILE"

    rm -f timing.tmp
}

LARGE_FILES="/data/testing_audio/medium_sized_soundscapes/" # 155
SMALL_FILES="/data/testing_audio/small_files/" # 14018
TEST_CORES="10"

# warmup
run_benchmark "V1" "0" "$LARGE_FILES" "$TEST_CORES" "16"

# larger files
run_benchmark "V1" "0" "$LARGE_FILES" "$TEST_CORES" "16"
run_benchmark "V2" "1" "$LARGE_FILES" "$TEST_CORES" "16"
run_benchmark "V2" "2" "$LARGE_FILES" "$TEST_CORES" "16"
run_benchmark "V2" "3" "$LARGE_FILES" "$TEST_CORES" "16"
run_benchmark "V2" "7" "$LARGE_FILES" "$TEST_CORES" "16" # (155 files // 10 cores) // 2
run_benchmark "V2" "10" "$LARGE_FILES" "$TEST_CORES" "16"
run_benchmark "V2" "15" "$LARGE_FILES" "$TEST_CORES" "16" # 155 files // 10 cores
run_benchmark "V3" "1" "$LARGE_FILES" "$TEST_CORES" "16"
run_benchmark "V3" "2" "$LARGE_FILES" "$TEST_CORES" "16"
run_benchmark "V3" "3" "$LARGE_FILES" "$TEST_CORES" "16"
run_benchmark "V3" "10" "$LARGE_FILES" "$TEST_CORES" "16"
run_benchmark "V3" "10" "$LARGE_FILES" "$TEST_CORES" "16" # (155 files // 10 cores) // 2
run_benchmark "V3" "15" "$LARGE_FILES" "$TEST_CORES" "16" # 155 files // 10 cores

# small files
run_benchmark "V1" "0" "$SMALL_FILES" "$TEST_CORES" "16"
run_benchmark "V2" "1" "$SMALL_FILES" "$TEST_CORES" "16"
run_benchmark "V2" "2" "$SMALL_FILES" "$TEST_CORES" "16"
run_benchmark "V2" "3" "$SMALL_FILES" "$TEST_CORES" "16"
run_benchmark "V2" "7" "$SMALL_FILES" "$TEST_CORES" "16" # (14018 files // 10 cores) // 2
run_benchmark "V2" "700" "$SMALL_FILES" "$TEST_CORES" "16"
run_benchmark "V2" "1401" "$SMALL_FILES" "$TEST_CORES" "16" # 14018 files // 10 cores
run_benchmark "V3" "1" "$SMALL_FILES" "$TEST_CORES" "16"
run_benchmark "V3" "2" "$SMALL_FILES" "$TEST_CORES" "16"
run_benchmark "V3" "3" "$SMALL_FILES" "$TEST_CORES" "16"
run_benchmark "V3" "10" "$SMALL_FILES" "$TEST_CORES" "16"
run_benchmark "V3" "700" "$SMALL_FILES" "$TEST_CORES" "16" # (14018 files // 10 cores) // 2
run_benchmark "V3" "1401" "$SMALL_FILES" "$TEST_CORES" "16" # 14018 files // 10 cores