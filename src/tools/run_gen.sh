#!/usr/bin/env bash
# Runs gen_dataset.py for ONE split in chunks of $CHUNK new frames, a fresh Isaac process per chunk (each frame's
# timeline stop/play leaks renderer resources in Isaac 6.0: "[Fatal] [omni.rtx] Out of resource descriptors!" after
# 161-618 frames, then a hang). Each chunk resumes (gen_dataset --resume: saved frames kept, later frames identical to
# an uninterrupted run). If no new image is written for $STALL seconds anyway, that run is stopped and resumed.
#
#   ~/ip_ws/src/tools/run_gen.sh <out> <split> <frames> <seed> [more gen_dataset options]
#   e.g. ~/ip_ws/src/tools/run_gen.sh ~/ip_ws/datasets/box_v2 test 300 3 --palette heldout
set -u
OUT=$1; SPLIT=$2; FRAMES=$3; SEED=$4; shift 4
PY=~/isaac-sim-standalone-6.0.0-linux-x86_64/python.sh
GEN=~/ip_ws/src/tools/gen_dataset.py
STALL=${STALL:-150}          # s without a new image = hung (start-up + resume replay take ~70 s)
MAX_RUNS=${MAX_RUNS:-60}
CHUNK=${CHUNK:-100}          # new frames per process (lowest crash point seen: 161)
DIR=$OUT/$SPLIT

newest() { find "$DIR" -maxdepth 1 -name '*.jpg' -printf '%T@\n' 2>/dev/null | sort -n | tail -1; }
saved() { python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['images']))" "$DIR/_annotations.coco.json" 2>/dev/null || echo 0; }

for run in $(seq 1 "$MAX_RUNS"); do
    echo "[run_gen] $SPLIT: run $run, $(saved)/$FRAMES frames saved so far"
    setsid "$PY" "$GEN" --out "$OUT" --split "$SPLIT" --frames "$FRAMES" --seed "$SEED" --resume --max-new-frames "$CHUNK" "$@" &
    pid=$!                                   # leader of its own process group: python.sh and the Isaac python under it
    last_change=$(date +%s); last_seen=$(newest)
    while kill -0 "$pid" 2>/dev/null; do
        sleep 15
        now_seen=$(newest)
        if [ "$now_seen" != "$last_seen" ]; then last_seen=$now_seen; last_change=$(date +%s); fi
        if [ $(( $(date +%s) - last_change )) -gt "$STALL" ]; then
            echo "[run_gen] $SPLIT: no new image for ${STALL} s: stopping this run (process group $pid) and resuming"
            kill -TERM -- -"$pid" 2>/dev/null; sleep 15; kill -KILL -- -"$pid" 2>/dev/null
            break
        fi
    done
    wait "$pid" 2>/dev/null
    if [ "$(saved)" -ge "$FRAMES" ]; then
        echo "[run_gen] $SPLIT: complete, $(saved) frames, $run run(s)"
        exit 0
    fi
done
echo "[run_gen] $SPLIT: NOT complete after $MAX_RUNS runs ($(saved)/$FRAMES frames)"
exit 1
