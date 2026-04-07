#!/usr/bin/env bash
# ================================================================
# dumbbell-topo-run-cont.sh — Simple wired dumbbell experiment
# Baselines: Cubic, Reno
# DRNN agent: scratch/drnn_agent_cont.py (SAC continuous cwnd)
# ================================================================
set -euo pipefail

N_EPISODES=30
FRESH=""
for arg in "$@"; do
    case "$arg" in
        --fresh) FRESH="fresh" ;;
        [0-9]*)  N_EPISODES="$arg" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SIM="simulation"
AGENT="scratch/drnn_agent_cont.py"

# -- Output directories --
OUT_BASE="../results/simple-dumbell"
CSV_DIR="$OUT_BASE/csvs"
GRAPH_DIR="$OUT_BASE/graphs"
mkdir -p "$CSV_DIR" "$GRAPH_DIR"

MODEL="$OUT_BASE/db_cont_model.pth"
TRAINLOG="$CSV_DIR/db_cont_train_log.csv"
PORT=5557
BOTTLENECK_MBPS=5.0
MIN_RTT_MS=48.0
CWND_MIN_KB=4.0
CWND_MAX_KB=32.0

echo "============================================================"
echo "  Simple Dumbbell TCP Experiment (Continuous SAC)"
echo "  Topology: 2 Src -> R1 -[5Mbps/20ms]- R2 -> 2 Dst"
echo "  Agent   : drnn_agent_cont.py (LSTM-SAC)"
echo "  Episodes: $N_EPISODES"
echo "  Output  : $OUT_BASE/"
echo "============================================================"

echo -e "\n[1/6] Building simulation..."
./ns3 build "$SIM"
echo "      Build OK."

echo -e "\n[2/6] Cubic baseline..."
./ns3 run "$SIM" -- --transport=cubic
for f in db_cubic_*.csv; do [ -f "$f" ] && mv "$f" "$CSV_DIR/"; done
echo "      Done -> $CSV_DIR/db_cubic_*.csv"

echo -e "\n[3/6] Reno baseline..."
./ns3 run "$SIM" -- --transport=reno
for f in db_reno_*.csv; do [ -f "$f" ] && mv "$f" "$CSV_DIR/"; done
echo "      Done -> $CSV_DIR/db_reno_*.csv"

if [ "$N_EPISODES" -gt 0 ]; then
    echo -e "\n[4/6] DRNN training ($N_EPISODES episodes)..."

    if [ "${FRESH:-}" = "fresh" ]; then
        rm -f "$MODEL" "$TRAINLOG"
        echo "      --fresh: cleared model and trainlog"
    fi

    for ep in $(seq 1 "$N_EPISODES"); do
        printf "      Episode %3d / %3d  " "$ep" "$N_EPISODES"

        lsof -ti :"$PORT" | xargs kill -9 2>/dev/null || true
        sleep 0.3

        FRESH_FLAG=""
        if [ "$ep" -eq 1 ] && [ "${FRESH:-}" = "fresh" ]; then
            FRESH_FLAG="--fresh"
        fi

        python3 "$AGENT" \
            --port="$PORT" \
            --model="$MODEL" \
            --trainlog="$TRAINLOG" \
            --bottleneck-mbps="$BOTTLENECK_MBPS" \
            --min-rtt-ms="$MIN_RTT_MS" \
            --cwnd-min-kb="$CWND_MIN_KB" \
            --cwnd-max-kb="$CWND_MAX_KB" \
            $FRESH_FLAG \
            > "/tmp/db_agent_ep${ep}.log" 2>&1 &
        AGENT_PID=$!

        waited=0
        while ! nc -z localhost "$PORT" 2>/dev/null; do
            sleep 0.5
            waited=$((waited + 1))
            if ! kill -0 "$AGENT_PID" 2>/dev/null; then
                echo "AGENT CRASHED!"
                tail -40 "/tmp/db_agent_ep${ep}.log"
                exit 1
            fi
            if [ $waited -gt 120 ]; then
                echo "TIMEOUT waiting for agent on port $PORT"
                kill "$AGENT_PID" 2>/dev/null || true
                tail -40 "/tmp/db_agent_ep${ep}.log"
                exit 1
            fi
        done

        ./ns3 run "$SIM" -- --transport=drnn --port="$PORT" \
            > "/tmp/db_ns3_ep${ep}.log" 2>&1 || {
            echo "NS-3 FAILED on episode $ep"
            tail -20 "/tmp/db_ns3_ep${ep}.log"
            kill "$AGENT_PID" 2>/dev/null || true
            exit 1
        }

        wait "$AGENT_PID" || true

        alpha=$(grep "SAC α" "/tmp/db_agent_ep${ep}.log" \
            | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "?")
        avg_r=$(grep "AvgReward=" "/tmp/db_agent_ep${ep}.log" \
            | tail -1 | grep -oP 'AvgReward=\K[-+0-9.]+' || echo "?")
        echo "done  (α=${alpha} avgR=${avg_r})"
    done

    echo -e "\n      Training complete. Model -> $MODEL"
    echo "      Training log -> $TRAINLOG"
else
    echo -e "\n[4/6] Skipping DRNN training (N_EPISODES=0)."
fi

echo -e "\n[5/6] DRNN evaluation..."
lsof -ti :"$PORT" | xargs kill -9 2>/dev/null || true
sleep 0.3

python3 "$AGENT" \
    --eval \
    --port="$PORT" \
    --model="$MODEL" \
    --trainlog="$TRAINLOG" \
    --bottleneck-mbps="$BOTTLENECK_MBPS" \
    --min-rtt-ms="$MIN_RTT_MS" \
    --cwnd-min-kb="$CWND_MIN_KB" \
    --cwnd-max-kb="$CWND_MAX_KB" \
    > /tmp/db_agent_eval.log 2>&1 &
AGENT_PID=$!

waited=0
while ! nc -z localhost "$PORT" 2>/dev/null; do
    sleep 0.5
    waited=$((waited + 1))
    if ! kill -0 "$AGENT_PID" 2>/dev/null; then
        echo "EVAL AGENT CRASHED!"
        tail -40 /tmp/db_agent_eval.log
        exit 1
    fi
    if [ $waited -gt 120 ]; then
        echo "TIMEOUT waiting for eval agent"
        kill "$AGENT_PID" 2>/dev/null || true
        tail -40 /tmp/db_agent_eval.log
        exit 1
    fi
done

./ns3 run "$SIM" -- --transport=drnn --port="$PORT" \
    > /tmp/db_ns3_eval.log 2>&1 || {
    echo "NS-3 FAILED during eval!"
    tail -20 /tmp/db_ns3_eval.log
    kill "$AGENT_PID" 2>/dev/null || true
    exit 1
}
wait "$AGENT_PID" || true
for f in db_drnn_*.csv; do [ -f "$f" ] && mv "$f" "$CSV_DIR/"; done
echo "      Eval done -> $CSV_DIR/db_drnn_*.csv"

echo -e "\n[6/6] Generating plots..."
cd "$OUT_BASE"
python3 dumbbell_plot.py

echo -e "\n============================================================"
echo "  Done!"
echo "  • $GRAPH_DIR/dumbbell_full_comparison.png"
echo "  • $GRAPH_DIR/dumbbell_training_curves.png"
echo "  • $CSV_DIR/db_*.csv"
echo "============================================================"
