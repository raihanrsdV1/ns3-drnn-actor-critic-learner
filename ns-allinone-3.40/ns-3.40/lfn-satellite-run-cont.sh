#!/usr/bin/env bash
# =================================================================
# lfn-satellite-run-cont.sh — Long Fat Network / Satellite link test
# Baselines: Cubic, Reno
# DRNN agent: scratch/drnn_agent_cont.py (LSTM-SAC continuous cwnd)
# =================================================================
set -euo pipefail

N_EPISODES=40
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

SIM="lfn-satellite-sim"
AGENT="scratch/drnn_agent_cont.py"
OUT_BASE="../results/lfn"
CSV_DIR="$OUT_BASE/csvs"
GRAPH_DIR="$OUT_BASE/graphs"
mkdir -p "$CSV_DIR" "$GRAPH_DIR"
MODEL="$OUT_BASE/lf_cont_model.pth"
TRAINLOG="$CSV_DIR/lf_cont_train_log.csv"
PORT=5557
BOTTLENECK_MBPS=1000.0
MIN_RTT_MS=250.0
CWND_MIN_KB=16.0
CWND_MAX_KB=32768.0

echo "============================================================"
echo "  LFN / Satellite TCP Experiment (Continuous SAC)"
echo "  Topology: sender -> R1 -[1Gbps,125ms]- R2 -> receiver"
echo "  Agent   : drnn_agent_cont.py (LSTM-SAC)"
echo "  BDP     : ~31.25 MB (1Gbps * 250ms RTT)"
echo "  Episodes: $N_EPISODES"
echo "  Output  : $OUT_BASE/"
echo "============================================================"

echo -e "\n[1/6] Building lfn-satellite-sim..."
./ns3 build "$SIM"
echo "      Build OK."

echo -e "\n[2/6] Cubic baseline..."
./ns3 run "$SIM" -- --transport=cubic
for f in lf_cubic_*.csv; do [ -f "$f" ] && mv "$f" "$CSV_DIR/"; done
echo "      Done -> lf_cubic_*.csv"

echo -e "\n[3/6] Reno baseline..."
./ns3 run "$SIM" -- --transport=reno
for f in lf_reno_*.csv; do [ -f "$f" ] && mv "$f" "$CSV_DIR/"; done
echo "      Done -> lf_reno_*.csv"

if [ "$N_EPISODES" -gt 0 ]; then
    echo -e "\n[4/6] DRNN training ($N_EPISODES episodes)..."

    if [ "${FRESH:-}" = "fresh" ]; then
        rm -f "$MODEL" "$TRAINLOG"
        echo "      --fresh: cleared $MODEL and $TRAINLOG"
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
            > "/tmp/lf_agent_ep${ep}.log" 2>&1 &
        AGENT_PID=$!

        waited=0
        while ! nc -z localhost "$PORT" 2>/dev/null; do
            sleep 0.5
            waited=$((waited + 1))
            if ! kill -0 "$AGENT_PID" 2>/dev/null; then
                echo "AGENT CRASHED!"
                tail -40 "/tmp/lf_agent_ep${ep}.log"
                exit 1
            fi
            if [ $waited -gt 160 ]; then
                echo "TIMEOUT waiting for agent on port $PORT"
                kill "$AGENT_PID" 2>/dev/null || true
                tail -40 "/tmp/lf_agent_ep${ep}.log"
                exit 1
            fi
        done

        ./ns3 run "$SIM" -- --transport=drnn --port="$PORT" \
            > "/tmp/lf_ns3_ep${ep}.log" 2>&1 || {
            echo "NS-3 FAILED on episode $ep"
            tail -20 "/tmp/lf_ns3_ep${ep}.log"
            kill "$AGENT_PID" 2>/dev/null || true
            exit 1
        }

        wait "$AGENT_PID" || true

        alpha=$(grep "SAC α" "/tmp/lf_agent_ep${ep}.log" \
            | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "?")
        avg_r=$(grep "AvgReward=" "/tmp/lf_agent_ep${ep}.log" \
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
    > /tmp/lf_agent_eval.log 2>&1 &
AGENT_PID=$!

waited=0
while ! nc -z localhost "$PORT" 2>/dev/null; do
    sleep 0.5
    waited=$((waited + 1))
    if ! kill -0 "$AGENT_PID" 2>/dev/null; then
        echo "EVAL AGENT CRASHED!"
        tail -40 /tmp/lf_agent_eval.log
        exit 1
    fi
    if [ $waited -gt 160 ]; then
        echo "TIMEOUT waiting for eval agent"
        kill "$AGENT_PID" 2>/dev/null || true
        tail -40 /tmp/lf_agent_eval.log
        exit 1
    fi
done

./ns3 run "$SIM" -- --transport=drnn --port="$PORT" \
    > /tmp/lf_ns3_eval.log 2>&1 || {
    echo "NS-3 FAILED during eval!"
    tail -20 /tmp/lf_ns3_eval.log
    kill "$AGENT_PID" 2>/dev/null || true
    exit 1
}
wait "$AGENT_PID" || true
for f in lf_drnn_*.csv; do [ -f "$f" ] && mv "$f" "$CSV_DIR/"; done
echo "      Eval done -> lf_drnn_*.csv"

echo -e "\n[6/6] Generating plots..."
cd "$OUT_BASE"
python3 lfn_satellite_plot.py

echo -e "\n============================================================"
echo "  Done!"
echo "  • $GRAPH_DIR/lfn_satellite_full_comparison.png"
echo "  • $GRAPH_DIR/lfn_satellite_training_curves.png"
echo "  • $CSV_DIR/lf_cubic_*.csv / lf_reno_*.csv / lf_drnn_*.csv"
echo "  • $CSV_DIR/lf_cont_train_log.csv"
echo "============================================================"
