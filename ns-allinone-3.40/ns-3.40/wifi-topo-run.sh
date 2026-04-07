#!/usr/bin/env bash
# =================================================================
# wifi-topo-run.sh — Full WiFi Parking-Lot experiment
#
# Runs:
#   1. Build  wifi-sim
#   2. Cubic  baseline  (60 s simulation)
#   3. Reno   baseline  (60 s simulation)
#   4. DRNN   training  (N episodes, each 60 s sim)
#   5. DRNN   eval      (epsilon=0, frozen weights)
#   6. Generate comparison + training-curve plots
#
# Usage (from  ns-3.40/  directory):
#   ./wifi-topo-run.sh                  # 50 training episodes, resume model
#   ./wifi-topo-run.sh 30               # 30 episodes
#   ./wifi-topo-run.sh 50 --fresh       # 50 episodes, discard saved model
#   ./wifi-topo-run.sh --fresh          # 50 episodes, fresh start
#   ./wifi-topo-run.sh 0                # baselines only, no DRNN
# =================================================================
set -euo pipefail

# ── Parse arguments ──────────────────────────────────────────────
N_EPISODES=50
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

OUT_BASE="../results/wifi-parkinglot"
CSV_DIR="$OUT_BASE/csvs"
GRAPH_DIR="$OUT_BASE/graphs"
mkdir -p "$CSV_DIR" "$GRAPH_DIR"

SIM="wifi-sim"
AGENT="scratch/drnn_agent.py"
MODEL="$OUT_BASE/wifi_drnn_model.pth"
TRAINLOG="$CSV_DIR/w_drnn_train_log.csv"
PORT=5557
# Reward parameters for the WiFi parking-lot topology
BOTTLENECK_MBPS=4.0
MIN_RTT_MS=72.0

echo "============================================================"
echo "  WiFi Parking-Lot TCP Experiment"
echo "  Topology: 3-STA WiFi → AP → R1 ─[8M]─ R2 ─[4M]─ Server"
echo "  Training episodes : $N_EPISODES"
echo "  Simulation time   : 60 s per episode"
echo "============================================================"

# ── Step 1: Build ────────────────────────────────────────────────
echo -e "\n[1/6] Building wifi-sim..."
./ns3 build "$SIM"
echo "      Build OK."

# ── Step 2: Cubic baseline ───────────────────────────────────────
echo -e "\n[2/6] Cubic baseline..."
./ns3 run "$SIM" -- --transport=cubic
for f in w_cubic_*.csv; do [ -f "$f" ] && mv "$f" "$CSV_DIR/"; done
echo "      Done → w_cubic_*.csv"

# ── Step 3: Reno baseline ────────────────────────────────────────
echo -e "\n[3/6] Reno baseline..."
./ns3 run "$SIM" -- --transport=reno
for f in w_reno_*.csv; do [ -f "$f" ] && mv "$f" "$CSV_DIR/"; done
echo "      Done → w_reno_*.csv"

# ── Step 4: DRNN training loop ───────────────────────────────────
if [ "$N_EPISODES" -gt 0 ]; then

    echo -e "\n[4/6] DRNN training ($N_EPISODES episodes)..."
    echo "      Per-episode metrics → $TRAINLOG"
    echo ""

    # On --fresh: wipe training log so graphs start from ep 1
    if [ "${FRESH:-}" = "fresh" ]; then
        rm -f "$MODEL" "$TRAINLOG"
        echo "      --fresh: cleared saved model and training log."
    fi

    for ep in $(seq 1 "$N_EPISODES"); do
        printf "      Episode %3d / %3d  " "$ep" "$N_EPISODES"

        # Kill any stale process on the port
        lsof -ti :"$PORT" | xargs kill -9 2>/dev/null || true
        sleep 0.3

        # Fresh flag only for episode 1 when --fresh requested
        FRESH_FLAG=""
        if [ "$ep" -eq 1 ] && [ "${FRESH:-}" = "fresh" ]; then
            FRESH_FLAG="--fresh"
        fi

        # Start agent in background
        python3 "$AGENT" \
            --port="$PORT" \
            --model="$MODEL" \
            --trainlog="$TRAINLOG" \
            --bottleneck-mbps="$BOTTLENECK_MBPS" \
            --min-rtt-ms="$MIN_RTT_MS" \
            $FRESH_FLAG \
            > "/tmp/wifi_agent_ep${ep}.log" 2>&1 &
        AGENT_PID=$!

        # Wait for agent to open the port (max 60 s — PyTorch + LSTM can be slow)
        waited=0
        while ! nc -z localhost "$PORT" 2>/dev/null; do
            sleep 0.5
            waited=$((waited + 1))
            if ! kill -0 "$AGENT_PID" 2>/dev/null; then
                echo "AGENT CRASHED before opening port!"
                echo "--- Last 40 lines of agent log ---"
                tail -40 "/tmp/wifi_agent_ep${ep}.log"
                exit 1
            fi
            if [ $waited -gt 120 ]; then
                echo "TIMEOUT waiting for agent on port $PORT"
                kill "$AGENT_PID" 2>/dev/null || true
                tail -40 "/tmp/wifi_agent_ep${ep}.log"
                exit 1
            fi
        done

        # Run ns-3 simulation (blocks until done — 60 s sim ≈ 3–10 s wall time)
        ./ns3 run "$SIM" -- \
            --transport=drnn \
            --port="$PORT" \
            > "/tmp/wifi_ns3_ep${ep}.log" 2>&1 || {
            echo "NS-3 FAILED on episode $ep!"
            tail -20 "/tmp/wifi_ns3_ep${ep}.log"
            kill "$AGENT_PID" 2>/dev/null || true
            exit 1
        }

        # Wait for agent to finish saving model
        wait "$AGENT_PID" || true

        # Extract next epsilon from agent log
        eps=$(grep "Next epsilon:" "/tmp/wifi_agent_ep${ep}.log" \
              | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "?")
        avg_r=$(grep "AvgReward=" "/tmp/wifi_agent_ep${ep}.log" \
                | tail -1 | grep -oP 'AvgReward=\K[-+0-9.]+' || echo "?")
        echo "done  (eps=${eps}  avgR=${avg_r})"
    done

    echo -e "\n      Training complete. Model → $MODEL"
    echo "      Training log  → $TRAINLOG"

else
    echo -e "\n[4/6] Skipping DRNN training (N_EPISODES=0)."
fi

# ── Step 5: DRNN evaluation ──────────────────────────────────────
echo -e "\n[5/6] DRNN evaluation (epsilon=0, frozen weights)..."

lsof -ti :"$PORT" | xargs kill -9 2>/dev/null || true
sleep 0.3

python3 "$AGENT" \
    --eval \
    --port="$PORT" \
    --model="$MODEL" \
    --trainlog="$TRAINLOG" \
    --bottleneck-mbps="$BOTTLENECK_MBPS" \
    --min-rtt-ms="$MIN_RTT_MS" \
    > /tmp/wifi_agent_eval.log 2>&1 &
AGENT_PID=$!

waited=0
while ! nc -z localhost "$PORT" 2>/dev/null; do
    sleep 0.5
    waited=$((waited + 1))
    if ! kill -0 "$AGENT_PID" 2>/dev/null; then
        echo "EVAL AGENT CRASHED!"
        tail -40 /tmp/wifi_agent_eval.log
        exit 1
    fi
    if [ $waited -gt 120 ]; then
        echo "TIMEOUT waiting for eval agent"
        kill "$AGENT_PID" 2>/dev/null || true
        tail -40 /tmp/wifi_agent_eval.log
        exit 1
    fi
done

./ns3 run "$SIM" -- \
    --transport=drnn \
    --port="$PORT" \
    > /tmp/wifi_ns3_eval.log 2>&1 || {
    echo "NS-3 FAILED during eval!"
    tail -20 /tmp/wifi_ns3_eval.log
    kill "$AGENT_PID" 2>/dev/null || true
    exit 1
}

wait "$AGENT_PID" || true
for f in w_drnn_*.csv; do [ -f "$f" ] && mv "$f" "$CSV_DIR/"; done
echo "      Eval done → w_drnn_*.csv"

# ── Step 6: Generate plots ───────────────────────────────────────
echo -e "\n[6/6] Generating plots..."
cd "$OUT_BASE"
python3 wifi_full_plot.py

echo -e "\n============================================================"
echo "  Done!"
echo "  • wifi_full_comparison.png  — Cubic vs Reno vs DRNN"
echo "  • wifi_training_curves.png  — DRNN training progress"
echo "  • w_cubic_*.csv             — Cubic baseline"
echo "  • w_reno_*.csv              — Reno baseline"
echo "  • w_drnn_*.csv              — DRNN eval results"
echo "  • w_drnn_train_log.csv      — Per-episode training metrics"
echo "============================================================"
