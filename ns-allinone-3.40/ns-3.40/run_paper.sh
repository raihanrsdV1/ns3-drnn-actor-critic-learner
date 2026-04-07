#!/usr/bin/env bash
# =================================================================
# run_paper.sh — Full 100-episode paper experiment
# Based on: arXiv:2508.01047v3
#
# Usage (from the ns-3.40/ directory):
#   ./run_paper.sh                  # 100 episodes, resume if model exists
#   ./run_paper.sh 50               # 50 episodes, resume if model exists
#   ./run_paper.sh 50 --fresh       # 50 episodes, discard existing model
#   ./run_paper.sh --fresh          # 100 episodes, discard existing model
#   ./run_paper.sh 0                # baselines only, no DRL training
# =================================================================
set -euo pipefail

N_EPISODES=100
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

OUT_BASE="../results/paper"
CSV_DIR="$OUT_BASE/csvs"
GRAPH_DIR="$OUT_BASE/graphs"
mkdir -p "$CSV_DIR" "$GRAPH_DIR"

AGENT="scratch/drnn_agent.py"
SIM="paper-sim"
PORT=5556
MODEL="$OUT_BASE/paper_drl_model.pth"
TRAINLOG="$CSV_DIR/drnn_train_log.csv"
# Reward parameters for the paper dumbbell topology
BOTTLENECK_MBPS=10.0
MIN_RTT_MS=42.0

echo "============================================================"
echo "  Paper Experiment: DRL-based TCP Congestion Control"
echo "  arXiv:2508.01047v3"
echo "  Training episodes: $N_EPISODES"
echo "  Output: paper_comparison.png"
echo "============================================================"

# ── 1. Build ──────────────────────────────────────────────────
echo -e "\n[1/4] Building paper-sim..."
./ns3 build "$SIM"
echo "      Build OK."

# ── 2. Baselines (no agent needed) ───────────────────────────
echo -e "\n[2/4] Running baselines..."
echo "      Cubic..."
./ns3 run "$SIM" -- --transport=cubic
for f in p_cubic_*.csv; do [ -f "$f" ] && mv "$f" "$CSV_DIR/"; done
echo "      Reno..."
./ns3 run "$SIM" -- --transport=reno
for f in p_reno_*.csv; do [ -f "$f" ] && mv "$f" "$CSV_DIR/"; done
echo "      Baselines done."

# ── 3. DRL training loop ──────────────────────────────────────
if [ "$N_EPISODES" -gt 0 ]; then
    echo -e "\n[3/4] DRL training ($N_EPISODES episodes)..."
    echo "      Each episode = 1 simulation run (~10s of sim time)"
    echo "      Epsilon: 1.0 → 0.05 over ~50 episodes, then near-greedy"
    echo ""

    # Remove stale model to start fresh if no model exists
    if [ ! -f "$MODEL" ]; then
        echo "      Starting fresh (no saved model found)."
    else
        echo "      Resuming from saved model: $MODEL"
    fi

    for ep in $(seq 1 "$N_EPISODES"); do
        printf "      Episode %3d / %3d  " "$ep" "$N_EPISODES"

        # Build agent flags: --fresh only on episode 1 when requested
        FRESH_FLAG=""
        if [ "$ep" -eq 1 ] && [ "${FRESH:-}" = "fresh" ]; then
            FRESH_FLAG="--fresh"
            echo "      (discarding any existing model — fresh start)"
        fi

        # Kill any stale process still holding the port from a previous run
        lsof -ti :"${PORT}" | xargs kill -9 2>/dev/null || true
        sleep 0.3

        # Start agent in background; redirect its output to a log file
        python3 "$AGENT" \
            --port="$PORT" \
            --model="$MODEL" \
            --trainlog="$TRAINLOG" \
            --bottleneck-mbps="$BOTTLENECK_MBPS" \
            --min-rtt-ms="$MIN_RTT_MS" \
            $FRESH_FLAG \
            > "/tmp/paper_agent_ep${ep}.log" 2>&1 &
        AGENT_PID=$!

        # Wait for agent to open port 5556 (max 50s — PyTorch import can be slow).
        # Also check the agent process is still alive every iteration so a crash
        # surfaces immediately with its log instead of hanging until timeout.
        waited=0
        while ! nc -z localhost "$PORT" 2>/dev/null; do
            sleep 0.5
            waited=$((waited + 1))
            # Detect silent crash (e.g. incompatible model, import error)
            if ! kill -0 "$AGENT_PID" 2>/dev/null; then
                echo "AGENT CRASHED before opening port!"
                echo "--- Last 30 lines of agent log ---"
                tail -30 "/tmp/paper_agent_ep${ep}.log"
                echo "----------------------------------"
                exit 1
            fi
            if [ $waited -gt 100 ]; then
                echo "TIMEOUT waiting for agent on port $PORT"
                kill "$AGENT_PID" 2>/dev/null || true
                echo "--- Last 30 lines of agent log ---"
                tail -30 "/tmp/paper_agent_ep${ep}.log"
                exit 1
            fi
        done

        # Run ns-3 simulation (blocks until done)
        ./ns3 run "$SIM" -- --transport=drl --port="$PORT" \
            > "/tmp/paper_ns3_ep${ep}.log" 2>&1 || {
            echo "NS-3 FAILED on episode $ep!"
            tail -20 "/tmp/paper_ns3_ep${ep}.log"
            kill "$AGENT_PID" 2>/dev/null || true
            exit 1
        }

        # Wait for agent to finish saving model
        wait "$AGENT_PID" || true

        # Show next epsilon from agent log (matches: "  Next epsilon: 0.8876")
        eps=$(grep "Next epsilon:" "/tmp/paper_agent_ep${ep}.log" \
              | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "?")
        echo "done  (next eps: ${eps})"
    done

    echo -e "\n      Training complete. Model saved to: $MODEL"
else
    echo -e "\n[3/4] Skipping DRL training (N_EPISODES=0)."
fi

# ── 4. Final evaluation (epsilon=0, frozen weights) ──────────
echo -e "\n[4/4] Final evaluation (epsilon=0, no exploration)..."

lsof -ti :"${PORT}" | xargs kill -9 2>/dev/null || true
sleep 0.3
python3 "$AGENT" \
    --eval \
    --port="$PORT" \
    --model="$MODEL" \
    --bottleneck-mbps="$BOTTLENECK_MBPS" \
    --min-rtt-ms="$MIN_RTT_MS" \
    > /tmp/paper_agent_eval.log 2>&1 &
AGENT_PID=$!

waited=0
while ! nc -z localhost "$PORT" 2>/dev/null; do
    sleep 0.5
    waited=$((waited + 1))
    if ! kill -0 "$AGENT_PID" 2>/dev/null; then
        echo "EVAL AGENT CRASHED before opening port!"
        echo "--- Last 30 lines of eval agent log ---"
        tail -30 /tmp/paper_agent_eval.log
        exit 1
    fi
    if [ $waited -gt 100 ]; then
        echo "TIMEOUT waiting for eval agent"
        kill "$AGENT_PID" 2>/dev/null || true
        tail -30 /tmp/paper_agent_eval.log
        exit 1
    fi
done

./ns3 run "$SIM" -- --transport=drl --name=p_drl_eval --port="$PORT"
wait "$AGENT_PID" || true

# Copy eval results to the standard names paper_plot.py expects
cp p_drl_eval_throughput.csv "$CSV_DIR/p_drl_throughput.csv"
cp p_drl_eval_rtt.csv "$CSV_DIR/p_drl_rtt.csv"
cp p_drl_eval_cwnd.csv "$CSV_DIR/p_drl_cwnd.csv"
cp p_drl_eval_drops.csv "$CSV_DIR/p_drl_drops.csv" 2>/dev/null || true
for f in p_drl_eval_*.csv; do [ -f "$f" ] && mv "$f" "$CSV_DIR/"; done
echo "      Eval complete. Results saved as p_drl_*.csv"

# ── Plot ──────────────────────────────────────────────────────
echo -e "\n============================================================"
echo "  Generating paper_comparison.png ..."
echo "============================================================"
cd "$OUT_BASE"
python3 paper_plot.py

echo -e "\n============================================================"
echo "  Done!"
echo "  • paper_comparison.png  — main figure"
echo "  • p_drl_*.csv           — DRL results (eval run)"
echo "  • p_cubic_*.csv         — Cubic baseline"
echo "  • p_reno_*.csv          — Reno baseline"
echo "============================================================"
