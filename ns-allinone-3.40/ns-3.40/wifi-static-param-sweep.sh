#!/usr/bin/env bash
# =================================================================
# wifi-static-param-sweep.sh -- WiFi 802.11 Static Parameter Sweep
#
# Runs Cubic, Reno, and DRNN-SAC across parameter variations:
#   1. Number of nodes       (20, 40, 60, 80, 100)
#   2. Number of flows       (10, 20, 30, 40, 50)
#   3. Packets per second    (100, 200, 300, 400, 500)
#   4. Coverage multiplier   (1, 2, 3, 4, 5)  — area = mult × Tx_range
#
# Static nodes (no mobility) — Roll 46 assignment.
#
# All outputs organized under:  results/wifi-static/
#   results/wifi-static/csv/          — sweep results + training log
#   results/wifi-static/graphs/       — PNG plots
#   results/wifi-static/timeseries/   — per-run time-series CSVs
#
# Usage (from ns-3.40/ directory):
#   bash wifi-static-param-sweep.sh              # 20 training episodes
#   bash wifi-static-param-sweep.sh 50           # 50 training episodes
#   bash wifi-static-param-sweep.sh 0            # baselines only, skip DRNN
#   bash wifi-static-param-sweep.sh 20 --fresh   # discard saved model
# =================================================================
set -euo pipefail

# -- Parse arguments -----------------------------------------------
N_EPISODES=20
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

SIM="wifi-static-param-sim"
AGENT="scratch/drnn_agent_cont.py"

# -- Output directories --
OUT_BASE="../results/wifi-static"
CSV_DIR="$OUT_BASE/csv"
GRAPH_DIR="$OUT_BASE/graphs"
TS_DIR="$OUT_BASE/timeseries"
mkdir -p "$CSV_DIR" "$GRAPH_DIR" "$TS_DIR"

MODEL="$OUT_BASE/wifi_static_cont_model.pth"
TRAINLOG="$CSV_DIR/ws_cont_train_log.csv"
RESULTS="$CSV_DIR/ws_sweep_results.csv"
PORT=5558

# WiFi 802.11a infrastructure: AP + STAs with wired backhaul
# Bottleneck: AP -- R1 (10 Mbps, 10 ms)
BOTTLENECK_MBPS=10.0
MIN_RTT_MS=30.0
CWND_MIN_KB=2.0
CWND_MAX_KB=80.0

# Defaults (held constant when sweeping other params)
DEFAULT_NODES=10
DEFAULT_FLOWS=5
DEFAULT_PPS=200
DEFAULT_COVERAGE=1
SIM_TIME=30

# Sweep ranges
NODES_LIST="5 10 15 20 25"
FLOWS_LIST="2 5 10 15 20"
PPS_LIST="100 200 300 400 500"
COVERAGE_LIST="1 2 3 4 5"

echo "============================================================"
echo "  WiFi 802.11 Static Parameter Sweep (Roll 46)"
echo "  Topology: STAs -- AP --[10Mbps/10ms]-- R1 -- Dst"
echo "  Defaults: nNodes=$DEFAULT_NODES  nFlows=$DEFAULT_FLOWS  pps=$DEFAULT_PPS"
echo "            coverage=${DEFAULT_COVERAGE}x  (static, no mobility)"
echo "  DRNN training episodes: $N_EPISODES"
echo "  Output: $OUT_BASE/"
echo "============================================================"

# -- Helper: run a baseline (cubic/reno) simulation -----------------
run_baseline() {
    local transport=$1 nNodes=$2 nFlows=$3 pps=$4
    local coverage=${5:-$DEFAULT_COVERAGE}
    printf "  %-6s  nodes=%-3d flows=%-2d pps=%-3d cov=%-1d ... " \
           "$transport" "$nNodes" "$nFlows" "$pps" "$coverage"

    output=$(./ns3 run "$SIM" -- \
        --transport="$transport" \
        --nNodes="$nNodes" \
        --nFlows="$nFlows" \
        --pps="$pps" \
        --coverageMult="$coverage" \
        --simTime="$SIM_TIME" 2>&1) || {
        echo "FAILED"
        echo "$output" | tail -10
        return 1
    }

    # Move time-series CSVs to organized dir
    # IMPORTANT: do not reuse variable name 'f' (used by outer flow sweep loop)
    local ts_file
    for ts_file in ws_${transport}_*.csv; do
        [ -f "$ts_file" ] && mv "$ts_file" "$TS_DIR/"
    done

    # SUMMARY format: SUMMARY,transport,nNodes,nFlows,pps,coverageMult,
    #   throughput,delay,pdr,dropRatio,totalDrops,txPkts,rxPkts,energyJ
    summary=$(echo "$output" | grep "^SUMMARY," | head -1)
    if [ -n "$summary" ]; then
        echo "$summary" | sed 's/^SUMMARY,//' >> "$RESULTS"
        tput=$(echo "$summary" | cut -d',' -f7)
        delay=$(echo "$summary" | cut -d',' -f8)
        pdr_val=$(echo "$summary" | cut -d',' -f9)
        drop_val=$(echo "$summary" | cut -d',' -f10)
        energy=$(echo "$summary" | cut -d',' -f14)
        echo "done  tput=${tput}Mbps  delay=${delay}ms  PDR=${pdr_val}  drop=${drop_val}  energy=${energy}J"
    else
        echo "WARNING: no SUMMARY line"
    fi
}

# -- Helper: run DRNN (training or eval) ----------------------------
run_drnn() {
    local nNodes=$1 nFlows=$2 pps=$3
    local coverage=${4:-$DEFAULT_COVERAGE}
    local eval_flag=${5:-""}
    printf "  drnn   nodes=%-3d flows=%-2d pps=%-3d cov=%-1d ... " \
           "$nNodes" "$nFlows" "$pps" "$coverage"

    # Kill stale agent
    lsof -ti :"$PORT" | xargs kill -9 2>/dev/null || true
    sleep 0.3

    local extra=""
    [ "$eval_flag" = "eval" ] && extra="--eval"

    python3 "$AGENT" \
        --port="$PORT" \
        --model="$MODEL" \
        --trainlog="$TRAINLOG" \
        --bottleneck-mbps="$BOTTLENECK_MBPS" \
        --min-rtt-ms="$MIN_RTT_MS" \
        --cwnd-min-kb="$CWND_MIN_KB" \
        --cwnd-max-kb="$CWND_MAX_KB" \
        $extra \
        > /tmp/ws_agent.log 2>&1 &
    AGENT_PID=$!

    # Wait for agent port
    waited=0
    while ! nc -z localhost "$PORT" 2>/dev/null; do
        sleep 0.5
        waited=$((waited + 1))
        if ! kill -0 "$AGENT_PID" 2>/dev/null; then
            echo "AGENT CRASHED"
            tail -20 /tmp/ws_agent.log
            return 1
        fi
        if [ $waited -gt 120 ]; then
            echo "TIMEOUT"
            kill "$AGENT_PID" 2>/dev/null || true
            return 1
        fi
    done

    output=$(./ns3 run "$SIM" -- \
        --transport=drnn \
        --nNodes="$nNodes" \
        --nFlows="$nFlows" \
        --pps="$pps" \
        --coverageMult="$coverage" \
        --simTime="$SIM_TIME" \
        --port="$PORT" 2>&1) || {
        echo "NS3 FAILED"
        tail -10 <<< "$output"
        kill "$AGENT_PID" 2>/dev/null || true
        return 1
    }

    wait "$AGENT_PID" || true

    # Move time-series CSVs
    local ts_file
    for ts_file in ws_drnn_*.csv; do
        [ -f "$ts_file" ] && mv "$ts_file" "$TS_DIR/"
    done

    summary=$(echo "$output" | grep "^SUMMARY," | head -1)
    if [ -n "$summary" ]; then
        echo "$summary" | sed 's/^SUMMARY,//' >> "$RESULTS"
        tput=$(echo "$summary" | cut -d',' -f7)
        delay=$(echo "$summary" | cut -d',' -f8)
        pdr_val=$(echo "$summary" | cut -d',' -f9)
        drop_val=$(echo "$summary" | cut -d',' -f10)
        energy=$(echo "$summary" | cut -d',' -f14)
        echo "done  tput=${tput}Mbps  delay=${delay}ms  PDR=${pdr_val}  drop=${drop_val}  energy=${energy}J"
    else
        echo "WARNING: no SUMMARY line"
    fi
}

# ===================================================================
# Step 1: Build
# ===================================================================
echo -e "\n[1/7] Building $SIM ..."
./ns3 configure --build-profile=optimized --disable-examples --disable-tests >/dev/null 2>&1 || true
./ns3 build "$SIM"
echo "      Build OK."

# ===================================================================
# Step 2: Initialize results CSV
# ===================================================================
echo -e "\n[2/7] Initializing $RESULTS ..."
echo "Transport,nNodes,nFlows,PPS,CoverageMult,ThroughputMbps,AvgDelayMs,PDR,DropRatio,TotalDrops,TotalTx,TotalRx,EnergyJ" \
    > "$RESULTS"

# ===================================================================
# Step 3: Baseline sweeps (Cubic + Reno)
# ===================================================================
echo -e "\n[3/7] Running baseline sweeps (Cubic + Reno) ..."

echo -e "\n  --- Sweep: nNodes ($NODES_LIST) ---"
for n in $NODES_LIST; do
    run_baseline cubic "$n" "$DEFAULT_FLOWS" "$DEFAULT_PPS" "$DEFAULT_COVERAGE"
    run_baseline reno  "$n" "$DEFAULT_FLOWS" "$DEFAULT_PPS" "$DEFAULT_COVERAGE"
done

echo -e "\n  --- Sweep: nFlows ($FLOWS_LIST) ---"
for f in $FLOWS_LIST; do
    run_baseline cubic "$DEFAULT_NODES" "$f" "$DEFAULT_PPS" "$DEFAULT_COVERAGE"
    run_baseline reno  "$DEFAULT_NODES" "$f" "$DEFAULT_PPS" "$DEFAULT_COVERAGE"
done

echo -e "\n  --- Sweep: PPS ($PPS_LIST) ---"
for p in $PPS_LIST; do
    run_baseline cubic "$DEFAULT_NODES" "$DEFAULT_FLOWS" "$p" "$DEFAULT_COVERAGE"
    run_baseline reno  "$DEFAULT_NODES" "$DEFAULT_FLOWS" "$p" "$DEFAULT_COVERAGE"
done

echo -e "\n  --- Sweep: Coverage ($COVERAGE_LIST) ---"
for c in $COVERAGE_LIST; do
    run_baseline cubic "$DEFAULT_NODES" "$DEFAULT_FLOWS" "$DEFAULT_PPS" "$c"
    run_baseline reno  "$DEFAULT_NODES" "$DEFAULT_FLOWS" "$DEFAULT_PPS" "$c"
done

# ===================================================================
# Step 4: DRNN training (on default config)
# ===================================================================
if [ "$N_EPISODES" -gt 0 ]; then
    echo -e "\n[4/7] DRNN training ($N_EPISODES episodes on default config) ..."

    if [ "${FRESH:-}" = "fresh" ]; then
        rm -f "$MODEL" "$TRAINLOG"
        echo "      --fresh: cleared saved model and training log."
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
            > "/tmp/ws_agent_ep${ep}.log" 2>&1 &
        AGENT_PID=$!

        waited=0
        while ! nc -z localhost "$PORT" 2>/dev/null; do
            sleep 0.5
            waited=$((waited + 1))
            if ! kill -0 "$AGENT_PID" 2>/dev/null; then
                echo "AGENT CRASHED"
                tail -20 "/tmp/ws_agent_ep${ep}.log"
                exit 1
            fi
            if [ $waited -gt 120 ]; then
                echo "TIMEOUT"
                kill "$AGENT_PID" 2>/dev/null || true
                exit 1
            fi
        done

        ./ns3 run "$SIM" -- \
            --transport=drnn \
            --nNodes="$DEFAULT_NODES" \
            --nFlows="$DEFAULT_FLOWS" \
            --pps="$DEFAULT_PPS" \
            --coverageMult="$DEFAULT_COVERAGE" \
            --simTime="$SIM_TIME" \
            --port="$PORT" \
            > "/tmp/ws_ns3_ep${ep}.log" 2>&1 || {
            echo "NS-3 FAILED on episode $ep"
            tail -10 "/tmp/ws_ns3_ep${ep}.log"
            kill "$AGENT_PID" 2>/dev/null || true
            exit 1
        }

        wait "$AGENT_PID" || true

        avg_r=$(grep "AvgReward=" "/tmp/ws_agent_ep${ep}.log" \
                | tail -1 | grep -oP 'AvgReward=\K[-+0-9.]+' || echo "?")
        echo "done  (avgR=${avg_r})"
    done

    echo "      Training complete -> $MODEL"
else
    echo -e "\n[4/7] Skipping DRNN training (N_EPISODES=0)."
fi

# ===================================================================
# Step 5: DRNN eval sweeps
# ===================================================================
if [ -f "$MODEL" ]; then
    echo -e "\n[5/7] DRNN eval sweeps ..."

    echo -e "\n  --- Sweep: nNodes ($NODES_LIST) ---"
    for n in $NODES_LIST; do
        run_drnn "$n" "$DEFAULT_FLOWS" "$DEFAULT_PPS" "$DEFAULT_COVERAGE" eval
    done

    echo -e "\n  --- Sweep: nFlows ($FLOWS_LIST) ---"
    for f in $FLOWS_LIST; do
        run_drnn "$DEFAULT_NODES" "$f" "$DEFAULT_PPS" "$DEFAULT_COVERAGE" eval
    done

    echo -e "\n  --- Sweep: PPS ($PPS_LIST) ---"
    for p in $PPS_LIST; do
        run_drnn "$DEFAULT_NODES" "$DEFAULT_FLOWS" "$p" "$DEFAULT_COVERAGE" eval
    done

    echo -e "\n  --- Sweep: Coverage ($COVERAGE_LIST) ---"
    for c in $COVERAGE_LIST; do
        run_drnn "$DEFAULT_NODES" "$DEFAULT_FLOWS" "$DEFAULT_PPS" "$c" eval
    done
else
    echo -e "\n[5/7] Skipping DRNN eval (no trained model found)."
fi

# ===================================================================
# Step 6: Final comparison at default config (for time-series CSVs)
# ===================================================================
echo -e "\n[6/7] Final comparison at default config ..."
run_baseline cubic "$DEFAULT_NODES" "$DEFAULT_FLOWS" "$DEFAULT_PPS" "$DEFAULT_COVERAGE"
run_baseline reno  "$DEFAULT_NODES" "$DEFAULT_FLOWS" "$DEFAULT_PPS" "$DEFAULT_COVERAGE"
if [ -f "$MODEL" ]; then
    run_drnn "$DEFAULT_NODES" "$DEFAULT_FLOWS" "$DEFAULT_PPS" "$DEFAULT_COVERAGE" eval
fi

# ===================================================================
# Step 7: Generate plots
# ===================================================================
echo -e "\n[7/7] Generating plots ..."
cd ..
python3 results/wifi-static/wifi_static_param_plot.py

echo -e "\n============================================================"
echo "  Done!"
echo "  Sweep results   : $OUT_BASE/csv/ws_sweep_results.csv"
echo "  Training log    : $OUT_BASE/csv/ws_cont_train_log.csv"
echo "  Time-series CSV : $OUT_BASE/timeseries/"
echo "  Plots           : $OUT_BASE/graphs/"
echo "============================================================"
