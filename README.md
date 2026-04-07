# DRNN-SAC: Continuous-Action LSTM Soft Actor-Critic for TCP Congestion Control

A Deep Recurrent Neural Network (DRNN) agent using the Soft Actor-Critic (SAC) algorithm to learn TCP congestion control in the [ns-3](https://www.nsnam.org/) network simulator. The agent outputs a continuous target congestion window (cwnd) using LSTM-based temporal memory, replacing traditional loss-based heuristics like TCP Cubic and Reno.

**Author:** Mohammad Raihan Rashid (Student ID: 2105046)  
**Course:** CSE ns-3 Simulation Project, BUET  
**Framework:** ns-3.40 + [ns3-gym](https://github.com/tkn-tub/ns3-gym) (OpenAI Gym interface)

---

## Key Idea

Traditional TCP (Cubic, Reno) uses reactive AIMD — it floods the buffer until packets drop, then halves the window. This guarantees periodic packet loss and high latency.

DRNN-SAC takes a different approach:
- **Continuous action space** — outputs a precise float target cwnd (e.g., 7.3 KB) instead of discrete AIMD multipliers
- **LSTM temporal memory** — observes the last 8 timesteps to detect trends (e.g., rising RTT) and act preemptively
- **Entropy-regularized exploration** — SAC's auto-tuned temperature balances exploitation with exploration

The result: **competitive throughput with near-zero packet drops and minimal queuing delay** on topologies the agent was trained on.

---

## Repository Structure

```
ns3-project/
├── README.md
├── slides.tex                          # Full project report (LaTeX)
│
└── ns-allinone-3.40/
    └── ns-3.40/
        ├── scratch/                    # ns-3 simulation files
        │   ├── drnn_agent_cont.py      # DRNN-SAC Python agent (continuous)
        │   ├── drnn_agent.py           # DRNN DQN Python agent (discrete)
        │   ├── simulation.cc           # Wired dumbbell simulation
        │   ├── wifi-sim-cont.cc        # WiFi parking-lot (continuous SAC)
        │   ├── wifi-sim.cc             # WiFi parking-lot (discrete DQN)
        │   ├── multi-bottleneck-sim.cc # Multi-bottleneck topology
        │   ├── lfn-satellite-sim.cc    # Long Fat Network / satellite
        │   ├── fat-tree-incast-sim.cc  # Data-center fat-tree incast
        │   ├── paper-sim.cc            # Paper replication topology
        │   ├── wired-param-sim.cc      # Wired dumbbell parameter sweep
        │   └── wifi-static-param-sim.cc# WiFi 802.11 static parameter sweep
        │
        ├── dumbbell-topo-run-cont.sh   # Run script: simple dumbbell
        ├── multi-bottleneck-topo-run-cont.sh
        ├── lfn-satellite-run-cont.sh
        ├── fat-tree-incast-run-cont.sh
        ├── wifi-topo-run-cont.sh       # Run script: WiFi parking-lot (SAC)
        ├── wifi-topo-run.sh            # Run script: WiFi parking-lot (DQN)
        ├── run_paper.sh                # Run script: paper replication
        ├── wired-param-sweep.sh        # Parameter sweep: wired dumbbell
        └── wifi-static-param-sweep.sh  # Parameter sweep: WiFi static
    │
    └── results/                        # All outputs organized by topology
        ├── simple-dumbell/             # Wired dumbbell (2 src, 2 dst)
        │   ├── csvs/                   #   Time-series & training CSVs
        │   ├── graphs/                 #   Generated plots
        │   ├── dumbbell_plot.py        #   Plot script
        │   └── db_cont_model.pth       #   Trained SAC model
        │
        ├── multi-bottleneck/           # Multi-bottleneck + UDP cross-traffic
        │   ├── csvs/
        │   ├── graphs/
        │   ├── multi_bottleneck_plot.py
        │   └── mb_cont_model.pth
        │
        ├── lfn/                        # Long Fat Network / satellite
        │   ├── csvs/
        │   ├── graphs/
        │   ├── lfn_satellite_plot.py
        │   └── lf_cont_model.pth
        │
        ├── fat-tree/                   # Data-center fat-tree incast
        │   ├── csvs/
        │   ├── graphs/
        │   ├── fat_tree_incast_plot.py
        │   └── ft_cont_model.pth
        │
        ├── wifi-parkinglot/            # WiFi parking-lot (3 STA, AP)
        │   ├── csvs/
        │   ├── graphs/
        │   ├── wifi_full_plot.py
        │   └── wifi_cont_model.pth
        │
        ├── paper/                      # Paper replication experiment
        │   ├── csvs/
        │   ├── graphs/
        │   ├── paper_plot.py
        │   └── paper_drl_model.pth
        │
        ├── wired/                      # Wired dumbbell parameter sweep
        │   ├── csv/                    #   Sweep results + time-series
        │   ├── graphs/
        │   ├── timeseries/
        │   ├── wired_param_plot.py
        │   └── wired_cont_model.pth
        │
        ├── wifi-static/                # WiFi 802.11 static parameter sweep
        │   ├── csv/
        │   ├── graphs/
        │   ├── graphs_orig/            #   10s sim, 10-50 nodes
        │   ├── graphs_less_nodes/      #   30s sim, 5-25 nodes
        │   ├── timeseries/
        │   ├── wifi_static_param_plot.py
        │   └── wifi_static_cont_model.pth
        │
        └── cwnd_timeseries_plot.py     # Cross-topology cwnd comparison plots
```

---

## Topologies Tested

| Topology | Bottleneck | Flows | DRNN Outcome |
|---|---|---|---|
| **Wired Dumbbell** | 5-10 Mbps, 20ms | 2-50 | Matches throughput, 90% fewer drops, 63% lower RTT |
| **WiFi Parking-Lot** | 4 Mbps wired + WiFi | 3 | Near-zero drops after 100 episodes |
| **Multi-Bottleneck** | 6 Mbps, shallow queues + UDP | 4 | 2x throughput vs Cubic (aggressive learned policy) |
| **LFN / Satellite** | 1 Gbps, 250ms RTT | 1 | Identical to baselines (no congestion) |
| **Fat-Tree Incast** | 100 Mbps, 20-pkt queue | 12 | Collapse — RL too slow for microsecond bursts |
| **WiFi 802.11 Static** | 10 Mbps + WiFi 802.11a | 2-20 | Matches baselines at low flows; degrades at high flows |

---

## Prerequisites

- **ns-3.40** with [ns3-gym](https://github.com/tkn-tub/ns3-gym) (OpenGym module)
- **Python 3.8+** with: `torch`, `numpy`, `pandas`, `matplotlib`
- **Linux recommended** (ns3-gym is easier to configure on Linux; tested via Parallels on macOS)

```bash
pip install torch numpy pandas matplotlib
```

---

## Quick Start

All run scripts execute from the `ns-3.40/` directory.

### 1. Build ns-3

```bash
cd ns-allinone-3.40/ns-3.40
./ns3 configure --build-profile=optimized --enable-examples --enable-tests
./ns3 build
```

### 2. Run a topology experiment

Each topology has a dedicated run script that handles building, running baselines, training the DRNN agent, evaluation, and plot generation:

```bash
# Simple wired dumbbell (20 episodes)
bash dumbbell-topo-run-cont.sh 20

# WiFi parking-lot with continuous SAC (50 episodes)
bash wifi-topo-run-cont.sh 50

# Multi-bottleneck (100 episodes, fresh start)
bash multi-bottleneck-topo-run-cont.sh 100 --fresh

# Fat-tree incast (30 episodes)
bash fat-tree-incast-run-cont.sh 30

# LFN satellite (10 episodes)
bash lfn-satellite-run-cont.sh 10

# Baselines only (no DRNN training)
bash dumbbell-topo-run-cont.sh 0
```

### 3. Run parameter sweeps

```bash
# Wired dumbbell sweep (nodes, flows, PPS)
bash wired-param-sweep.sh 100

# WiFi 802.11 static sweep (nodes, flows, PPS, coverage)
bash wifi-static-param-sweep.sh 100
```

### 4. Generate plots from existing data

```bash
# From results/ directory:
cd ../results

# Sweep plots (auto-detect parameters from CSV)
python3 wired/wired_param_plot.py
python3 wifi-static/wifi_static_param_plot.py

# Per-topology comparison plots
python3 simple-dumbell/dumbbell_plot.py
python3 multi-bottleneck/multi_bottleneck_plot.py
python3 lfn/lfn_satellite_plot.py
python3 fat-tree/fat_tree_incast_plot.py

# Cwnd time-series comparison (wired + wifi)
python3 cwnd_timeseries_plot.py
```

---

## Agent Architecture

```
State Sequence (8 timesteps) ──┬──> Actor LSTM (2 layers, 64 units)
                               │       ├── FC Mean (μ)
                               │       └── FC Log-Std (σ)
                               │           └── Sample N(μ,σ) → tanh → Action (target cwnd)
                               │
                               └──> Twin Critic LSTMs
                                        ├── Q₁(s, a)
                                        └── Q₂(s, a)
```

**State vector per flow:** `[cwnd, RTT, bytes_in_flight]` + bottleneck drop counts  
**Action:** Single continuous float → target congestion window in bytes  
**Reward:** `R = 1.0 - P_underutil - P_rtt - P_drop` (bounded [-3, +1])

---

## Key Results

### Wired Dumbbell (Training Config: 20 flows, 200 PPS)

| Protocol | Throughput | Delay | PDR | Drops |
|---|---|---|---|---|
| TCP Cubic | 9.881 Mbps | 78.50 ms | 0.974 | 1,781 |
| TCP Reno | 9.881 Mbps | 72.74 ms | 0.979 | 1,375 |
| **DRNN-SAC** | **9.881 Mbps** | **78.35 ms** | **0.997** | **138** |

At low contention (10 flows): DRNN achieves **28 ms delay** (vs 77 ms Cubic) with **zero drops**.

### Limitations

- **Multi-flow degradation:** Agent trained on fixed flow count; performance drops when flows exceed training distribution
- **WiFi MAC opacity:** Agent cannot distinguish WiFi channel contention from bottleneck congestion
- **Microsecond-scale incast:** 0.1s observation interval is too slow for data-center micro-bursts

---

## Report

The full project report with all experiment results, analysis, and topology diagrams is in [slides.tex](slides.tex). Compile with:

```bash
pdflatex slides.tex
```

Note: Image paths in the LaTeX file reference `results/` subdirectories. Ensure the working directory is `ns3-project/` (or `ns-allinone-3.40/`, depending on where you place the compiled PDF).

---

## Acknowledgments

Inspired by: *"A Deep Reinforcement Learning-Based TCP Congestion Control Algorithm: Design, Simulation, and Evaluation"* (arXiv:2508.01047v3)
