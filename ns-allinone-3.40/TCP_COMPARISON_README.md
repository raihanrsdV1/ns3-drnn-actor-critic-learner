# TCP Comparison Study: Reno vs Cubic vs DRNN

This project compares three TCP congestion control algorithms on the same dumbbell topology with 6 competing flows.

## Topology
```
6 Senders → Gateway1 ←→ [10Mbps Bottleneck] ←→ Gateway2 → 6 Receivers
            100Mbps                                 100Mbps
```

## Metrics Collected
- **Throughput**: Per-flow throughput measured every 0.1s
- **RTT**: Round-trip time for each flow
- **Cwnd**: Congestion window size evolution

## Running the Simulations

### 1. Run TCP Reno
```bash
cd ns-3.40
./ns3 run "scratch/tcp-evolution --transport=TcpNewReno --name=reno"
```

### 2. Run TCP Cubic
```bash
./ns3 run "scratch/tcp-evolution --transport=TcpCubic --name=cubic"
```

### 3. Run DRNN (Two Terminals Required)

**Terminal 1 - Start C++ Simulation:**
```bash
./ns3 run scratch/drnn-simulator
```

**Terminal 2 - Start Python Agent (after seeing "START YOUR PYTHON AGENT NOW"):**
```bash
python3 scratch/agent.py
```

## Generate Comparison Graphs

After all three simulations complete, you should have these CSV files:
- `reno_throughput.csv`, `reno_rtt.csv`, `reno_cwnd.csv`
- `cubic_throughput.csv`, `cubic_rtt.csv`, `cubic_cwnd.csv`
- `drnn_throughput.csv`, `drnn_rtt.csv`, `drnn_cwnd.csv`

Then run:
```bash
python3 ../plot_comparison.py
```

This generates:
- **tcp_comparison.png** - Main comparison dashboard with 6 subplots:
  - Aggregate throughput over time
  - Average RTT over time
  - Average Cwnd over time
  - Throughput distribution (boxplot)
  - RTT distribution (boxplot)
  - Performance summary table (with fairness index)

- **per_flow_fairness.png** - Per-flow throughput for each algorithm

## Dependencies
```bash
pip install pandas matplotlib numpy
```

## Understanding the Results

### Throughput
- Higher is better
- Look for stability in steady state (after ~40s)

### RTT
- Lower is better
- Lower variance = more stable

### Fairness Index
- 1.0 = perfect fairness (all flows get equal share)
- < 0.8 = some flows are starved
- Jain's Fairness Index = (Σx)² / (n·Σx²)

### Expected Behavior
- **Reno**: Conservative, slower ramp-up, fair
- **Cubic**: Aggressive, faster convergence, may favor newer flows
- **DRNN**: Learns optimal policy (depends on training)

## File Structure
```
ns-3.40/
├── scratch/
│   ├── tcp-evolution.cc    # Reno/Cubic simulation
│   ├── drnn-simulator.cc   # DRNN with OpenGym
│   └── agent.py            # Python RL agent
├── *.csv                   # Output metrics
└── plot_comparison.py      # Graphing script
```

## Troubleshooting

**Issue**: "CSV files not found"
- Make sure you're running `plot_comparison.py` from the `ns-3.40/` directory
- Check that all three simulations completed successfully

**Issue**: DRNN simulation hangs
- Make sure Python agent is started within ~10s after C++ sim
- Check port 5555 is not blocked by firewall

**Issue**: Plots look empty
- Check CSV files have data: `head reno_throughput.csv`
- Verify simulation ran for 60 seconds

## Next Steps

To improve DRNN performance:
1. Implement actual TCP control in `ExecuteAction()` (modify ssthresh, cwnd)
2. Train with Deep Q-Network (DQN) or similar
3. Design better reward function (throughput/delay tradeoff)
4. Add more observations (packet loss, queue length)
