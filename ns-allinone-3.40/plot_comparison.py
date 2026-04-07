#!/usr/bin/env python3
"""
TCP Protocol Comparison: DRNN vs Cubic vs Reno
Generates a multi-panel dashboard comparing throughput, RTT, CWND, and fairness.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

# -- Locate CSV data files --
def find_csv(name):
    for prefix in ["", "ns-3.40/"]:
        path = prefix + name
        if os.path.exists(path):
            return path
    return None

def load_csv(name, cols):
    path = find_csv(name)
    if path is None:
        print(f"  [MISSING] {name}")
        return None
    try:
        df = pd.read_csv(path, names=cols, header=0)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        if len(df) == 0:
            print(f"  [EMPTY]   {name}")
            return None
        print(f"  [OK]      {name} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"  [ERROR]   {name}: {e}")
        return None

# -- Load all data --
print("Loading CSV data...")
protocols = {}

drnn_tp = load_csv("drnn_throughput.csv", ["Time", "FlowID", "Throughput"])
drnn_rtt = load_csv("drnn_rtt.csv", ["Time", "FlowID", "RTT"])
drnn_cwnd = load_csv("drnn_cwnd.csv", ["Time", "FlowID", "Cwnd"])
if drnn_tp is not None:
    protocols["DRNN"] = {"tp": drnn_tp, "rtt": drnn_rtt, "cwnd": drnn_cwnd}

cubic_tp = load_csv("cubic_throughput.csv", ["Time", "FlowID", "Throughput"])
cubic_rtt = load_csv("cubic_rtt.csv", ["Time", "FlowID", "RTT"])
cubic_cwnd = load_csv("cubic_cwnd.csv", ["Time", "FlowID", "Cwnd"])
if cubic_tp is not None:
    protocols["Cubic"] = {"tp": cubic_tp, "rtt": cubic_rtt, "cwnd": cubic_cwnd}

reno_tp = load_csv("reno_throughput.csv", ["Time", "FlowID", "Throughput"])
reno_rtt = load_csv("reno_rtt.csv", ["Time", "FlowID", "RTT"])
reno_cwnd = load_csv("reno_cwnd.csv", ["Time", "FlowID", "Cwnd"])
if reno_tp is not None:
    protocols["Reno"] = {"tp": reno_tp, "rtt": reno_rtt, "cwnd": reno_cwnd}

if len(protocols) == 0:
    print("\nNo data files found! Run the simulations first.")
    sys.exit(1)

print(f"\nProtocols loaded: {list(protocols.keys())}")

# -- Colors and styling --
COLORS = {"DRNN": "#2ecc71", "Cubic": "#3498db", "Reno": "#e74c3c"}
plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
    'figure.facecolor': 'white', 'axes.grid': True,
    'grid.alpha': 0.3, 'lines.linewidth': 1.2
})

def aggregate(df, value_col, time_bin=0.5):
    df = df.copy()
    df['TimeBin'] = (df['Time'] / time_bin).round() * time_bin
    agg = df.groupby('TimeBin')[value_col].sum().reset_index()
    agg.columns = ['Time', value_col]
    return agg

def avg_across_flows(df, value_col, time_bin=0.5):
    df = df.copy()
    df['TimeBin'] = (df['Time'] / time_bin).round() * time_bin
    agg = df.groupby('TimeBin')[value_col].mean().reset_index()
    agg.columns = ['Time', value_col]
    return agg

def smooth(series, window=5):
    return series.rolling(window=window, min_periods=1, center=True).mean()

def jain_fairness_over_time(tp_df, time_bin=1.0):
    df = tp_df.copy()
    df['TimeBin'] = (df['Time'] / time_bin).round() * time_bin
    fairness = []
    for t, group in df.groupby('TimeBin'):
        tputs = group.groupby('FlowID')['Throughput'].mean().values
        if len(tputs) >= 2 and np.sum(tputs**2) > 0:
            jain = (np.sum(tputs)**2) / (len(tputs) * np.sum(tputs**2))
        else:
            jain = 1.0
        fairness.append({'Time': t, 'Fairness': jain})
    return pd.DataFrame(fairness)

# -- FIGURE 1: Main 6-panel dashboard --
fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.30)
fig.suptitle("TCP Congestion Control Comparison: DRNN vs Cubic vs Reno",
             fontsize=15, fontweight='bold', y=0.98)

# Panel 1: Aggregate Throughput
ax1 = fig.add_subplot(gs[0, 0])
for name, data in protocols.items():
    agg = aggregate(data['tp'], 'Throughput')
    ax1.plot(agg['Time'], smooth(agg['Throughput']), label=name, color=COLORS[name])
ax1.axhline(y=10.0, color='gray', linestyle='--', alpha=0.5, label='Link capacity')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Aggregate Throughput (Mbps)')
ax1.set_title('1. Aggregate Throughput')
ax1.legend(fontsize=8)
ax1.set_ylim(bottom=0)

# Panel 2: Average RTT
ax2 = fig.add_subplot(gs[0, 1])
for name, data in protocols.items():
    if data['rtt'] is not None:
        agg = avg_across_flows(data['rtt'], 'RTT')
        ax2.plot(agg['Time'], smooth(agg['RTT']), label=name, color=COLORS[name])
ax2.axhline(y=42.0, color='gray', linestyle='--', alpha=0.5, label='Min RTT (42 ms)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Average RTT (ms)')
ax2.set_title('2. Average Round-Trip Time')
ax2.legend(fontsize=8)
ax2.set_ylim(bottom=0)

# Panel 3: Average CWND per flow
ax3 = fig.add_subplot(gs[1, 0])
for name, data in protocols.items():
    if data['cwnd'] is not None:
        agg = avg_across_flows(data['cwnd'], 'Cwnd')
        ax3.plot(agg['Time'], smooth(agg['Cwnd'] / 1000), label=name, color=COLORS[name])
ax3.axhline(y=52.5, color='gray', linestyle='--', alpha=0.5, label='BDP (52.5 KB)')
ax3.axhline(y=52500/6/1000, color='orange', linestyle=':', alpha=0.5, label='Fair share (8.75 KB)')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Average CWND per flow (KB)')
ax3.set_title('3. Congestion Window Size')
ax3.legend(fontsize=8)
ax3.set_ylim(bottom=0)

# Panel 4: Jain's Fairness Index
ax4 = fig.add_subplot(gs[1, 1])
for name, data in protocols.items():
    fairness = jain_fairness_over_time(data['tp'])
    ax4.plot(fairness['Time'], smooth(fairness['Fairness'], window=3),
             label=name, color=COLORS[name])
ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel("Jain's Fairness Index")
ax4.set_title('4. Fairness Over Time')
ax4.legend(fontsize=8)
ax4.set_ylim(0.4, 1.05)

# Panel 5: Steady-state summary bars
ax5 = fig.add_subplot(gs[2, 0])
names = []
avg_tputs = []
avg_rtts = []
steady_start = 25.0

for name, data in protocols.items():
    names.append(name)
    tp_steady = data['tp'][data['tp']['Time'] >= steady_start]
    avg_tputs.append(tp_steady.groupby('Time')['Throughput'].sum().mean()
                     if len(tp_steady) > 0 else 0)
    if data['rtt'] is not None:
        rtt_steady = data['rtt'][data['rtt']['Time'] >= steady_start]
        avg_rtts.append(rtt_steady['RTT'].mean() if len(rtt_steady) > 0 else 0)
    else:
        avg_rtts.append(0)

x = np.arange(len(names))
bars1 = ax5.bar(x - 0.2, avg_tputs, 0.35, label='Throughput (Mbps)',
                color=[COLORS[n] for n in names], alpha=0.8)
ax5_twin = ax5.twinx()
bars2 = ax5_twin.bar(x + 0.2, avg_rtts, 0.35, label='Avg RTT (ms)',
                     color=[COLORS[n] for n in names], alpha=0.4, hatch='//')
ax5.set_ylabel('Throughput (Mbps)')
ax5_twin.set_ylabel('Avg RTT (ms)')
ax5.set_xticks(x)
ax5.set_xticklabels(names)
ax5.set_title(f'5. Steady-State Summary (t >= {steady_start}s)')
for bar, val in zip(bars1, avg_tputs):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.1f}', ha='center', fontsize=8)
for bar, val in zip(bars2, avg_rtts):
    ax5_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                  f'{val:.0f}ms', ha='center', fontsize=8)
lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax5_twin.get_legend_handles_labels()
ax5.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# Panel 6: CWND box plot
ax6 = fig.add_subplot(gs[2, 1])
box_data = []
box_labels = []
for name, data in protocols.items():
    if data['cwnd'] is not None:
        cwnd_steady = data['cwnd'][data['cwnd']['Time'] >= steady_start]
        if len(cwnd_steady) > 0:
            box_data.append(cwnd_steady['Cwnd'].values / 1000)
            box_labels.append(name)
if box_data:
    bp = ax6.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, name in zip(bp['boxes'], box_labels):
        patch.set_facecolor(COLORS.get(name, 'gray'))
        patch.set_alpha(0.6)
    ax6.axhline(y=52.5, color='gray', linestyle='--', alpha=0.5, label='BDP')
    ax6.set_ylabel('CWND (KB)')
    ax6.set_title('6. CWND Distribution (Steady State)')
    ax6.legend(fontsize=8)

plt.savefig('tcp_comparison_dashboard.png', dpi=150, bbox_inches='tight')
print("\nSaved: tcp_comparison_dashboard.png")

# -- FIGURE 2: Per-flow throughput --
if len(protocols) > 0:
    fig2, axes2 = plt.subplots(1, len(protocols), figsize=(6*len(protocols), 4),
                                squeeze=False)
    for idx, (name, data) in enumerate(protocols.items()):
        ax = axes2[0, idx]
        for fid in sorted(data['tp']['FlowID'].unique()):
            fdata = data['tp'][data['tp']['FlowID'] == fid]
            ax.plot(fdata['Time'], smooth(fdata['Throughput'], window=10),
                    label=f'Flow {int(fid)}', alpha=0.7, linewidth=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Throughput (Mbps)')
        ax.set_title(f'{name} - Per-Flow Throughput')
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tcp_per_flow_throughput.png', dpi=150, bbox_inches='tight')
    print("Saved: tcp_per_flow_throughput.png")

# -- Summary stats --
print("\n" + "=" * 70)
print("  COMPARISON SUMMARY (steady-state, t >= 25s)")
print("=" * 70)
print(f"{'Metric':<25} ", end="")
for name in protocols:
    print(f"{name:>12}", end="")
print()
print("-" * 70)

for metric_name, get_val in [
    ("Avg Throughput (Mbps)", lambda n: protocols[n]['tp'][protocols[n]['tp']['Time'] >= steady_start]
     .groupby('Time')['Throughput'].sum().mean() if protocols[n]['tp'] is not None else 0),
    ("Avg RTT (ms)", lambda n: protocols[n]['rtt'][protocols[n]['rtt']['Time'] >= steady_start]
     ['RTT'].mean() if protocols[n]['rtt'] is not None else 0),
    ("Avg CWND (KB)", lambda n: protocols[n]['cwnd'][protocols[n]['cwnd']['Time'] >= steady_start]
     ['Cwnd'].mean() / 1000 if protocols[n]['cwnd'] is not None else 0),
    ("RTT std dev (ms)", lambda n: protocols[n]['rtt'][protocols[n]['rtt']['Time'] >= steady_start]
     ['RTT'].std() if protocols[n]['rtt'] is not None else 0),
    ("Throughput std (Mbps)", lambda n: protocols[n]['tp'][protocols[n]['tp']['Time'] >= steady_start]
     .groupby('Time')['Throughput'].sum().std() if protocols[n]['tp'] is not None else 0),
]:
    print(f"{metric_name:<25} ", end="")
    for name in protocols:
        try:
            val = get_val(name)
            print(f"{val:>12.2f}", end="")
        except:
            print(f"{'N/A':>12}", end="")
    print()

print(f"{'Fairness (Jain avg)':<25} ", end="")
for name in protocols:
    fairness = jain_fairness_over_time(protocols[name]['tp'])
    fair_steady = fairness[fairness['Time'] >= steady_start]
    avg_fair = fair_steady['Fairness'].mean() if len(fair_steady) > 0 else 0
    print(f"{avg_fair:>12.4f}", end="")
print()

print("=" * 70)
print("""
Key Metrics Explained:
  * Throughput: Total data per second. Higher = better link utilization.
  * RTT: Round-trip latency. Lower = less queue buildup (bufferbloat).
    - Min RTT (42ms) = propagation delay only. Excess = queuing delay.
    - The paper's DRL achieved 46% lower RTT than Cubic.
  * CWND: Congestion window. Should be near BDP/nFlows for optimal perf.
    - BDP = 52,500 bytes (10 Mbps x 42ms). Fair share = ~8,750 bytes/flow.
    - Too high -> queue buildup -> RTT spikes -> packet loss -> throughput drop.
  * Fairness (Jain's Index): 1.0 = perfectly equal bandwidth sharing.
  * RTT std dev: Lower = more stable latency. DRL should be smoother.
""")

plt.show()
