#!/usr/bin/env python3
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drnn_agent_architecture.png")

fig, ax = plt.subplots(figsize=(18, 11))
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis("off")


def box(x, y, w, h, text, fc="#f8fafc", ec="#334155", lw=1.8, fontsize=11, color="#0f172a"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        linewidth=lw, edgecolor=ec, facecolor=fc
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=color, wrap=True)
    return patch


def arrow(x1, y1, x2, y2, text=None, color="#475569", lw=1.8, ms=14,
          style="-|>", rad=0.0, text_dx=0.0, text_dy=0.16, fontsize=9.5):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
        linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}"
    )
    ax.add_patch(a)
    if text:
        ax.text((x1 + x2) / 2 + text_dx, (y1 + y2) / 2 + text_dy,
                text, ha="center", va="bottom", fontsize=fontsize, color=color,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.94))
    return a


def elbow_arrow(points, text=None, color="#475569", lw=1.8, ms=14,
                text_pos=None, fontsize=9.5):
    for (x1, y1), (x2, y2) in zip(points[:-2], points[1:-1]):
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw)
    (x1, y1), (x2, y2) = points[-2], points[-1]
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=ms, linewidth=lw, color=color,
                        connectionstyle="arc3,rad=0")
    ax.add_patch(a)
    if text:
        tx, ty = text_pos if text_pos is not None else ((points[0][0] + points[-1][0]) / 2,
                                                        (points[0][1] + points[-1][1]) / 2)
        ax.text(tx, ty, text, ha="center", va="bottom", fontsize=fontsize, color=color,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.94))
    return a


def label(x, y, text, color="#475569", fontsize=9, ha="center", va="bottom"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.94))


# Title
ax.text(9, 10.45, "DRNN-TCP Continuous Agent Architecture", ha="center", va="center",
        fontsize=22, weight="bold", color="#0f172a")
ax.text(9, 10.0, "drnn_agent_cont.py — Recurrent Soft Actor-Critic for congestion-window control",
        ha="center", va="center", fontsize=11, color="#475569")


# Left column: environment + preprocessing
box(0.7, 7.6, 2.9, 1.35,
    "ns-3 / ns3-gym\nSimulation\n\nObserves per-flow:\ncwnd, RTT, bytes-in-flight\n+ queue-drop counters",
    fc="#dbeafe", ec="#2563eb")
box(4.1, 7.75, 2.9, 1.05,
    "Observation Parser\nAuto-detects flow count\nState dim = 3N + 2",
    fc="#f1f5f9", ec="#475569")
box(4.1, 6.2, 2.9, 1.0,
    "Running Normalizer\nOnline mean / variance",
    fc="#f8fafc", ec="#64748b")
box(4.1, 4.65, 2.9, 1.0,
    "State History\nDeque of last 8 states",
    fc="#f8fafc", ec="#64748b")

box(0.9, 5.75, 2.5, 0.95,
    "Action Space\nBox[min_cwnd, max_cwnd]\n(read from ns-3 at startup)",
    fc="#e0f2fe", ec="#0284c7")
box(0.9, 4.1, 2.5, 0.95,
    "Reward Function\n1 - underutil - RTT excess - drops\n(computed in Python)",
    fc="#dcfce7", ec="#16a34a")


# Center: actor pipeline
box(8.0, 7.55, 3.25, 1.3,
    "LSTM Actor\nLayerNorm → LSTM (2 layers)\nShared fully connected trunk",
    fc="#fae8ff", ec="#a21caf")
box(8.35, 6.0, 2.55, 0.95,
    "Gaussian Heads\nmean + log_std",
    fc="#f5d0fe", ec="#c026d3")
box(8.35, 4.55, 2.55, 0.95,
    "sample() + tanh squash\naction in [-1, 1]",
    fc="#f5d0fe", ec="#c026d3")
box(8.35, 3.1, 2.55, 0.95,
    "tanh_to_cwnd()\nmap action to bytes",
    fc="#ede9fe", ec="#7c3aed")


# Right: critics and learning
box(12.7, 7.5, 3.55, 1.2,
    "Twin LSTM Critics\nQ1(s,a), Q2(s,a)",
    fc="#fee2e2", ec="#dc2626")
box(12.7, 5.65, 3.55, 1.1,
    "SAC Update\ncritic loss + actor loss\n+ entropy temperature α",
    fc="#fecaca", ec="#ef4444")
box(12.7, 4.0, 3.55, 0.95,
    "Target Critics\nsoft update (τ)",
    fc="#fff1f2", ec="#e11d48")
box(12.7, 2.15, 3.55, 1.15,
    "Sequence Replay Buffer\nstores (s, a, r, s', done)\ntrains off-policy",
    fc="#fef3c7", ec="#d97706")

box(13.2, 9.15, 2.75, 0.7,
    "Train mode: sample()\nEval mode: forward()",
    fc="#f8fafc", ec="#64748b", fontsize=10)


# Bottom notes — separate rows, no overlap
box(0.7, 0.7, 5.15, 1.15,
    "What the agent sees\n[cwnd, RTT, bytes-in-flight] for each flow\nplus 2 queue-drop counters",
    fc="#ecfeff", ec="#0891b2", fontsize=10)
box(6.35, 0.7, 5.15, 1.15,
    "Why RTT stays low\nThe reward penalizes RTT inflation and packet drops,\nwhile underutilization penalty prevents starvation.",
    fc="#f0fdf4", ec="#16a34a", fontsize=10)
box(12.0, 0.7, 5.25, 1.15,
    "Why SAC helps\nStochastic policy + auto-tuned entropy α\nkeeps exploration alive and avoids conservative collapse.",
    fc="#fff7ed", ec="#ea580c", fontsize=10)


# Main left-to-right flow
arrow(3.6, 8.28, 4.1, 8.28, "observation", text_dx=-0.08, text_dy=0.13, fontsize=9)
arrow(5.55, 7.75, 5.55, 7.2)
arrow(5.55, 6.2, 5.55, 5.65)
arrow(7.0, 8.28, 8.0, 8.28, "normalized sequence", text_dx=0.0, text_dy=0.13, fontsize=9)

# Actor vertical stack
arrow(9.62, 7.55, 9.62, 6.95)
arrow(9.62, 6.0, 9.62, 5.5)
arrow(9.62, 4.55, 9.62, 4.05)

# To environment and to critics
arrow(10.9, 3.58, 12.55, 3.58)
label(11.72, 3.76, "target cwnd", color="#475569", fontsize=9)
elbow_arrow([(8.35, 3.58), (7.35, 3.58), (7.35, 6.1), (3.6, 6.1)],
            color="#7c3aed")
label(5.2, 4.0, "env.step([cwnd])", color="#7c3aed", fontsize=9)
elbow_arrow([(3.6, 4.55), (7.1, 4.55), (7.1, 2.72), (12.65, 2.72)],
            color="#475569")
label(8.55, 4.78, "reward + next state", color="#475569", fontsize=9)

# Critic and training connections
arrow(11.25, 8.15, 12.7, 8.15)
label(11.98, 8.33, "state sequence", color="#475569", fontsize=9)
arrow(10.9, 6.45, 12.7, 6.2)
label(11.95, 6.48, "state, action", color="#475569", fontsize=9)
arrow(14.47, 7.5, 10.92, 6.45, color="#b91c1c", rad=0.0)
label(12.82, 7.02, "policy gradient signal", color="#b91c1c", fontsize=9)

arrow(14.47, 3.3, 14.47, 5.65)
label(15.38, 4.75, "sample batches", color="#475569", fontsize=9, ha="left")
arrow(14.47, 5.65, 14.47, 4.95)
arrow(14.47, 4.0, 14.47, 3.3)
label(15.18, 3.72, "soft updates", color="#475569", fontsize=9, ha="left")

plt.tight_layout()
fig.savefig(OUT, dpi=240, bbox_inches="tight")
print(f"Saved -> {OUT}")
