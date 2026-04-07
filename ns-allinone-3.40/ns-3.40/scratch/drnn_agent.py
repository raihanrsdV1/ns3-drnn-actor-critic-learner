#!/usr/bin/env python3
"""
drnn_agent.py  —  Universal LSTM-DQN Agent for ns-3 TCP Congestion Control
===========================================================================
Auto-adapts to ANY ns-3 simulation that exposes a 3-values-per-flow
observation convention:

    obs = [cwnd_bytes, rtt_ms, bytes_in_flight] × N_flows

The number of TCP flows (and therefore the state dimension) is inferred
automatically from the first env.reset() call, so this single file works
for every simulation without any code changes:

    Simulation          Flows   Bottleneck  Min RTT   Port
    ─────────────────────────────────────────────────────────────────
    wifi-sim.cc         3       4  Mbps     72  ms    5557
    paper-sim.cc        6       10 Mbps     42  ms    5555
    Any future sim      auto    CLI arg     CLI arg   CLI arg

Actions (7 discrete AIMD multipliers — same for all sims):
    0  FAST_DEC   (−4×)   drain queues aggressively
    1  SLOW_DEC   (−1×)   gentle cwnd pullback
    2  MAINTAIN   ( 0×)   hold cwnd constant
    3  AIMD       (+1×)   standard TCP congestion-avoidance growth
    4  MOD_INC    (+2×)   moderate acceleration
    5  FAST_INC   (+4×)   fast ramp-up
    6  VERY_FAST  (+8×)   startup / recovery ramp

Architecture:
    2-layer stacked LSTM (hidden=128) + Dueling DQN head
    Double DQN updates, Huber (SmoothL1) loss, gradient clipping
    Online running normalizer for zero-mean, unit-variance inputs

Reward (Python-computed, overrides C++ GetReward):
    R = tput_ratio                                [BIF/RTT / capacity,  0 → 1]
      − (AvgRTT/MinRTT − 1)                      [RTT excess above baseline]
      − 0.5 × max(0, TotalBIF/BDP − 1)           [packet-loss proxy: queue overflow]
    Clipped to [−3, +2].

    NO Jain fairness term — removed by user request.
    NO cwnd anywhere — only RTT and bytes_in_flight are used.

    Why this doesn't cancel to ~0 like previous versions:
      • throughput saturates at 1.0 once bottleneck is full
      • RTT excess starts at 0 (baseline) and grows independently
      • At current operating point (RTT≈120ms, BIF≈2×BDP):
          tput=1.0  rtt_excess=0.67  loss=0.5  →  R=−0.17
      • At optimal (RTT≈75ms, BIF≈BDP):
          tput=0.96  rtt_excess=0.04  loss=0.0  →  R=+0.92
      • 1.1-point spread gives the LSTM a clear gradient to improve.

Training log (--trainlog, one row per episode):
    Episode, Steps, AvgReward, AvgLoss, AvgTput_Mbps, AvgRTT_ms, Epsilon

Usage:
    # WiFi parking-lot (3 flows):
    python3 scratch/drnn_agent.py \\
        --port 5557 --model wifi_drnn_model.pth \\
        --bottleneck-mbps 4.0 --min-rtt-ms 72.0 \\
        --trainlog w_drnn_train_log.csv

    # Paper dumbbell (6 flows):
    python3 scratch/drnn_agent.py \\
        --port 5555 --model drnn_tcp_model.pth \\
        --bottleneck-mbps 10.0 --min-rtt-ms 42.0 \\
        --trainlog drnn_train_log.csv

    # Evaluation only (frozen weights, ε = 0):
    python3 scratch/drnn_agent.py --eval \\
        --port 5557 --model wifi_drnn_model.pth \\
        --bottleneck-mbps 4.0 --min-rtt-ms 72.0

    # Start completely fresh:
    python3 scratch/drnn_agent.py --fresh \\
        --port 5557 --model wifi_drnn_model.pth \\
        --bottleneck-mbps 4.0 --min-rtt-ms 72.0
"""

import os, sys, csv, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ── Optional imports (fail fast with a helpful message) ──────────────────────
try:
    import gym
    import ns3gym  # noqa: F401
except ImportError as e:
    print(f"  ERROR: Missing dependency — {e}")
    print("  Install with:  pip install ns3gym")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SEQUENCE REPLAY BUFFER  (fixed-length temporal windows for LSTM)
# ══════════════════════════════════════════════════════════════════════════════

class SequenceReplayBuffer:
    """
    Stores sliding windows of consecutive (s, a, r, s', done) tuples.
    The LSTM needs *sequences* — not isolated transitions — so it can
    learn temporal correlations such as 'RTT rising 3 steps → decrease cwnd'.
    """

    def __init__(self, capacity: int = 3000, seq_length: int = 8):
        self.buffer     = deque(maxlen=capacity)
        self.seq_length = seq_length
        self._ep: list  = []

    def push(self, state, action, reward, next_state, done):
        self._ep.append((state, action, reward, next_state, done))
        if len(self._ep) >= self.seq_length:
            self.buffer.append(list(self._ep[-self.seq_length:]))
        if done:
            self._ep = []

    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DRNN MODEL  (2-layer stacked LSTM + Dueling DQN head)
# ══════════════════════════════════════════════════════════════════════════════

class DRNN(nn.Module):
    """
    Deep Recurrent Neural Network for congestion control.

    Input  : (batch, seq_len, state_dim)   — normalised observation sequence
    Output : (batch, action_dim)           — Q-values for each action
             hidden                        — LSTM state (for online inference)

    Dueling architecture separates state value V(s) from advantage A(s,a):
        Q(s,a) = V(s) + A(s,a) − mean_a A(s,a)
    This helps the agent distinguish states that are inherently good/bad
    regardless of which action is taken (critical in TCP where the network
    state may be congested no matter what the agent does).
    """

    def __init__(self, state_dim: int, hidden_dim: int,
                 action_dim: int, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Per-feature normalisation stabilises LSTM gradients
        self.input_norm = nn.LayerNorm(state_dim)

        self.lstm = nn.LSTM(
            state_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

        # Value stream: V(s) — how good is this state?
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream: A(s,a) — how much better is each action?
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, x, hidden=None):
        B = x.size(0)
        x = self.input_norm(x)
        if hidden is None:
            h0 = torch.zeros(self.num_layers, B, self.hidden_dim)
            c0 = torch.zeros(self.num_layers, B, self.hidden_dim)
            hidden = (h0, c0)
        out, hidden = self.lstm(x, hidden)
        last = out[:, -1, :]                    # use last timestep output
        V = self.value_stream(last)
        A = self.advantage_stream(last)
        Q = V + A - A.mean(dim=1, keepdim=True) # dueling combination
        return Q, hidden


# ══════════════════════════════════════════════════════════════════════════════
# 3.  RUNNING NORMALIZER  (online Welford mean/var for stable LSTM inputs)
# ══════════════════════════════════════════════════════════════════════════════

class RunningNormalizer:
    """
    Keeps a running mean and variance (Welford's algorithm) so observations
    are normalised to zero-mean, unit-variance without a fixed dataset.

    Raw ns-3 observations span very different scales:
        cwnd           ~ 10 000 – 200 000 bytes
        rtt            ~ 40 – 300 ms
        bytes_in_flight~ 0 – 150 000 bytes
    Normalising them is critical for LSTM training stability.
    """

    def __init__(self, dim: int, clip: float = 5.0):
        self.mean  = np.zeros(dim, dtype=np.float64)
        self.var   = np.ones(dim,  dtype=np.float64)
        self.count = 1e-4          # warm start prevents div-by-zero
        self.clip  = clip

    def update(self, x):
        x     = np.asarray(x, dtype=np.float64)
        delta = x - self.mean
        self.count += 1
        self.mean  += delta / self.count
        self.var   += delta * (x - self.mean)

    def normalize(self, x):
        x   = np.asarray(x, dtype=np.float32)
        std = np.sqrt(self.var / max(self.count, 2) + 1e-8).astype(np.float32)
        return np.clip((x - self.mean.astype(np.float32)) / std,
                       -self.clip, self.clip)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  REWARD FUNCTION  (Throughput + RTT + Packet-loss — simple, no Jain)
# ══════════════════════════════════════════════════════════════════════════════

def compute_reward(obs:       np.ndarray,
                   bottleneck_bps: float,
                   min_rtt_ms:     float) -> float:
    """
    Three-component reward — BIF and RTT only, no cwnd, no Jain's fairness.

    R = tput_ratio  −  rtt_excess  −  0.5 × loss_excess

    tput_ratio  = min(Σ BIF/RTT / bottleneck, 1.0)
                  Actual delivery rate normalised to [0, 1].
                  Using BIF not cwnd: BIF/RTT saturates at bottleneck capacity,
                  so over-sending gives zero extra throughput reward.

    rtt_excess  = max(0, avgRTT/minRTT − 1)
                  0 at baseline, 0.67 at RTT=120 ms, 1.50 at RTT=180 ms.
                  Starts at 0, grows with queueing — does NOT cancel tput.

    loss_excess = 0.5 × max(0, totalBIF/BDP − 1)
                  BDP = bottleneck × minRTT (pipe capacity at zero queue).
                  Exceeding BDP means bytes are sitting in the FifoQueueDisc.
                  Penalises queue buildup BEFORE tail-drop fires.

    Operating points (WiFi: 4 Mbps, minRTT=72 ms, BDP=36000 bytes):
        Optimal  RTT=75 ms  BIF=BDP       → R = 0.96−0.04−0.00 = +0.92
        Good     RTT=85 ms  BIF=1.2×BDP   → R = 0.98−0.18−0.10 = +0.70
        Typical  RTT=120 ms BIF=2.0×BDP   → R = 1.00−0.67−0.50 = −0.17
        Bad      RTT=180 ms BIF=3.0×BDP   → R = 1.00−1.50−1.00 = −1.50

    avgR starts near −0.17 (over-saturated exploration) and should trend
    toward +0.7–0.9 as the agent learns to keep BIF near BDP.
    The ~1.1-point spread (optimal→typical) gives a clear learning signal.

    Parameters
    ──────────
    obs            : [cwnd0, rtt0, bif0, cwnd1, rtt1, bif1, …]  (cwnd unused)
    bottleneck_bps : bottleneck link in bytes/s  (4 Mbps → 500 000)
    min_rtt_ms     : propagation-only RTT in ms (72 for WiFi, 42 for paper)
    """
    n_flows = len(obs) // 3
    rtts    = [float(obs[i * 3 + 1]) for i in range(n_flows)]
    bifs    = [float(obs[i * 3 + 2]) for i in range(n_flows)]
    active  = [i for i in range(n_flows) if bifs[i] > 0 and rtts[i] > 0]

    if not active:
        return 0.0

    # ── 1. Throughput — BIF/RTT, normalised, capped at 1.0  ─────────────────
    total_Bps = sum(bifs[i] / (rtts[i] / 1000.0) for i in active)
    r_tput    = min(total_Bps / bottleneck_bps, 1.0)

    # ── 2. RTT excess above baseline  ────────────────────────────────────────
    avg_rtt   = float(np.mean([rtts[i] for i in active]))
    r_rtt     = -max(0.0, avg_rtt / min_rtt_ms - 1.0)

    # ── 3. Packet-loss proxy — queue overflow  (weight 0.5)  ─────────────────
    bdp       = bottleneck_bps * (min_rtt_ms / 1000.0)
    total_bif = sum(bifs[i] for i in active)
    r_loss    = -0.5 * max(0.0, total_bif / bdp - 1.0)

    return float(np.clip(r_tput + r_rtt + r_loss, -3.0, 2.0))


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TRAINING STEP  (Double DQN + Huber loss + gradient clipping)
# ══════════════════════════════════════════════════════════════════════════════

def train_step(policy_net: DRNN, target_net: DRNN,
               replay_buffer: SequenceReplayBuffer,
               optimizer: optim.Optimizer,
               batch_size: int, gamma: float) -> float:
    """
    Sample a mini-batch of sequences, compute Double DQN targets,
    back-propagate Huber loss, and clip gradients to norm ≤ 1.
    Returns the scalar loss (0.0 if buffer is too small).
    """
    if len(replay_buffer) < batch_size:
        return 0.0

    sequences    = replay_buffer.sample(batch_size)
    batch_losses = []

    for seq in sequences:
        # seq[t] = (state, action, reward, next_state, done)
        states      = torch.FloatTensor(
            np.array([s[0] for s in seq])).unsqueeze(0)    # (1, T, D)
        next_states = torch.FloatTensor(
            np.array([s[3] for s in seq])).unsqueeze(0)    # (1, T, D)
        last_action = torch.LongTensor([seq[-1][1]])
        last_reward = float(seq[-1][2])
        last_done   = float(seq[-1][4])

        # Current Q-value for the action taken
        q_vals, _   = policy_net(states)
        q_val       = q_vals[0, last_action]

        # Double DQN target: policy net selects action, target net evaluates it
        with torch.no_grad():
            nq_policy, _ = policy_net(next_states)
            best_action  = nq_policy.argmax(dim=1)
            nq_target, _ = target_net(next_states)
            max_nq       = nq_target[0, best_action]
            target_val   = last_reward + gamma * max_nq * (1.0 - last_done)

        batch_losses.append(nn.SmoothL1Loss()(q_val, target_val))

    total_loss = torch.stack(batch_losses).mean()
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()
    return total_loss.item()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── CLI ──────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Universal DRNN-TCP agent for ns-3 simulations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port",           type=int,   default=5555,
                        help="OpenGym port number")
    parser.add_argument("--model",          type=str,   default="drnn_model.pth",
                        help="Checkpoint file to load / save")
    parser.add_argument("--trainlog",       type=str,   default="drnn_train_log.csv",
                        help="Per-episode training metrics CSV (append mode)")
    parser.add_argument("--bottleneck-mbps",type=float, default=4.0,
                        help="Bottleneck link speed in Mbps (used to normalise reward)")
    parser.add_argument("--min-rtt-ms",     type=float, default=72.0,
                        help="Propagation-only RTT in ms (RTT without queueing delay)")
    parser.add_argument("--fresh",   action="store_true",
                        help="Ignore any saved checkpoint — start from scratch")
    parser.add_argument("--eval",    action="store_true",
                        help="Evaluation mode: load model, ε=0, no weight updates")
    args = parser.parse_args()

    # Derived constant
    bottleneck_bps = args.bottleneck_mbps * 1e6 / 8.0   # Mbps → bytes/sec

    # ── Hyper-parameters ─────────────────────────────────────────────────────
    HIDDEN_DIM    = 128
    NUM_LAYERS    = 2
    SEQ_LENGTH    = 8
    ACTION_DIM    = 7
    LR            = 1e-4     # lower LR for stable learning in noisy TCP env
    GAMMA         = 0.97
    EPSILON_START = 0.50
    EPSILON_END   = 0.02
    # Per-EPISODE decay: 0.97/episode → 0.50 → 0.27@ep20 → 0.11@ep40
    # Slower decay gives the agent more exploration episodes to discover
    # that holding cwnd near BDP is better than over-pushing.
    EPSILON_DECAY = 0.97    # per EPISODE
    BATCH_SIZE    = 32      # larger batch → lower gradient variance
    BUFFER_CAP    = 10000   # retain ~17 episodes of experience
    TARGET_UPDATE = 200     # ~3 target updates/episode (was 6) → stabler Q targets
    TRAIN_EVERY   = 4       # train every 4 steps → 145 updates/episode

    ACTION_NAMES = [
        "FAST_DEC", "SLOW_DEC", "MAINTAIN",
        "AIMD",     "MOD_INC",  "FAST_INC", "VERY_FAST",
    ]

    print("=" * 72)
    print("  DRNN-TCP — Universal LSTM-DQN Congestion Control Agent")
    print("=" * 72)
    print(f"  Mode:         {'EVALUATION' if args.eval else 'TRAINING'}")
    print(f"  Port:         {args.port}")
    print(f"  Model:        {args.model}")
    print(f"  Bottleneck:   {args.bottleneck_mbps:.1f} Mbps")
    print(f"  Min RTT:      {args.min_rtt_ms:.1f} ms")
    print(f"  Train log:    {args.trainlog}")
    print("=" * 72)

    # ── Connect to ns-3 OpenGym ───────────────────────────────────────────────
    print(f"\n  Connecting to ns-3 on port {args.port} ...")
    try:
        env = gym.make("ns3-v0", port=args.port, startSim=False)
    except Exception as exc:
        print(f"  ERROR — could not connect to ns-3: {exc}")
        sys.exit(1)
    print("  Connected.\n")

    obs = env.reset()
    obs = np.array(obs, dtype=np.float32)

    # ── Auto-detect dimensions ────────────────────────────────────────────────
    obs_size  = len(obs)
    num_flows = obs_size // 3          # 3 obs values per flow
    state_dim = obs_size
    print(f"  Auto-detected: obs_size={obs_size}  →  {num_flows} TCP flows"
          f"  (state_dim={state_dim})")

    # ── Build networks ────────────────────────────────────────────────────────
    policy_net = DRNN(state_dim, HIDDEN_DIM, ACTION_DIM, NUM_LAYERS)
    target_net = DRNN(state_dim, HIDDEN_DIM, ACTION_DIM, NUM_LAYERS)
    optimizer  = optim.Adam(policy_net.parameters(), lr=LR)
    buf        = SequenceReplayBuffer(BUFFER_CAP, SEQ_LENGTH)
    norm       = RunningNormalizer(state_dim)

    # ── Determine episode number from the training log (not checkpoint) ────
    # This is robust: even if checkpoint loading fails the episode counter
    # stays correct because it's derived from rows already written to the CSV.
    if not args.fresh and os.path.exists(args.trainlog):
        try:
            with open(args.trainlog) as _f:
                # count data rows (skip header)
                episode = max(0, sum(1 for ln in _f if ln.strip()) - 1)
        except Exception:
            episode = 0
    else:
        episode = 0

    epsilon     = EPSILON_START
    total_steps = 0

    # ── Load checkpoint ───────────────────────────────────────────────────────
    if args.fresh and os.path.exists(args.model):
        print(f"  --fresh: ignoring saved model '{args.model}'")
        target_net.load_state_dict(policy_net.state_dict())
    elif os.path.exists(args.model):
        try:
            # weights_only=False required for PyTorch >= 2.6
            ckpt        = torch.load(args.model, map_location="cpu",
                                     weights_only=False)
            policy_net.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            norm.mean   = ckpt.get("norm_mean",   norm.mean)
            norm.var    = ckpt.get("norm_var",    norm.var)
            norm.count  = ckpt.get("norm_count",  norm.count)
            # epsilon from checkpoint — episode counter comes from CSV above
            epsilon     = ckpt.get("epsilon",     EPSILON_START)
            total_steps = ckpt.get("total_steps", 0)
            target_net.load_state_dict(policy_net.state_dict())
            print(f"  Loaded checkpoint: CSV episode={episode}, "
                  f"epsilon={epsilon:.4f}, total_steps={total_steps}")
        except Exception as exc:
            print(f"  ERROR loading checkpoint: {exc}")
            print(f"  Starting with fresh weights (epsilon/steps reset).")
            target_net.load_state_dict(policy_net.state_dict())
    else:
        target_net.load_state_dict(policy_net.state_dict())
        print("  No saved model found — starting fresh.")

    target_net.eval()

    if args.eval:
        epsilon = 0.0
        policy_net.eval()
        print("  EVAL MODE: ε=0, weights frozen.\n")

    # ── Episode initialisation ────────────────────────────────────────────────
    norm.update(obs)
    state_history = deque(
        [norm.normalize(obs)] * SEQ_LENGTH, maxlen=SEQ_LENGTH
    )

    hidden        = None
    done          = False
    step          = 0
    total_reward  = 0.0
    action_counts = [0] * ACTION_DIM

    # Per-episode accumulators for the training log
    ep_rewards: list[float] = []
    ep_tputs:   list[float] = []
    ep_rtts:    list[float] = []
    ep_losses:  list[float] = []

    print(f"  Episode {episode + 1}  |  ε={epsilon:.4f}  |  "
          f"{num_flows} flows  |  bottleneck {args.bottleneck_mbps:.1f} Mbps")
    print(f"  {'Step':>5}  {'Mode':>7}  {'Action':<10}  "
          f"{'Reward':>8}  {'~Tput':>7}  {'AvgRTT':>7}  {'ε':>6}")
    print("  " + "─" * 60)

    # ── Main loop ─────────────────────────────────────────────────────────────
    while not done:
        step        += 1
        total_steps += 1

        # Build sequence tensor for LSTM
        state_seq    = np.array(list(state_history), dtype=np.float32)
        state_tensor = torch.FloatTensor(state_seq).unsqueeze(0)  # (1, T, D)

        # ── Action selection (ε-greedy) ──────────────────────────────────────
        # Always run the LSTM forward pass so the hidden state stays current
        # even during exploration steps. Without this, hidden is stale/zero
        # whenever we switch back to exploitation.
        policy_net.eval()
        with torch.no_grad():
            q_vals, hidden = policy_net(state_tensor, hidden)
        # Detach hidden states to prevent gradient graphs accumulating across
        # the entire episode (would corrupt gradients + leak memory).
        hidden = (hidden[0].detach(), hidden[1].detach())
        if not args.eval:
            policy_net.train()

        if not args.eval and random.random() < epsilon:
            action_idx = random.randrange(ACTION_DIM)
            mode       = "explore"
        else:
            action_idx = int(q_vals.argmax(dim=1).item())
            mode       = "exploit"

        action_counts[action_idx] += 1

        # ── Step ns-3 ────────────────────────────────────────────────────────
        next_obs, _cpp_reward, done, _ = env.step(action_idx)
        next_obs = np.array(next_obs, dtype=np.float32)

        # ── Python reward (richer signal than C++ GetReward) ─────────────────
        reward       = compute_reward(next_obs, bottleneck_bps,
                                      args.min_rtt_ms)
        total_reward += reward
        ep_rewards.append(reward)

        # Estimate per-step throughput & RTT for the training log
        act_rtts  = [float(next_obs[i*3+1]) for i in range(num_flows)
                     if next_obs[i*3] > 0 and next_obs[i*3+1] > 0]
        act_cwnds = [float(next_obs[i*3])   for i in range(num_flows)
                     if next_obs[i*3] > 0 and next_obs[i*3+1] > 0]
        if act_rtts:
            tput_mbps = (sum(c / (r / 1000.0) for c, r in zip(act_cwnds, act_rtts))
                         * 8.0 / 1e6)
            ep_tputs.append(tput_mbps)
            ep_rtts.append(float(np.mean(act_rtts)))
        else:
            ep_tputs.append(0.0)
            ep_rtts.append(0.0)

        # ── Normalise & push to buffer ────────────────────────────────────────
        norm.update(next_obs)
        norm_cur  = norm.normalize(obs)
        norm_next = norm.normalize(next_obs)
        state_history.append(norm_next)

        if not args.eval:
            buf.push(norm_cur, action_idx, reward, norm_next, done)

            # Train
            if step % TRAIN_EVERY == 0 and len(buf) >= BATCH_SIZE:
                loss = train_step(policy_net, target_net, buf,
                                  optimizer, BATCH_SIZE, GAMMA)
                ep_losses.append(loss)

            # Hard-update target network
            if total_steps % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # ── Console log every 10 steps ───────────────────────────────────────
        if step % 10 == 0:
            t_now = ep_tputs[-1] if ep_tputs else 0.0
            r_now = ep_rtts[-1]  if ep_rtts  else 0.0
            print(f"  {step:5d}  {mode:>7}  "
                  f"{ACTION_NAMES[action_idx]:<10}  "
                  f"{reward:+8.3f}  "
                  f"{t_now:7.2f}  {r_now:7.1f}  {epsilon:.4f}")

        obs      = next_obs

    # ── Episode summary ───────────────────────────────────────────────────────
    episode    += 1
    avg_r_ep    = float(np.mean(ep_rewards))                       if ep_rewards else 0.0
    avg_l_ep    = float(np.mean(ep_losses))                        if ep_losses  else 0.0
    avg_t_ep    = float(np.mean(ep_tputs))                         if ep_tputs   else 0.0
    avg_rtt_ep  = float(np.mean([r for r in ep_rtts if r > 0]))    if ep_rtts    else 0.0

    print(f"\n{'='*72}")
    print(f"  Episode {episode} complete")
    print(f"  Steps={step}  TotalReward={total_reward:.2f}  "
          f"AvgReward={avg_r_ep:.4f}  ε={epsilon:.4f}")
    print(f"  AvgTput≈{avg_t_ep:.2f} Mbps  AvgRTT≈{avg_rtt_ep:.1f} ms  "
          f"AvgLoss={avg_l_ep:.5f}")
    print(f"  Action distribution:")
    tot_acts = max(sum(action_counts), 1)
    for i, nm in enumerate(ACTION_NAMES):
        pct = 100 * action_counts[i] / tot_acts
        bar = "█" * int(pct / 2)
        print(f"    {nm:<12}: {action_counts[i]:4d}  ({pct:5.1f}%)  {bar}")
    print(f"{'='*72}")

    # ── Append row to training log (skip eval episodes) ────────────────────
    if not args.eval:
        write_hdr = (not os.path.exists(args.trainlog) or
                     os.path.getsize(args.trainlog) == 0)
        with open(args.trainlog, "a", newline="") as fh:
            w = csv.writer(fh)
            if write_hdr:
                w.writerow(["Episode", "Steps", "AvgReward", "AvgLoss",
                            "AvgTput_Mbps", "AvgRTT_ms", "Epsilon"])
            w.writerow([episode, step,
                        f"{avg_r_ep:.5f}",  f"{avg_l_ep:.5f}",
                        f"{avg_t_ep:.4f}",  f"{avg_rtt_ep:.2f}",
                        f"{epsilon:.5f}"])

    # ── Per-episode epsilon decay ───────────────────────────────────────────
    # Decay ONCE per episode so the agent has a full episode of exploration
    # before reducing its curiosity. This keeps epsilon meaningful across
    # 50-100 episodes instead of crashing to ~0 in the first episode.
    if not args.eval:
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # ── Save checkpoint ───────────────────────────────────────────────────────
    if not args.eval:
        torch.save({
            "model_state_dict":     policy_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "norm_mean":    norm.mean,
            "norm_var":     norm.var,
            "norm_count":   norm.count,
            "episode":      episode,
            "epsilon":      epsilon,
            "total_steps":  total_steps,
        }, args.model)
        print(f"  Model saved -> {args.model}")
        print(f"  Next epsilon: {epsilon:.5f}")

    env.close()
    print("  Done.\n")


if __name__ == "__main__":
    main()
