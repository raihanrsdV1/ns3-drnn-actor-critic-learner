#!/usr/bin/env python3
"""
DRNN-TCP Agent: Deep Recurrent Neural Network for TCP Congestion Control
=========================================================================
This agent implements an LSTM-based Deep Q-Network (DQN) that learns to
control the congestion window (cwnd) of 6 competing TCP flows over an
ns-3 simulated dumbbell topology.

Architecture:
  - LSTM layer captures temporal patterns in network state sequences
  - DQN framework (policy net + target net) for stable Q-learning
  - Sequence replay buffer stores windows of consecutive states
  - Epsilon-greedy exploration with decay

Observation (18 floats):
  For each of 6 flows: [cwnd, rtt_ms, bytes_in_flight]

Actions (7 discrete AIMD multipliers — all LINEAR growth/decay):
  0 = Fast decrease   (-4x AIMD rate, drain queues)
  1 = Slow decrease   (-1x AIMD rate, gentle pullback)
  2 = Maintain        (hold cwnd steady)
  3 = AIMD            (+1x standard TCP congestion avoidance)
  4 = Moderate increase (+2x AIMD rate)
  5 = Fast increase   (+4x AIMD rate)
  6 = Very fast       (+8x AIMD rate, startup recovery)

The reward function balances:
  - High throughput (cwnd utilization)
  - Low latency (penalize RTT inflation)
  - Fairness across flows (Jain's fairness index)
"""

import gym
import ns3gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import time
from collections import deque
import random

# ============================================================
# 1. SEQUENCE REPLAY BUFFER (stores temporal windows for LSTM)
# ============================================================
class SequenceReplayBuffer:
    """
    Stores fixed-length sequences of transitions for LSTM training.
    Unlike a standard replay buffer that stores individual (s,a,r,s') tuples,
    this stores windows of consecutive states so the LSTM can learn temporal
    patterns like 'RTT has been rising for 3 steps -> decrease cwnd'.
    """
    def __init__(self, capacity=2000, seq_length=8):
        self.buffer = deque(maxlen=capacity)
        self.seq_length = seq_length
        self.current_episode = []

    def push(self, state, action, reward, next_state, done):
        self.current_episode.append((state, action, reward, next_state, done))
        if len(self.current_episode) >= self.seq_length:
            sequence = list(self.current_episode[-self.seq_length:])
            self.buffer.append(sequence)
        if done:
            self.current_episode = []

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# ============================================================
# 2. DRNN MODEL (LSTM + Dueling DQN)
# ============================================================
class DRNN(nn.Module):
    """
    Deep Recurrent Neural Network for congestion control.

    Input:  (batch, seq_len, 18) -- sequence of network state snapshots
    Output: (batch, 7) -- Q-values for each action

    The LSTM learns temporal correlations:
      - Rising RTT trend => network congestion building
      - Falling cwnd + stable RTT => recovering from loss
      - High bytes_in_flight + high RTT => buffer bloat
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(DRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input normalization (learned)
        self.input_norm = nn.LayerNorm(input_dim)

        # Stacked LSTM for deeper temporal feature extraction
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=0.1)

        # Dueling DQN head: separate value and advantage streams
        # This helps the agent learn which states are inherently good/bad
        # independently from which actions are good/bad
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x, hidden=None):
        batch_size = x.size(0)

        # Normalize inputs across feature dimension
        x = self.input_norm(x)

        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
            hidden = (h0, c0)

        lstm_out, hidden = self.lstm(x, hidden)

        # Use last timestep output
        last_out = lstm_out[:, -1, :]

        # Dueling architecture: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        value = self.value_stream(last_out)
        advantage = self.advantage_stream(last_out)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values, hidden


# ============================================================
# 3. STATE NORMALIZER (running statistics for stable learning)
# ============================================================
class RunningNormalizer:
    """
    Keeps running mean/std of observations for online normalization.
    Raw ns-3 observations can have wildly different scales:
      cwnd ~ 10000-200000 bytes, rtt ~ 40-200 ms, inflight ~ 0-150000
    Normalizing to zero-mean, unit-variance helps the LSTM learn faster.
    """
    def __init__(self, dim, clip=5.0):
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.count = 1e-4
        self.clip = clip

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        delta = x - self.mean
        self.count += 1
        self.mean += delta / self.count
        self.var += delta * (x - self.mean)

    def normalize(self, x):
        x = np.asarray(x, dtype=np.float32)
        std = np.sqrt(self.var / max(self.count, 2) + 1e-8).astype(np.float32)
        return np.clip((x - self.mean.astype(np.float32)) / std,
                       -self.clip, self.clip)


# ============================================================
# 4. REWARD FUNCTION (computed in Python for richer shaping)
# ============================================================
def compute_reward(obs, prev_obs, action, bottleneck_bw_bytes=1250000):
    """
    Multi-objective reward that balances throughput, latency, and fairness.

    Components:
      R_throughput: Reward for high link utilization
      R_latency:    Penalty for RTT inflation above minimum
      R_fairness:   Bonus for equal bandwidth sharing (Jain's index)
      R_smooth:     Penalty for oscillating between actions

    Args:
        obs: current state [cwnd0, rtt0, inflight0, cwnd1, rtt1, inflight1, ...]
        prev_obs: previous state (same format)
        action: action taken
        bottleneck_bw_bytes: bottleneck capacity in bytes/sec (10Mbps = 1,250,000 B/s)
    """
    n_flows = 6

    # Extract per-flow metrics
    cwnds = [obs[i * 3] for i in range(n_flows)]
    rtts = [obs[i * 3 + 1] for i in range(n_flows)]
    inflights = [obs[i * 3 + 2] for i in range(n_flows)]

    prev_rtts = [prev_obs[i * 3 + 1] for i in range(n_flows)] if prev_obs is not None else rtts

    active_flows = [i for i in range(n_flows) if cwnds[i] > 0 and rtts[i] > 0]

    if len(active_flows) == 0:
        return 0.0

    # --- Component 1: Throughput reward ---
    # Approximate throughput per flow = cwnd / rtt
    # Normalize by bottleneck capacity
    total_throughput = 0.0
    for i in active_flows:
        rtt_sec = rtts[i] / 1000.0  # ms -> seconds
        if rtt_sec > 0:
            flow_tput = cwnds[i] / rtt_sec  # bytes/sec
            total_throughput += flow_tput

    # Target: use close to 100% of bottleneck (10 Mbps), cap at 1.0
    utilization = min(total_throughput / bottleneck_bw_bytes, 1.0)
    r_throughput = utilization  # 0 to 1.0 — no bonus for overshooting

    # --- Component 2: Latency penalty (STRONG — paper's key advantage) ---
    # Min RTT ~42ms. Paper achieved 46% lower RTT than Cubic.
    # Penalize as soon as RTT rises >10% above minimum.
    min_rtt = 42.0
    avg_rtt = np.mean([rtts[i] for i in active_flows])
    rtt_ratio = avg_rtt / min_rtt if min_rtt > 0 else 1.0
    r_latency = -0.8 * max(0, rtt_ratio - 1.1)  # Heavy penalty for any RTT inflation

    # --- Component 3: Fairness bonus (Jain's fairness index) ---
    # Weighted heavily (0.4) because fairness was a visible problem
    if len(active_flows) >= 2:
        tputs = []
        for i in active_flows:
            rtt_sec = rtts[i] / 1000.0
            if rtt_sec > 0:
                tputs.append(cwnds[i] / rtt_sec)
            else:
                tputs.append(0)
        tputs = np.array(tputs)
        if np.sum(tputs ** 2) > 0:
            jain = (np.sum(tputs) ** 2) / (len(tputs) * np.sum(tputs ** 2))
        else:
            jain = 1.0
        r_fairness = 0.3 * jain  # Jain's fairness index bonus
    else:
        r_fairness = 0.3

    # --- Component 4: CWND over-inflation penalty ---
    # BDP = 10Mbps * 42ms = 52,500 bytes. No flow should exceed this.
    # Excess cwnd just fills the router queue for zero throughput gain.
    bdp_cap = 52500  # 1x BDP
    r_cwnd_penalty = 0.0
    for i in active_flows:
        if cwnds[i] > bdp_cap:
            # Proportional penalty: how far over the cap are we?
            overshoot = (cwnds[i] - bdp_cap) / bdp_cap
            r_cwnd_penalty -= 0.5 * overshoot  # Strong penalty for queue buildup

    # --- Component 5: Smoothness penalty ---
    # Penalize large RTT jumps (sign of oscillation)
    rtt_changes = []
    for i in active_flows:
        if prev_rtts[i] > 0:
            change = abs(rtts[i] - prev_rtts[i]) / prev_rtts[i]
            rtt_changes.append(change)
    avg_rtt_change = np.mean(rtt_changes) if rtt_changes else 0
    r_smooth = -0.1 * avg_rtt_change

    # --- Total reward ---
    reward = r_throughput + r_latency + r_fairness + r_cwnd_penalty + r_smooth

    return float(reward)


# ============================================================
# 5. TRAINING STEP
# ============================================================
def train_step(policy_net, target_net, replay_buffer, optimizer, batch_size, gamma):
    """
    Train on a batch of sequences from the replay buffer.
    Each sequence is fed through the LSTM so it can learn temporal patterns.
    Uses Double DQN + Huber loss (SmoothL1) for stability.
    """
    if len(replay_buffer) < batch_size:
        return 0.0

    sequences = replay_buffer.sample(batch_size)
    batch_losses = []

    for seq in sequences:
        states = torch.FloatTensor(np.array([s[0] for s in seq])).unsqueeze(0)
        next_states = torch.FloatTensor(np.array([s[3] for s in seq])).unsqueeze(0)
        last_action = torch.LongTensor([seq[-1][1]])
        last_reward = seq[-1][2]
        last_done = float(seq[-1][4])

        # Q(s, a) from policy network
        q_values, _ = policy_net(states)
        q_value = q_values[0, last_action]

        # Double DQN: use policy net to SELECT action, target net to EVALUATE
        with torch.no_grad():
            next_q_policy, _ = policy_net(next_states)
            best_action = next_q_policy.argmax(dim=1)
            next_q_target, _ = target_net(next_states)
            max_next_q = next_q_target[0, best_action]
            target_q = last_reward + gamma * max_next_q * (1 - last_done)

        loss = nn.SmoothL1Loss()(q_value, target_q)
        batch_losses.append(loss)

    total_loss = torch.stack(batch_losses).mean()

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss.item()


# ============================================================
# 6. MAIN
# ============================================================
def main():
    print("=" * 80)
    print("  DRNN-TCP: Deep Recurrent Neural Network for Congestion Control")
    print("  LSTM-based DQN agent controlling 6 competing TCP flows")
    print("=" * 80)

    # === HYPERPARAMETERS ===
    STATE_DIM = 18          # 6 flows x 3 metrics
    ACTION_DIM = 7          # 7 discrete cwnd adjustment levels
    HIDDEN_DIM = 128        # LSTM hidden units
    NUM_LAYERS = 2          # Stacked LSTM layers
    SEQ_LENGTH = 8          # LSTM looks back 8 steps
    LR = 0.0003
    GAMMA = 0.97            # Discount factor (high = long-term thinking)
    EPSILON_START = 0.5     # Moderate exploration — safe AIMD default handles startup
    EPSILON_END = 0.02
    EPSILON_DECAY = 0.985   # Faster decay -> reach exploitation sooner
    BATCH_SIZE = 16
    BUFFER_CAPACITY = 3000
    TARGET_UPDATE = 15      # Update target net every N steps
    PORT = 5555

    ACTION_NAMES = [
        "FAST_DEC", "SLOW_DEC", "MAINTAIN",
        "AIMD", "MODERATE", "FAST_INC", "VERY_FAST"
    ]

    print(f"\n{'Config':=^60}")
    print(f"  State dim:     {STATE_DIM} (6 flows x [cwnd, rtt, inflight])")
    print(f"  Action dim:    {ACTION_DIM}")
    print(f"  LSTM layers:   {NUM_LAYERS}, hidden: {HIDDEN_DIM}")
    print(f"  Sequence len:  {SEQ_LENGTH} steps (temporal memory window)")
    print(f"  Learning rate: {LR}")
    print(f"  Gamma:         {GAMMA}")
    print(f"  Epsilon:       {EPSILON_START} -> {EPSILON_END} (decay {EPSILON_DECAY})")
    print(f"{'':=^60}\n")

    # === Initialize networks ===
    policy_net = DRNN(STATE_DIM, HIDDEN_DIM, ACTION_DIM, NUM_LAYERS)
    target_net = DRNN(STATE_DIM, HIDDEN_DIM, ACTION_DIM, NUM_LAYERS)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
    replay_buffer = SequenceReplayBuffer(BUFFER_CAPACITY, SEQ_LENGTH)
    normalizer = RunningNormalizer(STATE_DIM)

    # === Connect to ns-3 ===
    print("Connecting to ns-3 on port", PORT, "...")
    try:
        env = gym.make('ns3-v0', port=PORT, startSim=False)
    except Exception as e:
        print(f"ERROR: {e}")
        print("Make sure the ns-3 simulation is running first!")
        sys.exit(1)
    print("Connected to ns-3 simulation\n")

    obs = env.reset()
    obs = np.array(obs, dtype=np.float32)

    # Auto-adjust dimensions
    if len(obs) != STATE_DIM:
        print(f"Warning: Adjusting state_dim: {STATE_DIM} -> {len(obs)}")
        STATE_DIM = len(obs)
        policy_net = DRNN(STATE_DIM, HIDDEN_DIM, ACTION_DIM, NUM_LAYERS)
        target_net = DRNN(STATE_DIM, HIDDEN_DIM, ACTION_DIM, NUM_LAYERS)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        optimizer = optim.Adam(policy_net.parameters(), lr=LR)
        normalizer = RunningNormalizer(STATE_DIM)

    # === State tracking ===
    state_history = deque(maxlen=SEQ_LENGTH)
    state_history.append(normalizer.normalize(obs))
    normalizer.update(obs)

    epsilon = EPSILON_START
    hidden = None
    prev_obs = None
    done = False

    step = 0
    total_reward = 0.0
    total_loss = 0.0
    loss_count = 0
    action_counts = [0] * ACTION_DIM

    # Tracking for logging
    reward_window = deque(maxlen=50)
    tput_window = deque(maxlen=50)

    print("=" * 80)
    print("  TRAINING STARTED -- Agent learning to control TCP cwnd in real-time")
    print("=" * 80)
    print()

    while not done:
        step += 1

        # --- 1. Build LSTM input sequence ---
        while len(state_history) < SEQ_LENGTH:
            state_history.appendleft(np.zeros(STATE_DIM, dtype=np.float32))

        state_seq = np.array(list(state_history), dtype=np.float32)
        state_tensor = torch.FloatTensor(state_seq).unsqueeze(0)

        # --- 2. Action selection ---
        # Early phase: moderate growth to ramp up cwnd safely
        if step <= 10:
            action_idx = 4  # MODERATE_INC: safe ramp-up, no explosion
            exploration = True
        elif random.random() < epsilon:
            # Balanced exploration — let agent learn both directions
            weights = [0.10, 0.10, 0.15, 0.25, 0.15, 0.15, 0.10]
            action_idx = random.choices(range(ACTION_DIM), weights=weights, k=1)[0]
            exploration = True
            hidden = None
        else:
            # Exploit: use LSTM with persistent memory
            with torch.no_grad():
                q_values, hidden = policy_net(state_tensor, hidden)
                action_idx = torch.argmax(q_values).item()
            exploration = False

        action_counts[action_idx] += 1

        # --- 3. Execute action in ns-3 ---
        next_obs, ns3_reward, done, info = env.step(action_idx)
        next_obs = np.array(next_obs, dtype=np.float32)

        # --- 4. Compute our own reward (richer than C++ version) ---
        reward = compute_reward(next_obs, prev_obs, action_idx)

        total_reward += reward
        reward_window.append(reward)

        # Estimate throughput for logging
        active_tput = 0
        for i in range(6):
            c, r = next_obs[i*3], next_obs[i*3+1]
            if c > 0 and r > 0:
                active_tput += c / (r / 1000.0) * 8 / 1e6  # Mbps
        tput_window.append(active_tput)

        # --- 5. Normalize and store ---
        normalizer.update(next_obs)
        norm_next = normalizer.normalize(next_obs)
        norm_obs = normalizer.normalize(obs)

        state_history.append(norm_next)
        replay_buffer.push(norm_obs, action_idx, reward, norm_next, done)

        # --- 6. Train ---
        if len(replay_buffer) >= BATCH_SIZE:
            loss = train_step(policy_net, target_net, replay_buffer,
                              optimizer, BATCH_SIZE, GAMMA)
            if loss > 0:
                total_loss += loss
                loss_count += 1

        # --- 7. Target network update ---
        if step % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # --- 8. Decay epsilon ---
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # --- 9. Update LR scheduler ---
        scheduler.step()

        # --- 10. Logging ---
        if step % 5 == 0:
            avg_reward = np.mean(reward_window) if reward_window else 0
            avg_loss = total_loss / loss_count if loss_count > 0 else 0
            avg_tput = np.mean(tput_window) if tput_window else 0
            mode = "EXPLORE" if exploration else "EXPLOIT"

            print(f"Step {step:4d} | {mode:7s} | Act: {ACTION_NAMES[action_idx]:10s} | "
                  f"R: {reward:+6.3f} | AvgR: {avg_reward:+6.3f} | "
                  f"~Tput: {avg_tput:5.1f}Mbps | eps: {epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f} | Buf: {len(replay_buffer)}")

            # Show LSTM temporal insight
            if not exploration and len(state_history) >= 3:
                recent_rtts = []
                for k in range(-3, 0):
                    rtt_vals = [state_history[k][j*3+1] for j in range(6)]
                    recent_rtts.append(np.mean(rtt_vals))
                trend = "UP" if recent_rtts[-1] > recent_rtts[0] else "DOWN"
                print(f"         LSTM sees RTT trend: {trend} "
                      f"({recent_rtts[0]:.2f} -> {recent_rtts[-1]:.2f})")

        prev_obs = obs
        obs = next_obs

    # === DONE ===
    print(f"\n{'':=^80}")
    print(f"  TRAINING COMPLETE")
    print(f"{'':=^80}")
    print(f"  Total Steps:      {step}")
    print(f"  Total Reward:     {total_reward:.2f}")
    print(f"  Avg Reward/Step:  {total_reward/max(step,1):.4f}")
    print(f"  Final Epsilon:    {epsilon:.4f}")
    print(f"  Avg Loss:         {total_loss/max(loss_count,1):.4f}")
    print(f"  Replay Buffer:    {len(replay_buffer)} sequences")
    print(f"  Action Distribution:")
    for i, name in enumerate(ACTION_NAMES):
        pct = 100 * action_counts[i] / max(sum(action_counts), 1)
        bar = "#" * int(pct / 2)
        print(f"    {name:12s}: {action_counts[i]:4d} ({pct:5.1f}%) {bar}")
    print(f"{'':=^80}")

    # Save model
    torch.save({
        'model_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'normalizer_mean': normalizer.mean,
        'normalizer_var': normalizer.var,
        'normalizer_count': normalizer.count,
        'step': step,
        'total_reward': total_reward,
    }, 'drnn_tcp_model.pth')
    print("\nModel saved: drnn_tcp_model.pth")

    env.close()


if __name__ == "__main__":
    main()
