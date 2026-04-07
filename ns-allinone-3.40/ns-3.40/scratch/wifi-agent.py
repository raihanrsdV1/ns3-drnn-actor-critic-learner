#!/usr/bin/env python3
"""
wifi-agent.py  —  LSTM-DQN agent for WiFi parking-lot topology
===============================================================
Adapts the DRNN architecture from agent.py to the 3-flow WiFi sim:

  Observation (9 floats):
    Per flow (3 flows): [cwnd_bytes, rtt_ms, bytes_in_flight]

  Actions (7 discrete AIMD multipliers):
    0=FAST_DEC (-4×)   1=SLOW_DEC (-1×)   2=MAINTAIN (0×)
    3=AIMD (+1×)       4=MOD_INC (+2×)    5=FAST_INC (+4×)
    6=VERY_FAST (+8×)

  Reward (computed in C++ via OpenGym):
    util(4Mbps) - 0.8×RTT_inflation - 0.3×fairness_bonus

Training logs written to:
  w_drnn_train_log.csv  — Episode, Steps, AvgReward, AvgLoss,
                          AvgTput_Mbps, AvgRTT_ms, TotalDrops, Epsilon
  (one row per episode, used by wifi_full_plot.py for training curves)

Usage:
  # Single training episode (called per episode by shell script):
  python3 scratch/wifi-agent.py [--port=5557] [--fresh] [--eval]

  # Eval run (epsilon=0, frozen weights):
  python3 scratch/wifi-agent.py --eval [--port=5557]
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
import argparse
import csv
from collections import deque
import random

# ─────────────────────────────────────────────────────────────────
# 1.  SEQUENCE REPLAY BUFFER
# ─────────────────────────────────────────────────────────────────
class SequenceReplayBuffer:
    """Stores fixed-length windows of consecutive transitions for LSTM."""
    def __init__(self, capacity=3000, seq_length=8):
        self.buffer     = deque(maxlen=capacity)
        self.seq_length = seq_length
        self._episode   = []

    def push(self, state, action, reward, next_state, done):
        self._episode.append((state, action, reward, next_state, done))
        if len(self._episode) >= self.seq_length:
            self.buffer.append(list(self._episode[-self.seq_length:]))
        if done:
            self._episode = []

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────
# 2.  DRNN MODEL  (LSTM + Dueling DQN — identical to agent.py)
# ─────────────────────────────────────────────────────────────────
class DRNN(nn.Module):
    """
    Input:  (batch, seq_len, input_dim)
    Output: (batch, output_dim)  Q-values
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.1 if num_layers > 1 else 0.0)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x, hidden=None):
        B = x.size(0)
        x = self.input_norm(x)
        if hidden is None:
            h0 = torch.zeros(self.num_layers, B, self.hidden_dim)
            c0 = torch.zeros(self.num_layers, B, self.hidden_dim)
            hidden = (h0, c0)
        out, hidden = self.lstm(x, hidden)
        last = out[:, -1, :]
        V = self.value_stream(last)
        A = self.advantage_stream(last)
        Q = V + A - A.mean(dim=1, keepdim=True)
        return Q, hidden


# ─────────────────────────────────────────────────────────────────
# 3.  RUNNING NORMALIZER
# ─────────────────────────────────────────────────────────────────
class RunningNormalizer:
    def __init__(self, dim, clip=5.0):
        self.mean  = np.zeros(dim, dtype=np.float64)
        self.var   = np.ones (dim, dtype=np.float64)
        self.count = 1e-4
        self.clip  = clip

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        delta = x - self.mean
        self.count += 1
        self.mean  += delta / self.count
        self.var   += delta * (x - self.mean)

    def normalize(self, x):
        x   = np.asarray(x, dtype=np.float32)
        std = np.sqrt(self.var / max(self.count, 2) + 1e-8).astype(np.float32)
        return np.clip((x - self.mean.astype(np.float32)) / std,
                       -self.clip, self.clip)


# ─────────────────────────────────────────────────────────────────
# 4.  TRAINING STEP  (Double DQN + Huber loss)
# ─────────────────────────────────────────────────────────────────
def train_step(policy_net, target_net, replay_buffer,
               optimizer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return 0.0
    sequences   = replay_buffer.sample(batch_size)
    batch_losses = []
    for seq in sequences:
        states      = torch.FloatTensor(np.array([s[0] for s in seq])).unsqueeze(0)
        next_states = torch.FloatTensor(np.array([s[3] for s in seq])).unsqueeze(0)
        last_action = torch.LongTensor([seq[-1][1]])
        last_reward = seq[-1][2]
        last_done   = float(seq[-1][4])

        q_vals, _ = policy_net(states)
        q_val     = q_vals[0, last_action]

        with torch.no_grad():
            next_q_p, _ = policy_net(next_states)
            best_a      = next_q_p.argmax(dim=1)
            next_q_t, _ = target_net(next_states)
            max_nq      = next_q_t[0, best_a]
            tgt         = last_reward + gamma * max_nq * (1 - last_done)

        loss = nn.SmoothL1Loss()(q_val, tgt)
        batch_losses.append(loss)

    total_loss = torch.stack(batch_losses).mean()
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    return total_loss.item()


# ─────────────────────────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",   type=int,  default=5557)
    parser.add_argument("--model",  type=str,  default="wifi_drnn_model.pth")
    parser.add_argument("--fresh",  action="store_true",
                        help="Discard saved model, start from scratch")
    parser.add_argument("--eval",   action="store_true",
                        help="Evaluation: load model, epsilon=0, no training")
    parser.add_argument("--trainlog", type=str, default="w_drnn_train_log.csv",
                        help="CSV to append per-episode training metrics")
    args = parser.parse_args()

    # ── Hyper-parameters ──────────────────────────────────────────
    STATE_DIM      = 9        # 3 flows × [cwnd, rtt, inflight]
    ACTION_DIM     = 7
    HIDDEN_DIM     = 128
    NUM_LAYERS     = 2
    SEQ_LENGTH     = 8
    LR             = 3e-4
    GAMMA          = 0.97
    EPSILON_START  = 0.5
    EPSILON_END    = 0.02
    EPSILON_DECAY  = 0.985    # per step (not per episode)
    BATCH_SIZE     = 16
    BUFFER_CAP     = 3000
    TARGET_UPDATE  = 15       # steps between target-net syncs

    ACTION_NAMES = ["FAST_DEC", "SLOW_DEC", "MAINTAIN",
                    "AIMD", "MOD_INC", "FAST_INC", "VERY_FAST"]

    print("=" * 70)
    print("  WiFi DRNN-TCP Agent  (LSTM DQN, 3-flow parking-lot topology)")
    print("=" * 70)
    print(f"  Mode:   {'EVALUATION' if args.eval else 'TRAINING'}")
    print(f"  Port:   {args.port}")
    print(f"  Model:  {args.model}")
    print("=" * 70)

    # ── Networks ─────────────────────────────────────────────────
    policy_net = DRNN(STATE_DIM, HIDDEN_DIM, ACTION_DIM, NUM_LAYERS)
    target_net = DRNN(STATE_DIM, HIDDEN_DIM, ACTION_DIM, NUM_LAYERS)
    optimizer  = optim.Adam(policy_net.parameters(), lr=LR)
    scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)
    buffer     = SequenceReplayBuffer(BUFFER_CAP, SEQ_LENGTH)
    normalizer = RunningNormalizer(STATE_DIM)

    episode       = 0
    epsilon       = EPSILON_START
    total_steps   = 0

    # ── Load checkpoint ──────────────────────────────────────────
    if args.fresh and os.path.exists(args.model):
        print(f"  --fresh: ignoring saved model '{args.model}'")
        target_net.load_state_dict(policy_net.state_dict())
    elif os.path.exists(args.model):
        try:
            ckpt = torch.load(args.model, map_location="cpu")
            policy_net.load_state_dict(ckpt["model_state_dict"])
            target_net.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            normalizer.mean  = ckpt.get("norm_mean",  normalizer.mean)
            normalizer.var   = ckpt.get("norm_var",   normalizer.var)
            normalizer.count = ckpt.get("norm_count", normalizer.count)
            episode          = ckpt.get("episode",    0)
            epsilon          = ckpt.get("epsilon",    EPSILON_START)
            total_steps      = ckpt.get("total_steps", 0)
            print(f"  Loaded checkpoint: episode={episode}, epsilon={epsilon:.4f}")
        except Exception as e:
            print(f"  WARNING: Failed to load model ({e}). Starting fresh.")
            target_net.load_state_dict(policy_net.state_dict())
    else:
        target_net.load_state_dict(policy_net.state_dict())
        print("  No saved model — starting fresh.")

    target_net.eval()

    if args.eval:
        epsilon = 0.0
        policy_net.eval()
        print("  EVAL MODE: epsilon=0, no weight updates")

    # ── Connect to ns-3 ──────────────────────────────────────────
    print(f"\n  Connecting to ns-3 on port {args.port}...")
    try:
        env = gym.make("ns3-v0", port=args.port, startSim=False)
    except Exception as e:
        print(f"  ERROR connecting to ns-3: {e}")
        sys.exit(1)
    print("  Connected.\n")

    obs  = env.reset()
    obs  = np.array(obs, dtype=np.float32)

    # Adapt dims if ns-3 returns different length
    actual_dim = len(obs)
    if actual_dim != STATE_DIM:
        print(f"  Adapting state dim: {STATE_DIM} → {actual_dim}")
        STATE_DIM  = actual_dim
        policy_net = DRNN(STATE_DIM, HIDDEN_DIM, ACTION_DIM, NUM_LAYERS)
        target_net = DRNN(STATE_DIM, HIDDEN_DIM, ACTION_DIM, NUM_LAYERS)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        optimizer  = optim.Adam(policy_net.parameters(), lr=LR)
        normalizer = RunningNormalizer(STATE_DIM)

    # ── State history queue for LSTM ──────────────────────────────
    state_history = deque(maxlen=SEQ_LENGTH)
    state_history.append(normalizer.normalize(obs))
    normalizer.update(obs)

    hidden       = None
    prev_obs     = None
    done         = False
    step         = 0
    total_reward = 0.0
    total_loss   = 0.0
    loss_count   = 0
    action_counts = [0] * ACTION_DIM

    # Per-episode accumulators for training log
    ep_rewards   = []
    ep_tputs     = []    # estimated Mbps per step
    ep_rtts      = []    # avg RTT across flows per step
    ep_losses    = []

    print(f"  Episode {episode + 1}  |  epsilon = {epsilon:.4f}")
    print(f"  {'Step':>5}  {'Mode':>7}  {'Action':<10}  "
          f"{'Reward':>7}  {'~Tput':>7}  {'AvgRTT':>7}  {'eps':>6}")
    print("  " + "─" * 60)

    while not done:
        step       += 1
        total_steps += 1

        # ── Pad history if needed ──
        while len(state_history) < SEQ_LENGTH:
            state_history.appendleft(np.zeros(STATE_DIM, dtype=np.float32))

        state_seq    = np.array(list(state_history), dtype=np.float32)
        state_tensor = torch.FloatTensor(state_seq).unsqueeze(0)

        # ── Action selection ──
        if step <= 10:
            action_idx  = 4      # MOD_INC: safe ramp-up
            exploration = True
        elif random.random() < epsilon:
            weights     = [0.10, 0.10, 0.15, 0.25, 0.15, 0.15, 0.10]
            action_idx  = random.choices(range(ACTION_DIM), weights=weights)[0]
            exploration = True
            hidden      = None
        else:
            with torch.no_grad():
                q_vals, hidden = policy_net(state_tensor, hidden)
                action_idx     = torch.argmax(q_vals).item()
            exploration = False

        action_counts[action_idx] += 1

        # ── Step ns-3 ──
        next_obs, reward, done, info = env.step(action_idx)
        next_obs = np.array(next_obs, dtype=np.float32)

        # Use ns-3 reward (computed in C++ GetReward) directly
        total_reward += reward
        ep_rewards.append(reward)

        # Estimate aggregate throughput & avg RTT from obs for logging
        n_flows = STATE_DIM // 3
        tput = 0.0; avg_rtt = 0.0; active = 0
        for i in range(n_flows):
            c = next_obs[i*3];  r = next_obs[i*3+1]
            if c > 0 and r > 0:
                tput    += (c / (r / 1000.0)) * 8.0 / 1e6
                avg_rtt += r
                active  += 1
        if active:
            avg_rtt /= active
        ep_tputs.append(tput)
        ep_rtts.append(avg_rtt)

        # ── Normalize + store in buffer ──
        normalizer.update(next_obs)
        norm_next = normalizer.normalize(next_obs)
        norm_obs  = normalizer.normalize(obs)
        state_history.append(norm_next)

        if not args.eval:
            buffer.push(norm_obs, action_idx, float(reward), norm_next, done)

            # ── Training step ──
            if len(buffer) >= BATCH_SIZE:
                loss = train_step(policy_net, target_net, buffer,
                                  optimizer, BATCH_SIZE, GAMMA)
                if loss > 0.0:
                    total_loss += loss
                    loss_count += 1
                    ep_losses.append(loss)

            # ── Target-net sync ──
            if step % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # ── Epsilon decay ──
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

            # ── LR schedule ──
            scheduler.step()

        # ── Logging ──
        if step % 10 == 0:
            avg_r = np.mean(ep_rewards[-20:]) if ep_rewards else 0.0
            avg_l = np.mean(ep_losses[-20:])  if ep_losses  else 0.0
            mode  = "EXPLORE" if exploration else "EXPLOIT"
            print(f"  {step:5d}  {mode:>7}  {ACTION_NAMES[action_idx]:<10}  "
                  f"{reward:+7.3f}  {tput:7.2f}  {avg_rtt:7.1f}  {epsilon:.4f}"
                  f"  loss={avg_l:.4f}")

        prev_obs = obs
        obs      = next_obs

    # ── Episode summary ──────────────────────────────────────────
    episode    += 1
    avg_r_ep    = float(np.mean(ep_rewards))  if ep_rewards else 0.0
    avg_l_ep    = float(np.mean(ep_losses))   if ep_losses  else 0.0
    avg_t_ep    = float(np.mean(ep_tputs))    if ep_tputs   else 0.0
    avg_rtt_ep  = float(np.mean([r for r in ep_rtts if r > 0])) if ep_rtts else 0.0

    print(f"\n{'='*70}")
    print(f"  Episode {episode} complete")
    print(f"  Steps={step}  TotalReward={total_reward:.2f}  "
          f"AvgReward={avg_r_ep:.4f}  epsilon={epsilon:.4f}")
    print(f"  AvgTput≈{avg_t_ep:.2f} Mbps  AvgRTT≈{avg_rtt_ep:.1f} ms  "
          f"AvgLoss={avg_l_ep:.4f}")
    print(f"  Action distribution:")
    for i, nm in enumerate(ACTION_NAMES):
        pct = 100 * action_counts[i] / max(sum(action_counts), 1)
        print(f"    {nm:<12}: {action_counts[i]:4d} ({pct:5.1f}%) "
              f"{'#' * int(pct / 2)}")
    print(f"{'='*70}")

    # ── Write to training log CSV ────────────────────────────────
    write_header = not os.path.exists(args.trainlog) or \
                   os.path.getsize(args.trainlog) == 0
    with open(args.trainlog, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["Episode", "Steps", "AvgReward", "AvgLoss",
                        "AvgTput_Mbps", "AvgRTT_ms", "Epsilon"])
        w.writerow([episode, step,
                    f"{avg_r_ep:.5f}", f"{avg_l_ep:.5f}",
                    f"{avg_t_ep:.4f}", f"{avg_rtt_ep:.2f}",
                    f"{epsilon:.5f}"])

    # ── Save model ───────────────────────────────────────────────
    if not args.eval:
        torch.save({
            "model_state_dict":    policy_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "norm_mean":  normalizer.mean,
            "norm_var":   normalizer.var,
            "norm_count": normalizer.count,
            "episode":    episode,
            "epsilon":    epsilon,
            "total_steps": total_steps,
        }, args.model)
        print(f"  Model saved → {args.model}")
        print(f"  Next epsilon: {epsilon:.4f}")

    env.close()
    print("  Done.\n")


if __name__ == "__main__":
    main()
