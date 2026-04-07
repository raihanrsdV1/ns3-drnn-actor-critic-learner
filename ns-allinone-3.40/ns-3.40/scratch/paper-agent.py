#!/usr/bin/env python3
"""
Paper Replication Agent — exact match to arXiv:2508.01047v3 Section 3.3
========================================================================
Architecture (from paper):
  - Input layer: 5 neurons  [BytesInFlight, cWnd, RTT, SegmentsAcked, ssThresh]
  - Hidden layer: 64 neurons, ReLU activation
  - Output layer: 4 neurons, linear activation  (Q-values for 4 actions)
  - Optimizer: Adam
  - Loss: mean squared error (standard DQN Bellman loss)

Actions (absolute byte deltas added to cWnd):
  0: Maintain        (delta =    0)
  1: Standard Inc    (delta = +1500)
  2: Conservative Dec(delta =  -150)
  3: Rocket Inc      (delta = +4000)

Reward:  throughput_Mbps  -  0.5 * RTT_seconds   (beta=0.5 per paper)

Training: 100 episodes, epsilon 1.0->0.05 decaying per-episode.
Evaluation: load saved model, epsilon=0 (pure exploitation).

Usage:
  # Training episode (repeat 100 times):
  python3 scratch/paper-agent.py

  # Final evaluation:
  python3 scratch/paper-agent.py --eval
"""

import gym
import ns3gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import sys
import argparse
from collections import deque


# ============================================================
# 1. DQN NETWORK  (paper: 1 hidden layer, 64 neurons, ReLU)
# ============================================================
class DQN(nn.Module):
    def __init__(self, state_dim=5, action_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)   # linear output (paper: linear activation)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 2. REPLAY BUFFER
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buffer.append((s, a, r, ns, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        s, a, r, ns, d = zip(*batch)
        return (np.array(s,  dtype=np.float32),
                np.array(a,  dtype=np.int64),
                np.array(r,  dtype=np.float32),
                np.array(ns, dtype=np.float32),
                np.array(d,  dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ============================================================
# 3. STATE NORMALIZATION
# For 2 Mbps bottleneck:
#   BDP = 2e6/8 * 0.042 = ~10,500 bytes
#   Max queue delay at 2 Mbps, 100 packets = 100*1500*8/2e6 = 600 ms
#   So max RTT ≈ 42 + 600 = ~642 ms
# ============================================================
def normalize_state(obs):
    s = np.zeros(5, dtype=np.float32)
    if len(obs) >= 5:
        s[0] = np.clip(obs[0] / 20000.0,  0, 5)   # bytesInFlight / ~2xBDP
        s[1] = np.clip(obs[1] / 20000.0,  0, 5)   # cwnd            / ~2xBDP
        s[2] = np.clip(obs[2] / 600.0,    0, 5)   # rtt_ms  (max ~642ms)
        s[3] = np.clip(obs[3] / 20.0,     0, 5)   # segmentsAcked per step
        s[4] = np.clip(obs[4] / 65535.0,  0, 5)   # ssThresh / typical max
    elif len(obs) >= 4:
        # fallback if ssThresh not yet in obs
        s[0] = np.clip(obs[0] / 20000.0, 0, 5)
        s[1] = np.clip(obs[1] / 20000.0, 0, 5)
        s[2] = np.clip(obs[2] / 600.0,   0, 5)
        s[3] = np.clip(obs[3] / 20.0,    0, 5)
    return s


# ============================================================
# 4. MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",  action="store_true",
                        help="Evaluation: load saved weights, epsilon=0")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore saved model and start training from scratch")
    parser.add_argument("--port",  type=int, default=5556)
    parser.add_argument("--model", type=str, default="paper_drl_model.pth")
    args = parser.parse_args()

    # Paper hyperparameters
    STATE_DIM     = 5
    ACTION_DIM    = 4
    HIDDEN        = 64
    LR            = 0.001
    GAMMA         = 0.95        # discount factor
    BATCH_SIZE    = 32
    BUFFER_CAP    = 10000
    TARGET_UPDATE = 20          # steps between target-net syncs
    EPS_DECAY_EP  = 0.941       # per-episode decay: 0.941^50 ≈ 0.05

    ACTION_NAMES = ["MAINTAIN", "STD_INC(+1500)", "CONS_DEC(-150)", "ROCKET(+4000)"]

    print("=" * 65)
    print("  Paper Replication: DQN Agent  (arXiv:2508.01047v3)")
    print("=" * 65)
    print(f"  State:   {STATE_DIM}  [BytesInFlight, cWnd, RTT, SegsAcked, ssThresh]")
    print(f"  Actions: {ACTION_DIM}  {ACTION_NAMES}")
    print(f"  Reward:  throughput_Mbps - 0.5 * RTT_sec")
    print(f"  Hidden:  {HIDDEN} neurons (1 layer)")
    print(f"  Mode:    {'EVALUATION (epsilon=0)' if args.eval else 'TRAINING'}")
    print("=" * 65)

    policy_net = DQN(STATE_DIM, ACTION_DIM, HIDDEN)
    target_net = DQN(STATE_DIM, ACTION_DIM, HIDDEN)
    optimizer  = optim.Adam(policy_net.parameters(), lr=LR)
    buffer     = ReplayBuffer(BUFFER_CAP)

    episode           = 0
    epsilon           = 1.0
    total_steps       = 0

    if args.fresh and os.path.exists(args.model):
        print(f"\n  --fresh: ignoring saved model '{args.model}', starting from scratch.")
        target_net.load_state_dict(policy_net.state_dict())
        print(f"  Starting fresh (epsilon=1.0)")
    elif os.path.exists(args.model):
        try:
            ckpt = torch.load(args.model, weights_only=False)
            policy_net.load_state_dict(ckpt['model'])
            target_net.load_state_dict(ckpt['model'])
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            episode     = ckpt.get('episode', 0)
            epsilon     = ckpt.get('epsilon', 1.0)
            total_steps = ckpt.get('total_steps', 0)
            print(f"\n  Loaded: {args.model}")
            print(f"  Episode {episode}, epsilon {epsilon:.4f}, steps {total_steps}")
        except Exception as e:
            print(f"\n  WARNING: Could not load {args.model}: {e}")
            print(f"  Deleting incompatible checkpoint and starting fresh.")
            os.remove(args.model)
            policy_net = DQN(STATE_DIM, ACTION_DIM, HIDDEN)
            target_net = DQN(STATE_DIM, ACTION_DIM, HIDDEN)
            target_net.load_state_dict(policy_net.state_dict())
            optimizer  = optim.Adam(policy_net.parameters(), lr=LR)
            episode, epsilon, total_steps = 0, 1.0, 0
    else:
        target_net.load_state_dict(policy_net.state_dict())
        print(f"\n  No saved model — starting fresh (epsilon=1.0)")

    target_net.eval()

    if args.eval:
        epsilon = 0.0
        print("\n  >>> EVALUATION MODE: epsilon=0 <<<")

    # Connect to ns-3
    print(f"\n  Connecting to ns-3 on port {args.port}...")
    try:
        env = gym.make('ns3-v0', port=args.port, startSim=False)
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    obs  = env.reset()
    obs  = np.array(obs, dtype=np.float32)
    print(f"  Connected. Obs shape: {obs.shape}\n")

    # Adjust STATE_DIM if ns-3 returns different number
    actual_dim = len(obs)
    if actual_dim != STATE_DIM:
        print(f"  Warning: expected {STATE_DIM} dims, got {actual_dim}. Rebuilding net.")
        STATE_DIM  = actual_dim
        policy_net = DQN(STATE_DIM, ACTION_DIM, HIDDEN)
        target_net = DQN(STATE_DIM, ACTION_DIM, HIDDEN)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        optimizer  = optim.Adam(policy_net.parameters(), lr=LR)

    step          = 0
    total_reward  = 0.0
    action_counts = [0] * ACTION_DIM
    done          = False
    reward_hist   = deque(maxlen=50)
    loss_hist     = deque(maxlen=50)

    print(f"  Episode {episode + 1}  |  epsilon = {epsilon:.4f}")
    print(f"  {'Step':>5}  {'Action':<16}  {'Reward':>8}  "
          f"{'AvgR':>8}  {'cWnd':>8}  {'RTT(ms)':>8}  {'eps':>6}")
    print("  " + "-" * 72)

    while not done:
        step        += 1
        total_steps += 1
        norm_obs     = normalize_state(obs)

        # Epsilon-greedy
        if random.random() < epsilon:
            action = random.randrange(ACTION_DIM)
        else:
            with torch.no_grad():
                q = policy_net(torch.FloatTensor(norm_obs).unsqueeze(0))
                action = q.argmax(dim=1).item()

        action_counts[action] += 1

        next_obs, reward, done, info = env.step(action)
        next_obs     = np.array(next_obs, dtype=np.float32)
        total_reward += reward
        reward_hist.append(reward)

        norm_next = normalize_state(next_obs)
        buffer.push(norm_obs, action, reward, norm_next, done)

        loss_val = 0.0
        if not args.eval and len(buffer) >= BATCH_SIZE:
            s, a, r, ns, d = buffer.sample(BATCH_SIZE)
            q_all   = policy_net(torch.FloatTensor(s))
            q_taken = q_all.gather(1, torch.LongTensor(a).unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_next  = target_net(torch.FloatTensor(ns)).max(dim=1)[0]
                targets = torch.FloatTensor(r) + GAMMA * q_next * (1 - torch.FloatTensor(d))
            loss = nn.MSELoss()(q_taken, targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()
            loss_val = loss.item()
            loss_hist.append(loss_val)

        if step % TARGET_UPDATE == 0 and not args.eval:
            target_net.load_state_dict(policy_net.state_dict())

        if step % 10 == 0:
            avg_r  = np.mean(reward_hist) if reward_hist else 0
            cwnd   = obs[1] if len(obs) > 1 else 0
            rtt    = obs[2] if len(obs) > 2 else 0
            print(f"  {step:>5}  {ACTION_NAMES[action]:<16}  "
                  f"{reward:>+8.3f}  {avg_r:>+8.3f}  "
                  f"{cwnd:>8.0f}  {rtt:>8.1f}  {epsilon:>6.3f}")

        obs = next_obs

    # Episode done
    episode += 1
    print(f"\n{'='*65}")
    print(f"  Episode {episode} complete")
    print(f"  Steps: {step} | Total reward: {total_reward:.2f} "
          f"| Avg: {total_reward/max(step,1):.4f}")
    print(f"  Action distribution:")
    for i, name in enumerate(ACTION_NAMES):
        pct = 100 * action_counts[i] / max(sum(action_counts), 1)
        bar = "#" * int(pct / 2)
        print(f"    {name:<18}: {action_counts[i]:4d} ({pct:5.1f}%)  {bar}")

    if not args.eval:
        epsilon = max(0.05, epsilon * EPS_DECAY_EP)
        print(f"\n  Next epsilon: {epsilon:.4f}  "
              f"({'exploiting' if epsilon < 0.1 else 'exploring'})")
        torch.save({
            'model':       policy_net.state_dict(),
            'optimizer':   optimizer.state_dict(),
            'episode':     episode,
            'epsilon':     epsilon,
            'total_steps': total_steps,
        }, args.model)
        print(f"  Saved: {args.model}")
    else:
        print(f"\n  Eval mode — model not modified.")

    env.close()
    print("  Done.\n")


if __name__ == "__main__":
    main()
