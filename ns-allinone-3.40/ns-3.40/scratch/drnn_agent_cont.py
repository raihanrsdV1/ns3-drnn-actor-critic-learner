#!/usr/bin/env python3
"""
drnn_agent_cont.py  —  Continuous-Action LSTM Actor-Critic (SAC) for ns-3
==========================================================================
Experimental companion to drnn_agent.py.

Instead of picking 1-of-7 AIMD multipliers the agent outputs ONE float:
    target_cwnd  ∈  [cwnd_min, cwnd_max]  (bytes)

The simulation (wifi-sim-cont.cc) receives that float and sets
tcb->m_cWnd directly — no AIMD arithmetic at all.

Why this should be better than the discrete version
────────────────────────────────────────────────────
  • Fine-grained control: the agent can nudge cwnd by e.g. 200 bytes
    when it senses RTT just beginning to rise — impossible with x8 steps.
  • No discretisation dead-zones: the optimal cwnd is ~12 KB per flow;
    the 7 discrete actions skip over most of that range coarsely.
  • The LSTM sees SEQ_LENGTH=8 previous steps of (cwnd,rtt,bif) per flow
    so it can watch RTT trend over 800 ms and pre-emptively reduce cwnd
    before the queue fills — the 2-3 steps before drastic action you want.

Architecture: SAC (Soft Actor-Critic) with LSTM backbones
──────────────────────────────────────────────────────────
  Actor   : LSTM(hidden=64,layers=2) → fc_shared → fc_mean/fc_logstd → Gaussian
  Critic1 : LSTM(hidden=64,layers=2) → fc(last_hidden + action) → Q
  Critic2 : same  (twin critic — reduces Q overestimation bias)
  Target networks for critics — soft-updated each step (τ=0.005)

  SAC (Soft Actor-Critic):
    1. Twin critics  — min(Q1,Q2) as bootstrap target (same as TD3)
    2. Stochastic actor — Gaussian policy with entropy regularisation
    3. Auto-tuned temperature α — balances exploration vs exploitation
    4. No noise schedule needed — entropy provides natural exploration

Feature attribution (printed every episode)
────────────────────────────────────────────
  Compute ∂actor_output/∂input (input-gradient saliency) over sampled
  states from the episode.  Prints a ranked table of which features
  [flow0_cwnd, flow0_rtt, flow0_bif, ...] drive cwnd decisions most.
  This directly answers "which factor is influencing cwnd the most".

Reward  —  throughput-weighted geometric mean  ∈ [0, 1]
────────────────────────────────────────────────────────────────
  R = (tput_ratio² × rtt_quality × loss_quality)^(1/4)

  tput_ratio   = actual_throughput / bottleneck_capacity          [0, 1]
  rtt_quality  = min_rtt / actual_rtt                             (0, 1]
  loss_quality = exp(−0.2 × total_drops_this_step)                (0, 1]

  Uses ACTUAL packet drops from the C++ observation vector
  (obs[9]=dropsL1, obs[10]=dropsL2) — not a BIF/BDP proxy.
  This gives the agent a direct, immediate signal whenever packets
  are lost at either bottleneck queue.

  Why throughput gets double weight (power 2):
    The equal-weight version trapped the agent: low cwnd scored 0.58
    (tput=0.22, rtt=0.90, loss=1.0) while moderate queue scored 0.50
    (tput=0.75, rtt=0.85, loss=0.14).  The agent rationally chose starvation.
    Doubling the tput exponent makes under-utilisation the more expensive
    mistake — low cwnd now scores 0.45 vs 0.66 for moderate-queue/high-tput.

  Why exp(−0.2 × drops):
    At the bottleneck (~4 Mbps, ~34 pkts/step), a few drops ≈ 5-10% loss:
      0 drops → 1.000   3 drops → 0.549   10 drops → 0.135
    The exponential keeps a usable gradient across the learning region
    while collapsing reward multiplicatively under heavy loss.

  avgR → 1.0 = full link, min RTT, zero drops  (all three optimal)
  avgR → 0.0 = either link idle OR severely flooded (loss term collapses)

CLI (superset of drnn_agent.py — same flags work + two new ones)
─────────────────────────────────────────────────────────────────
    --cwnd-min-kb   float  min cwnd in KB  (default 2.8 KB ≈ 2 segments)
    --cwnd-max-kb   float  max cwnd in KB  (default 144 KB)

Usage:
    python3 scratch/drnn_agent_cont.py \\
        --port 5557 --model wifi_cont_model.pth \\
        --bottleneck-mbps 4.0 --min-rtt-ms 72.0 \\
        --trainlog w_cont_train_log.csv

    python3 scratch/drnn_agent_cont.py --fresh \\
        --port 5557 --model wifi_cont_model.pth \\
        --bottleneck-mbps 4.0 --min-rtt-ms 72.0

    python3 scratch/drnn_agent_cont.py --eval \\
        --port 5557 --model wifi_cont_model.pth \\
        --bottleneck-mbps 4.0 --min-rtt-ms 72.0
"""

import os, sys, csv, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

try:
    import gym
    import ns3gym  # noqa: F401
except ImportError as e:
    print(f"  ERROR: Missing dependency — {e}")
    print("  Install with:  pip install ns3gym")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SEQUENCE REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class SequenceReplayBuffer:
    """
    Stores sliding windows of consecutive (s, a, r, s', done) tuples.
    'a' is now a scalar float (target cwnd bytes) instead of an int index.
    """
    def __init__(self, capacity: int = 10000, seq_length: int = 8):
        self.buffer     = deque(maxlen=capacity)
        self.seq_length = seq_length
        self._ep: list  = []

    def push(self, state, action, reward, next_state, done):
        self._ep.append((state, float(action), reward, next_state, done))
        if len(self._ep) >= self.seq_length:
            self.buffer.append(list(self._ep[-self.seq_length:]))
        if done:
            self._ep = []

    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  RUNNING NORMALIZER  (Welford online mean/var)
# ══════════════════════════════════════════════════════════════════════════════

class RunningNormalizer:
    def __init__(self, dim: int, clip: float = 5.0):
        self.mean  = np.zeros(dim, dtype=np.float64)
        self.var   = np.ones(dim,  dtype=np.float64)
        self.count = 1e-4
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
# 3.  REWARD FUNCTION  —  linear penalty  ∈ [-3, +1]
#     Uses ACTUAL PACKET DROPS + RTT excess + under-utilisation penalty
# ══════════════════════════════════════════════════════════════════════════════

def compute_reward(obs: np.ndarray,
                   bottleneck_bps: float,
                   min_rtt_ms:     float) -> float:
    """
    R = 1.0  − underutil_pen  − rtt_pen  − drop_pen     clipped to [−3, +1]

    Three linear penalties, each targeting a different failure mode:

    underutil_pen = 2.0 × max(0, 1 − BIF/BDP)
        Penalises under-filling the pipe.  BIF/BDP is reliable here because
        BIF < BDP ⟺ not enough bytes in the pipe (no queue ambiguity).
        Penalty vanishes once BIF ≥ BDP (pipe is full).

    rtt_pen       = max(0, avgRTT / minRTT − 1)
        Penalises queuing delay above propagation baseline.
        0 at baseline, 0.5 at 1.5×, 1.0 at 2×.  Linear ⇒ strong gradient.

    drop_pen      = 0.1 × total_drops_this_step
        Direct packet-loss signal from the C++ observation (obs[-2:]).
        Each drop costs 0.1 reward points.  10 drops/step = −1.0 penalty.

    Reward landscape (WiFi: 4 Mbps, minRTT=72 ms, BDP=36 KB):
        0.5×BDP  cwnd=6KB/flow   →  1.0 − 1.0 − 0 − 0   =  0.00
        0.75×BDP cwnd=9KB/flow   →  1.0 − 0.5 − 0 − 0   =  0.50
        1.0×BDP  cwnd=12KB/flow  →  1.0 − 0   − 0 − 0   =  1.00  ← MAXIMUM
        1.25×BDP (light queue)   →  1.0 − 0   − 0.3 − 0.1 =  0.60
        1.5×BDP  (moderate)      →  1.0 − 0   − 0.5 − 0.2 =  0.30
        2.0×BDP  (heavy)         →  1.0 − 0   − 1.1 − 0.5 = −0.60
        3.0×BDP  (flooding)      →  1.0 − 0   − 1.8 − 2.0 = −2.80

    Why this beats the geometric mean:
        Geometric mean spread: optimal 1.0 → flooding 0.34 (Δ = 0.66)
        Linear penalty spread: optimal 1.0 → flooding −2.80 (Δ = 3.80)
        ~6× stronger gradient — the agent sees a MUCH clearer signal.

    avgR → +1.0  = full pipe, min RTT, zero drops  (ideal)
    avgR →  0.0  = half-utilised pipe  (under-sending)
    avgR → −3.0  = catastrophic flooding  (clipped floor)
    """
    obs_len = len(obs)
    # Detect format: 11-dim (with drops) or 9-dim (legacy, no drops)
    if obs_len >= 5 and obs_len % 3 == 2:
        n_flows   = (obs_len - 2) // 3
        has_drops = True
    else:
        n_flows   = obs_len // 3
        has_drops = False

    rtts   = [float(obs[i*3+1]) for i in range(n_flows)]
    bifs   = [float(obs[i*3+2]) for i in range(n_flows)]
    active = [i for i in range(n_flows) if bifs[i] > 0 and rtts[i] > 0]
    if not active:
        return 0.0

    avg_rtt   = float(np.mean([rtts[i] for i in active]))
    total_bif = sum(bifs[i] for i in active)
    bdp       = bottleneck_bps * (min_rtt_ms / 1000.0)

    # ── Under-utilisation: penalise when BIF < BDP (pipe not full) ──
    underutil_pen = 2.0 * max(0.0, 1.0 - total_bif / bdp)

    # ── RTT excess: penalise queuing delay above propagation baseline ──
    rtt_pen = max(0.0, avg_rtt / min_rtt_ms - 1.0)

    # ── Drop penalty: DIRECT packet-loss signal ──
    if has_drops:
        total_drops = float(obs[-2]) + float(obs[-1])
        drop_pen = 0.1 * total_drops
    else:
        # Legacy fallback for 9-dim obs: BIF/BDP proxy
        queue_excess = max(0.0, total_bif / bdp - 1.0)
        drop_pen = 0.5 * queue_excess

    R = 1.0 - underutil_pen - rtt_pen - drop_pen
    return float(np.clip(R, -3.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# 4.  LSTM ACTOR  — outputs one continuous cwnd value per step
# ══════════════════════════════════════════════════════════════════════════════

class LSTMActor(nn.Module):
    """
    SAC stochastic actor — outputs mean + log_std for a Gaussian policy.

    forward()  : deterministic tanh(mean) for evaluation
    sample()   : stochastic tanh(mean + std*ε) for training
                 also returns log_prob for the entropy objective
    """
    def __init__(self, state_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_norm = nn.LayerNorm(state_dim)
        self.lstm = nn.LSTM(
            state_dim, hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc_shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mean   = nn.Linear(hidden_dim // 2, 1)
        self.fc_logstd = nn.Linear(hidden_dim // 2, 1)
        self.LOG_STD_MIN = -5.0
        self.LOG_STD_MAX = 2.0

    def _lstm_forward(self, x, hidden=None):
        B = x.size(0)
        x = self.input_norm(x)
        if hidden is None:
            h0 = torch.zeros(self.num_layers, B, self.hidden_dim)
            c0 = torch.zeros(self.num_layers, B, self.hidden_dim)
            hidden = (h0, c0)
        out, hidden = self.lstm(x, hidden)
        h = self.fc_shared(out[:, -1, :])
        return h, hidden

    def forward(self, x, hidden=None):
        """Deterministic: tanh(mean). Used in eval mode."""
        h, hidden = self._lstm_forward(x, hidden)
        mean = self.fc_mean(h)
        return torch.tanh(mean), hidden

    def sample(self, x, hidden=None):
        """Stochastic: sample from N(mean, std), squash via tanh.
        Returns (action ∈ [-1,1], log_prob, hidden)."""
        h, hidden = self._lstm_forward(x, hidden)
        mean    = self.fc_mean(h)
        log_std = self.fc_logstd(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std     = log_std.exp()
        # Reparameterisation trick
        z      = mean + std * torch.randn_like(std)
        action = torch.tanh(z)
        # Log-prob with tanh squashing correction
        log_prob = (
            -0.5 * ((z - mean) / (std + 1e-8)).pow(2)
            - 0.5 * np.log(2 * np.pi)
            - log_std
            - torch.log(1 - action.pow(2) + 1e-6)
        ).sum(dim=-1, keepdim=True)
        return action, log_prob, hidden


# ══════════════════════════════════════════════════════════════════════════════
# 5.  LSTM CRITIC  — Q(state_sequence, action) estimator
# ══════════════════════════════════════════════════════════════════════════════

class LSTMCritic(nn.Module):
    """
    Input  : state sequence (batch, seq_len, state_dim)
             action scalar  (batch, 1) normalised to [−1, +1]
    Output : (batch, 1) scalar Q-value
    """
    def __init__(self, state_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_norm = nn.LayerNorm(state_dim)
        self.lstm = nn.LSTM(
            state_dim, hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, action):
        B  = x.size(0)
        x  = self.input_norm(x)
        h0 = torch.zeros(self.num_layers, B, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, B, self.hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        last   = out[:, -1, :]
        return self.fc(torch.cat([last, action], dim=1))


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SCALE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def tanh_to_cwnd(t: torch.Tensor, cwnd_min: float, cwnd_max: float) -> torch.Tensor:
    """tanh ∈ [−1,1] → cwnd ∈ [cwnd_min, cwnd_max]"""
    return cwnd_min + (t + 1.0) / 2.0 * (cwnd_max - cwnd_min)


def cwnd_to_tanh_tensor(cwnd: torch.Tensor, cwnd_min: float, cwnd_max: float) -> torch.Tensor:
    """cwnd → normalised tanh-space tensor"""
    return 2.0 * (cwnd - cwnd_min) / (cwnd_max - cwnd_min) - 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 7.  SAC TRAINING STEP
# ══════════════════════════════════════════════════════════════════════════════

def train_step_sac(
    actor, critic1, critic2, critic1_target, critic2_target,
    replay_buffer, actor_opt, critic1_opt, critic2_opt,
    log_alpha, alpha_opt, target_entropy,
    batch_size, gamma, tau=0.005,
) -> tuple:
    """
    One SAC update.  Returns (critic_loss, actor_loss, alpha).

    SAC vs TD3 — why this fixes the conservative collapse:
      1. Entropy bonus α·H(π) keeps the policy stochastic — can't collapse
         to a deterministic conservative point
      2. Auto-tuned temperature α balances explore vs exploit
      3. Actor updated every step (no delay) with entropy-regularised objective
      4. Stochastic policy = natural exploration, no noise schedule needed
    """
    if len(replay_buffer) < batch_size:
        return 0.0, 0.0, 0.0

    seqs = replay_buffer.sample(batch_size)

    states_l, nstates_l, acts_l, rews_l, dones_l = [], [], [], [], []
    for seq in seqs:
        states_l.append([t[0] for t in seq])
        nstates_l.append([t[3] for t in seq])
        acts_l.append(seq[-1][1])   # tanh-space action ∈ [-1, 1]
        rews_l.append(seq[-1][2])
        dones_l.append(float(seq[-1][4]))

    states      = torch.FloatTensor(np.array(states_l))    # (B,T,D)
    next_states = torch.FloatTensor(np.array(nstates_l))
    acts_tanh   = torch.FloatTensor(acts_l).unsqueeze(1)   # (B,1)
    rewards     = torch.FloatTensor(rews_l).unsqueeze(1)
    dones       = torch.FloatTensor(dones_l).unsqueeze(1)

    alpha = log_alpha.exp().detach()

    # ── Critic update (entropy-augmented target) ──────────────────────────────
    with torch.no_grad():
        na, nlp, _ = actor.sample(next_states)
        tq1 = critic1_target(next_states, na)
        tq2 = critic2_target(next_states, na)
        target_q = rewards + gamma * (1.0 - dones) * (
            torch.min(tq1, tq2) - alpha * nlp
        )

    q1 = critic1(states, acts_tanh)
    q2 = critic2(states, acts_tanh)
    cl = nn.SmoothL1Loss()(q1, target_q) + nn.SmoothL1Loss()(q2, target_q)

    critic1_opt.zero_grad(); critic2_opt.zero_grad()
    cl.backward()
    torch.nn.utils.clip_grad_norm_(critic1.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(critic2.parameters(), 1.0)
    critic1_opt.step(); critic2_opt.step()

    # ── Actor update (every step — no delay) ─────────────────────────────────
    a_new, lp_new, _ = actor.sample(states)
    q1_new = critic1(states, a_new)
    al = (alpha * lp_new - q1_new).mean()

    actor_opt.zero_grad()
    al.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    actor_opt.step()

    # ── Temperature α update (auto-tune) ─────────────────────────────────────
    alpha_loss = -(log_alpha * (lp_new.detach() + target_entropy)).mean()
    alpha_opt.zero_grad()
    alpha_loss.backward()
    alpha_opt.step()

    # ── Soft update critic targets ────────────────────────────────────────────
    for p, tp in zip(critic1.parameters(), critic1_target.parameters()):
        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
    for p, tp in zip(critic2.parameters(), critic2_target.parameters()):
        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    return cl.item(), al.item(), log_alpha.exp().item()


# ══════════════════════════════════════════════════════════════════════════════
# 8.  FEATURE ATTRIBUTION  (input-gradient saliency)
# ══════════════════════════════════════════════════════════════════════════════

def compute_feature_attribution(actor, state_batch: np.ndarray,
                                 feature_names: list) -> None:
    """
    Compute mean |∂actor_output/∂input| across sampled episode states.
    Prints a ranked table showing which input features drive cwnd decisions.

    This directly answers "which factor is influencing cwnd the most" —
    e.g. if flow1_rtt tops the table, the LSTM is mainly reacting to
    flow 1's RTT when deciding cwnd.
    """
    if state_batch.shape[0] == 0:
        return
    x = torch.FloatTensor(state_batch)   # (N, T, D)
    x.requires_grad_(True)
    actor.eval()
    out, _ = actor(x)
    out.sum().backward()
    actor.train()

    grad       = x.grad.detach().abs()          # (N, T, D)
    importance = grad.mean(dim=(0, 1)).numpy()  # (D,)

    ranked = sorted(enumerate(importance), key=lambda kv: -kv[1])
    print("\n  ── Feature Attribution (|∂cwnd/∂feature| — higher = more influence) ──")
    print(f"  {'Feature':<24}  {'Importance':>12}  Bar")
    print("  " + "─" * 58)
    max_imp = max(v for _, v in ranked) + 1e-9
    for idx, imp in ranked:
        fname = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        bar   = "█" * int(20 * imp / max_imp)
        print(f"  {fname:<24}  {imp:12.5f}  {bar}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Continuous-action LSTM-SAC agent for ns-3 TCP control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port",            type=int,   default=5557)
    parser.add_argument("--model",           type=str,   default="wifi_cont_model.pth")
    parser.add_argument("--trainlog",        type=str,   default="w_cont_train_log.csv")
    parser.add_argument("--bottleneck-mbps", type=float, default=4.0)
    parser.add_argument("--min-rtt-ms",      type=float, default=72.0)
    parser.add_argument("--cwnd-min-kb",     type=float, default=4.0,
                        help="Minimum cwnd in KB  (default 4 KB ≈ ⅓ BDP/flow)")
    parser.add_argument("--cwnd-max-kb",     type=float, default=24.0,
                        help="Maximum cwnd in KB  (default 24 KB ≈ 2× BDP/flow)")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore saved checkpoint — start from scratch")
    parser.add_argument("--eval",  action="store_true",
                        help="Evaluation mode: deterministic policy, no updates")
    args = parser.parse_args()

    bottleneck_bps = args.bottleneck_mbps * 1e6 / 8.0
    CWND_MIN = args.cwnd_min_kb * 1024.0
    CWND_MAX = args.cwnd_max_kb * 1024.0
    requested_cwnd_min = CWND_MIN
    requested_cwnd_max = CWND_MAX

    # ── Hyper-parameters ──────────────────────────────────────────────────────
    HIDDEN_DIM       = 64
    NUM_LAYERS       = 2
    SEQ_LENGTH       = 8
    LR_ACTOR         = 3e-4
    LR_CRITIC        = 3e-4
    LR_ALPHA         = 3e-4    # SAC temperature learning rate
    GAMMA            = 0.97
    TARGET_ENTROPY   = -1.0    # -dim(action) — SAC auto-tunes α to match
    BATCH_SIZE       = 32
    BUFFER_CAP       = 10000
    TRAIN_EVERY      = 4       # train every N env steps

    print("=" * 72)
    print("  DRNN-TCP Continuous — LSTM-SAC Congestion Control Agent")
    print("=" * 72)
    print(f"  Mode:        {'EVALUATION' if args.eval else 'TRAINING'}")
    print(f"  Port:        {args.port}")
    print(f"  Model:       {args.model}")
    print(f"  Bottleneck:  {args.bottleneck_mbps:.1f} Mbps")
    print(f"  Min RTT:     {args.min_rtt_ms:.1f} ms")
    print(f"  Requested cwnd range (CLI): [{requested_cwnd_min/1024:.2f} KB, {requested_cwnd_max/1024:.1f} KB]")
    print(f"  Train log:   {args.trainlog}")
    print("=" * 72)

    print(f"\n  Connecting to ns-3 on port {args.port} ...")
    try:
        env = gym.make("ns3-v0", port=args.port, startSim=False)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        sys.exit(1)
    print("  Connected.\n")

    obs = np.array(env.reset(), dtype=np.float32)

    # Prefer cwnd bounds from ns-3 action space (single source of truth).
    # Fallback to CLI values if action-space metadata is unavailable.
    env_cwnd_min = None
    env_cwnd_max = None
    try:
        act_space = getattr(env, "action_space", None)
        if act_space is not None and hasattr(act_space, "low") and hasattr(act_space, "high"):
            low_arr  = np.asarray(act_space.low).reshape(-1)
            high_arr = np.asarray(act_space.high).reshape(-1)
            if low_arr.size >= 1 and high_arr.size >= 1:
                lo = float(low_arr[0])
                hi = float(high_arr[0])
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    env_cwnd_min = lo
                    env_cwnd_max = hi
    except Exception:
        pass

    if env_cwnd_min is not None and env_cwnd_max is not None:
        CWND_MIN = env_cwnd_min
        CWND_MAX = env_cwnd_max
        print(f"  Effective cwnd range (from ns-3 action space): [{CWND_MIN/1024:.2f} KB, {CWND_MAX/1024:.1f} KB]")
        if (abs(CWND_MIN - requested_cwnd_min) > 1e-6 or
                abs(CWND_MAX - requested_cwnd_max) > 1e-6):
            print("  NOTE: CLI cwnd bounds differ from ns-3; using ns-3 bounds for consistency.")
    else:
        print(f"  Effective cwnd range (CLI fallback): [{CWND_MIN/1024:.2f} KB, {CWND_MAX/1024:.1f} KB]")
    print(f"  midpoint:    {(CWND_MIN+CWND_MAX)/2/1024:.1f} KB  (tanh=0)")

    obs_size  = len(obs)
    # New 11-dim format: [cwnd,rtt,bif]×3 + [dropsL1,dropsL2]
    if obs_size >= 5 and obs_size % 3 == 2:
        num_flows = (obs_size - 2) // 3
        has_drops = True
    else:
        num_flows = obs_size // 3
        has_drops = False
    state_dim = obs_size
    print(f"  Auto-detected: obs_size={obs_size} → {num_flows} flows"
          f"{' + drop counters' if has_drops else ''}")

    feature_names = []
    for fi in range(num_flows):
        feature_names += [f"flow{fi}_cwnd", f"flow{fi}_rtt", f"flow{fi}_bif"]
    if has_drops:
        feature_names += ["drops_L1", "drops_L2"]

    # ── Build networks ─────────────────────────────────────────────────────────
    actor          = LSTMActor(state_dim, HIDDEN_DIM, NUM_LAYERS)
    critic1        = LSTMCritic(state_dim, HIDDEN_DIM, NUM_LAYERS)
    critic2        = LSTMCritic(state_dim, HIDDEN_DIM, NUM_LAYERS)
    critic1_target = LSTMCritic(state_dim, HIDDEN_DIM, NUM_LAYERS)
    critic2_target = LSTMCritic(state_dim, HIDDEN_DIM, NUM_LAYERS)

    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())

    actor_opt   = optim.Adam(actor.parameters(),   lr=LR_ACTOR)
    critic1_opt = optim.Adam(critic1.parameters(), lr=LR_CRITIC)
    critic2_opt = optim.Adam(critic2.parameters(), lr=LR_CRITIC)

    # SAC entropy temperature — auto-tuned
    log_alpha = torch.zeros(1, requires_grad=True)
    alpha_opt = optim.Adam([log_alpha], lr=LR_ALPHA)

    buf  = SequenceReplayBuffer(BUFFER_CAP, SEQ_LENGTH)
    norm = RunningNormalizer(state_dim)

    # Episode counter from CSV (robust to checkpoint failure)
    if not args.fresh and os.path.exists(args.trainlog):
        try:
            with open(args.trainlog) as f:
                episode = max(0, sum(1 for ln in f if ln.strip()) - 1)
        except Exception:
            episode = 0
    else:
        episode = 0

    total_steps = 0
    total_it    = 0

    # ── Load checkpoint ───────────────────────────────────────────────────────
    if args.fresh and os.path.exists(args.model):
        print(f"  --fresh: ignoring '{args.model}'")
    elif os.path.exists(args.model):
        try:
            ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
            actor.load_state_dict(ckpt["actor"])
            critic1.load_state_dict(ckpt["critic1"])
            critic2.load_state_dict(ckpt["critic2"])
            critic1_target.load_state_dict(ckpt["critic1_target"])
            critic2_target.load_state_dict(ckpt["critic2_target"])
            actor_opt.load_state_dict(ckpt["actor_opt"])
            critic1_opt.load_state_dict(ckpt["critic1_opt"])
            critic2_opt.load_state_dict(ckpt["critic2_opt"])
            if "log_alpha" in ckpt:
                log_alpha.data.copy_(ckpt["log_alpha"])
            if "alpha_opt" in ckpt:
                alpha_opt.load_state_dict(ckpt["alpha_opt"])
            norm.mean   = ckpt.get("norm_mean",   norm.mean)
            norm.var    = ckpt.get("norm_var",    norm.var)
            norm.count  = ckpt.get("norm_count",  norm.count)
            total_steps = ckpt.get("total_steps", 0)
            total_it    = ckpt.get("total_it",    0)
            print(f"  Loaded: episode={episode}, α={log_alpha.exp().item():.4f}")
        except Exception as exc:
            print(f"  WARNING: ckpt load failed ({exc}) — fresh weights.")
    else:
        print("  No saved model — starting fresh.")

    critic1_target.eval();  critic2_target.eval()

    if args.eval:
        actor.eval()
        print("  EVAL MODE: deterministic policy, weights frozen.\n")

    # ── Episode init ──────────────────────────────────────────────────────────
    norm.update(obs)
    state_history = deque([norm.normalize(obs)] * SEQ_LENGTH, maxlen=SEQ_LENGTH)

    actor_hidden  = None
    done          = False
    step          = 0
    total_reward  = 0.0

    ep_rewards:     list = []
    ep_tputs:       list = []
    ep_rtts:        list = []
    ep_crit_losses: list = []
    ep_act_losses:  list = []
    ep_cwnds:       list = []
    ep_state_seqs:  list = []   # sampled for attribution

    alpha_val = log_alpha.exp().item()
    print(f"  Episode {episode+1}  |  α={alpha_val:.4f}  |  "
          f"{num_flows} flows  |  cwnd [{CWND_MIN/1024:.1f},{CWND_MAX/1024:.1f}] KB")
    print(f"  {'Step':>5}  {'Mode':>7}  {'Cwnd_KB':>9}  "
          f"{'Reward':>8}  {'Tput':>6}  {'RTT':>6}  {'alpha':>6}")
    print("  " + "─" * 60)

    # ── Main loop ─────────────────────────────────────────────────────────────
    while not done:
        step        += 1
        total_steps += 1

        state_seq    = np.array(list(state_history), dtype=np.float32)
        state_tensor = torch.FloatTensor(state_seq).unsqueeze(0)   # (1,T,D)

        # ── SAC action selection ─────────────────────────────────────────
        if not args.eval:
            # Stochastic policy — entropy-driven exploration
            with torch.no_grad():
                tanh_action, _lp, actor_hidden = actor.sample(
                    state_tensor, actor_hidden)
            actor_hidden = (actor_hidden[0].detach(), actor_hidden[1].detach())
            tanh_val    = tanh_action.item()
            target_cwnd = float(tanh_to_cwnd(torch.tensor(tanh_val),
                                             CWND_MIN, CWND_MAX))
            tanh_stored = tanh_val
            mode = "explore"
        else:
            # Deterministic for evaluation
            with torch.no_grad():
                tanh_out, actor_hidden = actor(state_tensor, actor_hidden)
            actor_hidden = (actor_hidden[0].detach(), actor_hidden[1].detach())
            tanh_val    = tanh_out.item()
            target_cwnd = float(tanh_to_cwnd(torch.tensor(tanh_val),
                                             CWND_MIN, CWND_MAX))
            tanh_stored = tanh_val
            mode = "exploit"

        ep_cwnds.append(target_cwnd)

        # Step ns-3 — send cwnd as a 1-element list (Box action)
        next_obs, _cpp_r, done, _ = env.step([target_cwnd])
        next_obs = np.array(next_obs, dtype=np.float32)

        reward       = compute_reward(next_obs, bottleneck_bps, args.min_rtt_ms)
        total_reward += reward
        ep_rewards.append(reward)

        act_rtts = [float(next_obs[i*3+1]) for i in range(num_flows)
                    if next_obs[i*3+2] > 0 and next_obs[i*3+1] > 0]
        act_bifs = [float(next_obs[i*3+2]) for i in range(num_flows)
                    if next_obs[i*3+2] > 0 and next_obs[i*3+1] > 0]
        if act_rtts:
            ep_tputs.append(sum(b/(r/1000) for b,r in zip(act_bifs,act_rtts))*8/1e6)
            ep_rtts.append(float(np.mean(act_rtts)))
        else:
            ep_tputs.append(0.0); ep_rtts.append(0.0)

        norm.update(next_obs)
        norm_cur  = norm.normalize(obs)
        norm_next = norm.normalize(next_obs)
        state_history.append(norm_next)

        if step % 10 == 0:
            ep_state_seqs.append(state_seq.copy())

        if not args.eval:
            buf.push(norm_cur, tanh_stored, reward, norm_next, done)
            if step % TRAIN_EVERY == 0 and len(buf) >= BATCH_SIZE:
                total_it += 1
                cl, al, alpha_val = train_step_sac(
                    actor, critic1, critic2,
                    critic1_target, critic2_target,
                    buf, actor_opt, critic1_opt, critic2_opt,
                    log_alpha, alpha_opt, TARGET_ENTROPY,
                    BATCH_SIZE, GAMMA,
                )
                ep_crit_losses.append(cl)
                ep_act_losses.append(al)

        if step % 10 == 0:
            print(f"  {step:5d}  {mode:>7}  "
                  f"{target_cwnd/1024:9.2f}  "
                  f"{reward:+8.3f}  "
                  f"{ep_tputs[-1]:6.2f}  {ep_rtts[-1]:6.1f}  {log_alpha.exp().item():.4f}")

        obs = next_obs

    # ── Episode summary ───────────────────────────────────────────────────────
    episode     += 1
    avg_r        = float(np.mean(ep_rewards))                if ep_rewards      else 0.0
    avg_cl       = float(np.mean(ep_crit_losses))            if ep_crit_losses  else 0.0
    avg_al       = float(np.mean(ep_act_losses))             if ep_act_losses   else 0.0
    avg_t        = float(np.mean(ep_tputs))                  if ep_tputs        else 0.0
    avg_rtt      = float(np.mean([r for r in ep_rtts if r>0])) if ep_rtts       else 0.0
    avg_cwnd     = float(np.mean(ep_cwnds))                  if ep_cwnds        else 0.0

    print(f"\n{'='*72}")
    print(f"  Episode {episode} complete")
    print(f"  Steps={step}  TotalReward={total_reward:.2f}  AvgReward={avg_r:.4f}")
    print(f"  AvgTput={avg_t:.2f} Mbps  AvgRTT={avg_rtt:.1f} ms  α={log_alpha.exp().item():.4f}")
    print(f"  CritLoss={avg_cl:.5f}  ActorLoss={avg_al:.5f}")
    if ep_cwnds:
        print(f"  TargetCwnd avg={avg_cwnd/1024:.2f} KB  "
              f"min={min(ep_cwnds)/1024:.2f} KB  max={max(ep_cwnds)/1024:.2f} KB")

    # Feature attribution — printed every episode during training
    if ep_state_seqs and not args.eval:
        compute_feature_attribution(actor, np.array(ep_state_seqs), feature_names)

    print(f"{'='*72}")

    # ── Training log ──────────────────────────────────────────────────────────
    if not args.eval:
        write_hdr = (not os.path.exists(args.trainlog) or
                     os.path.getsize(args.trainlog) == 0)
        with open(args.trainlog, "a", newline="") as fh:
            w = csv.writer(fh)
            if write_hdr:
                w.writerow(["Episode","Steps","AvgReward","AvgCritLoss",
                             "AvgActorLoss","AvgTput_Mbps","AvgRTT_ms",
                             "AvgCwnd_KB","ExplNoise"])
            w.writerow([episode, step,
                        f"{avg_r:.5f}",   f"{avg_cl:.5f}",
                        f"{avg_al:.5f}",  f"{avg_t:.4f}",
                        f"{avg_rtt:.2f}", f"{avg_cwnd/1024:.3f}",
                        f"{log_alpha.exp().item():.5f}"])

    if not args.eval:
        torch.save({
            "actor": actor.state_dict(),
            "critic1": critic1.state_dict(), "critic2": critic2.state_dict(),
            "critic1_target": critic1_target.state_dict(),
            "critic2_target": critic2_target.state_dict(),
            "actor_opt": actor_opt.state_dict(),
            "critic1_opt": critic1_opt.state_dict(),
            "critic2_opt": critic2_opt.state_dict(),
            "log_alpha": log_alpha.data, "alpha_opt": alpha_opt.state_dict(),
            "norm_mean": norm.mean, "norm_var": norm.var, "norm_count": norm.count,
            "episode": episode,
            "total_steps": total_steps, "total_it": total_it,
        }, args.model)
        print(f"  Model saved → {args.model}")
        print(f"  SAC α = {log_alpha.exp().item():.5f}")

    env.close()
    print("  Done.\n")


if __name__ == "__main__":
    main()
