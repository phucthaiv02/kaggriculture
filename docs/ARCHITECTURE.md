# Dynamic agent architecture

## State encoder

The policy consumes the current observation on every turn. It never indexes an
action by episode step.

- A 44-channel `10 × 10` board contains both farms, objects, unit locations,
  age, yield and care/water/feed state.
- A 70-value global vector contains money, land, seeds, shed/carried inventory,
  market prices/inventory and shop multiplicity.
- Up to 32 unit tokens contain position, local tile and carried inventory.

The board passes through a residual CNN. Board tokens, the global token and all
active unit tokens then pass through a pre-norm Transformer encoder.

## Autoregressive action decoder

Farmer and farm-hand actions are decoded in unit order. Each decision conditions
the next decision through a GRU hidden state and an embedding of the previously
selected operation/item/quantity. Market orders use a second autoregressive
decoder for up to ten ordered operations.

At inference, `ReservationState` updates available seeds, shed inventory, unit
inventory and money after every decoded choice. Legal masks therefore prevent
multiple units from spending the same final seed or several market orders from
selling the same inventory.

## Training stages

1. Behavior cloning from public high-level replays.
2. Evaluation across complete games, not just offline classification accuracy.
3. Stochastic rollouts from the cloned policy to expose distribution-shift
   states.
4. Advantage-weighted regression (AWR), where actions from episodes exceeding
   the learned value receive exponentially larger weights.
5. On-policy rollout against a weighted pool of current and historical policies.
6. PPO with GAE, a clipped policy objective, value/entropy losses and a frozen
   BC KL anchor.
7. Paired-seed head-to-head evaluation; only promoted candidates enter the
   historical opponent league.

AWR is used for the first recovery iterations because Kaggriculture simulation
is CPU-heavy and BC initially has little coverage outside expert states. PPO
rollouts are also generated separately from GPU optimization: CPU actors write
trajectory shards, then the H100 performs several on-policy epochs without
waiting for synchronous environments.

## Relative value and reward

BC, AWR and PPO use the same terminal target family rather than mixing absolute
bank balance with competitive outcome:

```
score_margin / value_scale + win_bonus * {-1, 0, 1}
```

The target is clipped for outlier control. PPO places it on the final transition
and propagates it backward with generalized advantage estimation. This makes the
policy optimize performance relative to its opponent while retaining score
margin information when win/loss alone is too sparse.

## PPO action likelihood

The rollout policy records the exact dynamic legal mask used at every decoder
step. A turn's joint log-probability is the sum over active autoregressive
components only: operation, optional item and optional quantity for all active
units and market orders. The `NONE` operation that ends the market sequence is
included; unused slots after it are excluded.

The same `ReservationState` transition code drives normal inference and traced
rollout inference. This preserves the action distribution when seeds, money or
inventory are reserved by an earlier decision in the same turn.

## Opponent league and promotion

Rollouts sample from the current policy, the starter baseline and promoted
historical snapshots. The current implementation uses explicit weights in
`configs/opponent_league.json`; those weights can later be updated from matchup
statistics to implement PFSP without changing the rollout format.

PPO writes a candidate checkpoint. It never replaces the incumbent directly.
Candidates play paired seeds from both positions and must pass win-rate,
mean-margin and lower-tail margin thresholds before becoming `agent_best.pt`.
Promoted checkpoints are immutable league snapshots.

## H100 execution

The trainer enables BF16 autocast, TF32, fused AdamW, pinned asynchronous host
transfers, persistent loader workers, gradient checkpointing, `torch.compile`
and NCCL DDP. Replay data is stored in independently shuffled safetensors shards
so workers never deserialize the full corpus into RAM.
