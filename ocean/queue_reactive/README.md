# queue_reactive

PufferLib wrapper for a queue-reactive limit order book simulator.

This folder is self-contained with respect to the queue-reactive C++ model:

- `queue_reactive.cpp`, `queue_reactive.h`, `binding.c`: Puffer environment.
- `qr_core/`: vendored minimal C++ queue-reactive core.
- `queue_reactive.ini`: env-local default config.

The repo build script still needs the `queue_reactive` C++ special case so the
extra `qr_core/src/*.cpp` files are compiled into the native backend.

Build and run:

```bash
puffer build queue_reactive --cpu
puffer replay queue_reactive --fps 10
```

Phase-one training uses MPS automatically on a supported Apple machine, or it
can be selected explicitly:

```bash
puffer train queue_reactive --slowly --torch.device mps
```

The phase-one policy deliberately sees only the 16-value order book, inventory
(`18`), alpha (`21`), and time remaining (`25`). Its action is conditional:
first choose `Noop` or `Market`; only a market decision samples side, crossing
depth `0..3`, and size. For Noop those ignored fields are fixed to zero and do
not contribute to PPO log-probability. This is important here because waiting
is a genuine trading decision, often the right one, rather than a placeholder.
The phase-one default also sets `market_residual_rests = 0`, so unfilled market
residuals are cancelled immediately instead of creating passive own orders.
Rejected interventions receive an immediate configurable penalty of `0.01`;
Noop and partially filled market actions do not receive this penalty.

The full 44-value observation and four-field action ABIs remain unchanged. The
dormant encoder and decoder parameters are also retained in checkpoints, which
allows strict loading into the later curriculum phases. Start a fresh phase-one
run when switching from the old `DefaultEncoder` checkpoints.

To activate all observations and action types while freezing the learned core
encoder and recurrent network:

```bash
puffer train queue_reactive --slowly --torch.device mps \
  --torch.encoder QueueReactiveExpandedEncoder \
  --torch.decoder DefaultDecoder \
  --torch.freeze-modules encoder.core,network \
  --load-model-path latest
```

For the final fine-tuning phase, load that checkpoint with the same expanded
encoder/full decoder, omit `--torch.freeze-modules`, and normally lower the
learning rate:

```bash
puffer train queue_reactive --slowly --torch.device mps \
  --torch.encoder QueueReactiveExpandedEncoder \
  --torch.decoder DefaultDecoder \
  --train.learning-rate 0.0003 \
  --load-model-path latest
```

For calibrated runs, set `params_path` in the config or pass it on the CLI.
The calibration CSVs are intentionally external data, not vendored here.

The phase-one learning configuration disables the power-law market-impact
model so the first policy learns the alpha/order-book relationship without a
second directional process. The phase-one alpha intensity scale is `2.0`; the
unscaled OU alpha remains the value exposed at observation index `21`.

The constant-alpha calibration gives a useful economic interpretation of this
choice. At scale `2`, raw alpha `0.125` corresponds to effective bias `0.25`
and about `0.91` tick of mean ten-second drift, while raw alpha `0.25`
corresponds to effective bias `0.5` and about `1.82` ticks. Against an
approximately one-tick aggressive round trip, this leaves a genuine weak-signal
region where Noop is preferable while making stronger signals learnable. Scale
`1` makes profitable examples comparatively sparse; scale `4` makes even weak
signals profitable and is better reserved as an optional warm-start curriculum.

Later phases can enable `use_power_law_impact = 1`; with `strategy_impact = 1`,
filled aggressive agent trades then feed the impact process and bias subsequent
queue-reactive order flow. The impact parameters remain configured in
`queue_reactive.ini`.

To measure how that intensity bias propagates into price without any agent
interaction, run the no-op benchmark after building the CPU environment:

```bash
python scripts/benchmark_queue_alpha_drift.py \
  --output alpha_drift_results.json
```

It runs parallel replicas at several alpha scales, both with public power-law
impact disabled (the phase-one default) and enabled. It
reports unconditional drift, an alpha/drift regression slope, and the more
direct alpha-aligned drift `sign(alpha_t) * (mid_{t+T} - mid_t)`. The aligned
statistic is the useful one because the mean-zero OU alpha implies near-zero
unconditional drift even when alpha is strongly predictive.

For the simpler causal transmission curve, hold alpha constant in one direction,
disable public impact, submit only Noop, and measure final minus initial mid at
one fixed horizon:

```bash
python scripts/benchmark_queue_constant_alpha_drift.py
```

This stores every replica's price change—not only the aggregate—in
`benchmarks/queue_reactive/constant_alpha_drift.json`. The default experiment
uses constant raw alpha `+1`, a 10-second horizon, 4,096 replicas per scale, and
a base-two logarithmic scale grid plus an unbiased `0` baseline. Pass
`--constant-alpha -1` to run the opposite direction.

## Learning Environment at a Glance

| Interface | Current value |
| --- | --- |
| Agents | 1 |
| Observation | 44 `float32` values |
| Action | `MultiDiscrete([5, 2, 8, 6])` encoded as four floats by the native ABI |
| Reward | Change in marked-to-mid wealth minus quadratic inventory cost, emitted at event boundaries at least 20 ms apart |
| Terminal | Simulated-time horizon, 60 seconds by default |
| Market evolution | One sampled QR event per `c_step` |
| Private capacity | One resting bid and one resting ask |

The observation is a single policy-facing view of both sides of the learning
problem: the public market/environment state and the agent's private account,
last action, and resting orders. All ranges below are zero-based and inclusive.

| Indices | Count | Owner | Contents |
| --- | ---: | --- | --- |
| `0..15` | 16 | Environment | Four bid/ask price levels and their aggregate volumes |
| `16..17` | 2 | Environment | Spread and book imbalance |
| `18..20` | 3 | Agent | Inventory, cash, and marked-to-mid PnL |
| `21..25` | 5 | Environment | Alpha, market-impact bias, QR/agent latency, and clock phase |
| `26..33` | 8 | Agent | Last fill, active-order utilization, and last-action result |
| `34..39` | 6 | Environment | Last QR event |
| `40..43` | 4 | Agent | Fixed bid and ask slots, two values per slot |

In Python, the structural slices are:

```python
book = observation[0:16].reshape(4, 4)
state = observation[16:28]
last_action = observation[28:34]
last_qr_event = observation[34:40]
own_orders = observation[40:44].reshape(2, 2)
```

### Environment state

Each book row `book[level]`, starting at the best quotes with `level == 0`, is:

```text
[bid_price_from_mid / 16, bid_volume / 5000,
 ask_price_from_mid / 16, ask_volume / 5000]
```

The remaining environment fields are:

| Index | Field | Encoding |
| ---: | --- | --- |
| 16 | Spread | `spread / 16` |
| 17 | Imbalance | Queue-reactive book imbalance |
| 21 | Alpha | Current alpha value, unscaled |
| 22 | Impact bias | Current impact bias factor, unscaled |
| 23 | QR inter-event time | Last sampled QR delay in seconds |
| 24 | Agent latency | Last sampled agent delay in seconds; `0` for no-op |
| 25 | Time remaining | Fraction of the configured episode duration remaining, from `1` at reset to `0` at expiry |
| 34 | Last QR event type | Event enum divided by `4` |
| 35 | Last QR side | Bid/buy `+1`, ask/sell `-1` |
| 36 | Last QR size | `size / 5000` |
| 37 | Last QR price | `(price - current_mid) / 16`, or `0` when unset |
| 38 | QR rejected | Boolean `0` or `1` |
| 39 | QR partial | Boolean `0` or `1` |

QR event types before normalization are `0=Add`, `1=Cancel`, `2=Trade`,
`3=CreateBid`, and `4=CreateAsk`.

### Agent state

| Index | Field | Encoding |
| ---: | --- | --- |
| 18 | Inventory | `inventory / 20000` |
| 19 | Cash | `cash / 30_000_000` |
| 20 | Mark-to-mid PnL | `(cash + inventory * mid) / 30_000_000` |
| 26 | Filled size | Quantity filled for the agent during the last step, passive and/or aggressive, divided by `5000` |
| 27 | Order capacity used | Active private orders divided by `2` |
| 28 | Last action type | Action type divided by `4` |
| 29 | Last action side | Buy/bid `+1`, sell/ask `-1` |
| 30 | Action rejected | Boolean `0` or `1` |
| 31 | Action partial | Boolean `0` or `1` |
| 32 | Lost latency race | Boolean `0` or `1` |
| 33 | Last action price | `(price - current_mid) / 16`, or `0` when unset |

The two `own_orders` rows are fixed by side: row `0` is the bid slot and row
`1` is the ask slot. Each row is:

```text
[(price - current_mid) / 16, remaining / 5000]
```

An inactive slot is `[0, 0]`; a positive remaining quantity is the activity
mask. Queue-ahead volume remains tracked internally for fill accounting but is
not exposed to the policy.

Not every field in the C struct is policy-visible. In particular, the policy
does not receive the absolute step/time, private insertion ids,
`calibration_loaded`, or `config_error`; use the native state or renderer when
debugging those values.

## Action Space

An action has the form `[type, side, price_or_depth, size_bucket]`:

| Component | Values | Meaning |
| --- | --- | --- |
| Type | `0..4` | `Noop`, `Limit`, `Cancel`, `Market`, `Improve` |
| Side | `0..1` | `Buy/Bid`, `Sell/Ask` |
| Price/depth | `0..7` | Meaning depends on action type; see below |
| Size bucket | `0..5` | `100, 200, 400, 800, 1600, 3200`, multiplied by `lot_size` |

Price/depth semantics:

- `Limit`: `0` is the same-side best quote and `1..3` are successively deeper
  levels. Values `4..7` currently clamp to level `3`.
- `Market`: the value is a crossing limit depth on the opposite book, from the
  best quote (`0`) through level `3`. Values `4..7` clamp to level `3`. The
  default `market_residual_rests = 0` gives IOC behavior and cancels an unfilled
  residual. Set it to `1` to let the residual become a private resting order;
  if that side's slot is already occupied, it is still cancelled.
- `Improve`: ignored; the order improves the same-side best quote by one tick
  and is rejected when the spread is already one tick.
- `Cancel`: price/depth is ignored; the environment cancels the selected
  side's private order. The size bucket is ignored as well.
- `Noop`: side, price/depth, and size are ignored.

## Step and Latency Ordering

Every step samples one public QR event and its delay. For any non-noop action,
the environment also samples agent latency:

1. If agent latency is strictly smaller, the agent intervention is processed,
   time advances to the QR event, and the QR event is then processed.
2. Otherwise the QR event happens and the submitted intervention is dropped;
   `lost_race` is set to `1`.
3. Reward, terminal, observation, and aggregate logs are updated.

Consequently, a valid submitted action may have no effect without being marked
`rejected`; inspect `lost_race` separately. `max_events_per_step` and
`report_interval` are accepted configuration fields but are not used by the
current step loop.

At reset, the configured four-level book is recreated, inventory and cash are
zero, all private orders are inactive, and the clock starts at zero. The two
trace blocks contain placeholders until the first step: last action is
`Noop/Buy` and the QR trace uses enum value `0` (`Add`) with size zero, both at
`initial_bid`.

## Training and Debug Signals

The binding reports `episode_return`, `episode_length`, `agent_fills`,
`agent_orders`, `agent_cancels`, `agent_rejected`, `agent_lost_race`,
`qr_events`, `qr_trades`, `inventory`, `inventory_penalty`, and
`terminal_inventory_penalty`, in addition to `perf` and `score`.
The raylib replay view shows the live book ladder, spread/imbalance, last QR
event, last agent action, inventory, marked-to-mid PnL, and active-order count. Resting
private orders have gold `YOUR BID` / `YOUR ASK` markers on their ladder rows,
including remaining and queue-ahead quantities; orders outside the four visible
levels get an off-ladder marker.

Reward is expressed in trainer-sized units. Marked-to-mid PnL is divided by a
configurable notional scale, and inventory is normalized before applying the
quadratic costs:

```text
pnl_reward = delta(cash + inventory * mid) / pnl_reward_divisor
holding_cost = inventory_penalty_coef
               * (inventory / 20000)^2 * elapsed_seconds
rejection_cost = agent_rejection_penalty * action_rejected
terminal_cost = min(1, terminal_inventory_penalty_coef
                       * (inventory / terminal_inventory_target)^2)
reward = pnl_reward - holding_cost - rejection_cost - terminal_cost_at_expiry
```

The defaults are `pnl_reward_divisor = 3200`,
`inventory_penalty_coef = 0.1`, `agent_rejection_penalty = 0.01`, and
`terminal_inventory_target = 3200`. A reward
mark is taken at the first QR event boundary at least `reward_interval_ms`
after the previous mark; the default interval is 20 ms. The event-driven loop
does not interpolate book state at an exact wall-clock grid. The dashboard's
`inventory_penalty` is the positive normalized cost subtracted from reward,
averaged over logged transitions.

Episodes expire on the first QR event boundary at or after
`episode_duration_seconds`, which defaults to 60 simulated seconds. The policy
observes the fraction of time remaining. At expiry, pending marked-to-mid PnL
and holding cost are flushed even if the normal 20 ms reward interval has not
elapsed, then the normalized terminal cost is subtracted. With the default
coefficient of `1`, a 3200-unit or larger terminal position costs the maximum
`1`, while smaller positions retain a graded quadratic cost. The environment
resets immediately after the terminal transition.

## Own-Order Accounting

The QR core order book remains an aggregate price-level book. The environment
adds a small private overlay in `QueueReactive::own.orders` for our own resting
orders. Slot `0` is reserved for the agent's bid and slot `1` for its ask. This
keeps the hot path fixed-size and avoids adding per-order state to the
calibrated simulator.

The main `QueueReactive` struct keeps Puffer ABI fields flat, then groups QR
state into small sub-structures: `own`, `account`, `reward_state`,
`last_action`, `last_qr`, `clock`, and `status`.

Each own order stores:

- `side`, `price`, `remaining`: the resting order still owned by the agent.
- `ahead`: aggregate queue volume at the same price strictly before this order.
- `id`: insertion order for resolving same-price priority.

The update hooks are in `queue_reactive.cpp`:

- Agent limit/improve actions insert into both the aggregate book and the
  side's private slot. They are rejected while that side's slot is occupied. A
  plain limit starts with `ahead = volume_at(price)`; an improve/create order
  starts at the front with `ahead = 0`.
- QR trades call `apply_passive_fills`: marketable flow first consumes `ahead`,
  then fills own orders in queue order. Fills immediately update cash,
  inventory, and remove fully filled orders.
- QR public cancels never cancel our own orders. The env first caps the cancel
  size to public volume (`aggregate - own`). It then samples where the public
  cancel happened across inferred public queue segments around our orders. Only
  cancels sampled before an own order reduce that order's `ahead`.
- Agent cancels ignore the price/slot action component and cancel the single
  private order on the requested side.

Cash/inventory are only changed when fills occur. Passive buy fills increase
inventory and spend cash; passive sell fills decrease inventory and receive
cash. Aggressive market fills use the same sign convention.
