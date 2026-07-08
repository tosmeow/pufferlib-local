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

For calibrated runs, set `params_path` in the config or pass it on the CLI.
The calibration CSVs are intentionally external data, not vendored here.

## Own-Order Accounting

The QR core order book remains an aggregate price-level book. The environment
adds a small private overlay in `QueueReactive::own.orders` for our own resting
orders. This keeps the hot path fixed-size (`QR_MAX_ORDERS`) and avoids adding
per-order state to the calibrated simulator.

The main `QueueReactive` struct keeps Puffer ABI fields flat, then groups QR
state into small sub-structures: `own`, `account`, `last_action`, `last_qr`,
`clock`, and `status`.

Each own order stores:

- `side`, `price`, `remaining`: the resting order still owned by the agent.
- `ahead`: aggregate queue volume at the same price strictly before this order.
- `id`: insertion order for resolving same-price priority.

The update hooks are in `queue_reactive.cpp`:

- Agent limit/improve actions insert into both the aggregate book and the
  private overlay. A plain limit starts with `ahead = volume_at(price)`;
  an improve/create order starts at the front with `ahead = 0`.
- QR trades call `apply_passive_fills`: marketable flow first consumes `ahead`,
  then fills own orders in queue order. Fills immediately update cash,
  inventory, and remove fully filled orders.
- QR public cancels never cancel our own orders. The env first caps the cancel
  size to public volume (`aggregate - own`). It then samples where the public
  cancel happened across inferred public queue segments around our orders. Only
  cancels sampled before an own order reduce that order's `ahead`.
- Agent cancels currently ignore the price/slot action component and pop the
  worst-positioned order on the requested side: worse price first, then deeper
  queue position, then newest insertion. Later we can replace this with an
  explicit order-id/slot cancel action.

Cash/inventory are only changed when fills occur. Passive buy fills increase
inventory and spend cash; passive sell fills decrease inventory and receive
cash. Aggressive market fills use the same sign convention.
