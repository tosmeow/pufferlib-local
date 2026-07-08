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
