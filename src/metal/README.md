# Native Metal backend

This directory contains the incremental Apple GPU port of the native CUDA
trainer. It is separate from PyTorch MPS: kernels are written in Metal Shading
Language and dispatched directly through Metal.

The C API is not wired into the training CLI yet. That is intentional: these
milestones establish trustworthy buffer, dispatch, model-kernel, and numerical
baselines before trainer integration makes failures harder to isolate.

## Milestone 1: runtime and primitive parity

The first milestone provides:

- a Metal device, command queue, pipeline cache, and synchronous dispatch API;
- shared `MTLBuffer` allocations visible to both CPU environments and the GPU;
- float32 fill, add, clamp, uint8-to-float cast, and `[A,B,C] -> [B,A,C]`
  transpose kernels mirroring operations in `src/kernels.cu`;
- bounds validation, empty-tensor behavior, and surfaced Metal errors;
- CPU differential tests over boundary sizes and repeated dispatches.

Build the development dynamic library with:

```bash
scripts/build_metal_backend.sh
```

Run the tests with direct Apple GPU access:

```bash
.venv/bin/python -m unittest -v tests.test_metal_backend
```

`puf_metal_create` accepts either `kernels.metal` source or a precompiled
`.metallib`. Runtime compilation keeps development possible with Command Line
Tools alone; release builds should precompile the library with full Xcode.

## Milestone 2: float32 GEMM

The backend uses Apple's optimized `MPSMatrixMultiplication` over the same
shared, tightly packed row-major buffers. `puf_metal_gemm_f32` covers every
transpose combination plus BLAS `alpha` and `beta` semantics. Its cache avoids
rebuilding an MPS kernel for repeated calls with the same operation signature.
`PufMetalCommandBatch` records dependent GEMMs into one command buffer and can
commit asynchronously, so synchronization can occur at model boundaries rather
than after every matrix operation.

Correctness tests compare Metal results with NumPy CPU results over rectangular
and odd dimensions, exact identity products, all transpose modes, nontrivial
scales, empty dimensions, and invalid inputs. Run the representative synchronous
GEMM benchmark with:

```bash
scripts/benchmark_metal_gemm.py
```

The same benchmark also records the default-size encoder, four MinGRU
projections, and decoder projection into one command buffer. This measures the
operation mix separately from the recurrent scan.

## Milestone 3: linear layers and MinGRU

Bias-free linear forward/backward functions implement the matrix layouts used
by the default encoder, decoder, and recurrent projection. Metal MinGRU kernels
cover rollout gating and checkpointed training forward/backward scans. The
backward scan recomputes four-timestep chunks from forward checkpoints and uses
the direct recurrence derivative, which remains finite when the initial state
is zero.

Tests validate linear and recurrent gradients against independently derived CPU
gradients and finite differences. Forward tests include exact CUDA-style
rollout nonlinearities, direct CPU recurrences, sparse checkpoint contents,
in-place highway updates, empty horizons, and horizons not divisible by four.
The benchmark includes a full encoder + four MinGRU scans + decoder forward
microbenchmark in one command buffer.

## Planned milestones

1. Decoder gradient assembly and complete encoder/decoder integration.
2. Native rollout, sampling, advantage, and PPO loss.
3. Muon and fused training command submission.
4. Convolutional and environment-specific kernels from `ocean.cu`.
