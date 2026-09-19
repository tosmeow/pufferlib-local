Architecture
============

For the CPU/GPU training call paths, loss and optimizer implementations,
logging behavior, and algorithm extension points, see the
:download:`training code map (Markdown) <training_code_map.md>`.

Mental Model
------------

PufferLib 4.0 is organized around a compiled environment backend and a compact
training loop.

The main pieces are:

* ``build.sh``: compiles an environment-specific backend into ``pufferlib/_C...so``
* ``config/*.ini``: defines environment, vectorization, policy, training, and sweep options
* ``ocean/*``: C environments and bindings
* ``src/vecenv.h``: vectorized environment runtime
* ``src/bindings.cu``: CUDA/native training backend
* ``src/bindings_cpu.cpp``: CPU vector backend used by the PyTorch fallback
* ``src/metal``: incremental native Apple Metal backend (primitives, GEMM, linear layers, and MinGRU)
* ``pufferlib/pufferl.py``: CLI, config loading, training/eval/sweep orchestration
* ``pufferlib/torch_pufferl.py``: PyTorch fallback training backend
* ``pufferlib/models.py``: default policy/model components
* ``pufferlib/sweep.py``: Protein hyperparameter sweep logic

Backend Selection
-----------------

``pufferlib.pufferl`` imports ``pufferlib._C`` at startup. That extension is
produced by ``build.sh`` and is specific to the environment you built. If you
build ``breakout``, then ``_C.env_name`` should be ``"breakout"``.

``pufferlib.pufferl._resolve_backend`` selects:

* ``pufferlib.torch_pufferl.PuffeRL`` when ``--slowly`` is passed or the compiled
  extension reports ``_C.gpu == 0`` (CPU build)
* the native CUDA ``_C`` trainer otherwise

The PyTorch backend chooses CUDA, Apple MPS, or CPU through
``--torch.device auto`` (the default). On Apple Silicon, use ``--slowly`` after
a ``--cpu`` build; the policy and optimizer run on MPS when it is available,
while the compiled vector environment remains on CPU.

Environment Template
--------------------

For writing a new Ocean environment, start from:

* ``ocean/squared`` for a compact single-agent example
* ``ocean/target`` for a compact multi-agent example
* ``config/squared.ini`` or ``config/target.ini`` for config shape

The binding file is where observation/action metadata becomes visible to the
Python side.
