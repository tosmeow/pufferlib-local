Local Setup
===========

Python Environment
------------------

The local virtual environment was created with ``uv`` and Python 3.12:

.. code-block:: bash

   uv --cache-dir .uv-cache venv --python /opt/anaconda3/bin/python3.12 .venv
   uv --cache-dir .uv-cache pip install -e .

Use ``uv --cache-dir .uv-cache pip ...`` for package changes. This uv-created
environment may not include ``pip`` as a Python module.

Apple Silicon Backend
---------------------

PufferLib needs a compiled native backend for each environment. On this Mac, use
the CLI build wrapper:

.. code-block:: bash

   puffer build breakout

On Apple Silicon, this wraps the required compiler choices:

* Homebrew LLVM for OpenMP support
* Homebrew raylib linked dynamically
* the project venv's Python headers and PyTorch ``libomp.dylib``
* ARM-safe build flags

Smoke Test
----------

.. code-block:: bash

   .venv/bin/puffer train breakout --slowly \
     --torch.device mps \
     --train.total-timesteps 256 \
     --vec.total-agents 32 \
     --vec.num-buffers 1 \
     --vec.num-threads 4 \
     --train.horizon 8 \
     --train.minibatch-size 64 \
     --policy.hidden-size 16 \
     --policy.num-layers 1 \
     --checkpoint-dir /private/tmp/pufferlib-checkpoints \
     --log-dir /private/tmp/pufferlib-logs

Omit ``--torch.device mps`` to use automatic selection, or pass
``--torch.device cpu`` for a numerical/performance baseline.

Building These Docs
-------------------

.. code-block:: bash

   .venv/bin/sphinx-build -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` in your browser.
