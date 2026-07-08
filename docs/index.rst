PufferLib Local Docs
====================

This is a local, readable documentation layer for the PufferLib checkout in this
repository. It is meant to help you work from source: build a backend, find the
important modules, understand the CLI, and inspect the Python API.

The upstream public docs live at https://puffer.ai/docs.html. These local docs
are intentionally more navigational and developer-oriented.

Quick Start
-----------

.. code-block:: bash

   source .venv/bin/activate
   puffer build breakout
   .venv/bin/puffer train breakout --slowly

On Apple Silicon, use ``--slowly`` for local smoke tests. The CUDA-native path is
for Linux/NVIDIA machines, ideally via PufferTank.

Contents
--------

.. toctree::
   :maxdepth: 2

   local_setup
   workflow
   architecture
   configuration
   examples
   api/index

Useful Links
------------

* Upstream source: https://github.com/PufferAI/PufferLib/tree/4.0
* Upstream docs: https://puffer.ai/docs.html
* Docker/GPU environment: https://github.com/PufferAI/PufferTank
* Support Discord: https://discord.gg/puffer
