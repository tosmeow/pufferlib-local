Workflow
========

Command Shape
-------------

PufferLib's CLI expects a mode and environment before the options:

.. code-block:: bash

   puffer train breakout [OPTIONS]
   puffer eval breakout [OPTIONS]
   puffer sweep breakout [OPTIONS]
   puffer match breakout [OPTIONS]

To see the valid options for an environment:

.. code-block:: bash

   .venv/bin/puffer train breakout --help

Local CPU Training
------------------

On this Mac, the practical local path is:

.. code-block:: bash

   puffer build breakout
   .venv/bin/puffer train breakout --slowly

``--slowly`` means: use the compiled CPU vector environment plus the PyTorch
training backend. This is good for debugging, reading, and smoke testing. It is
not the headline high-performance CUDA path.

CUDA/GPU Training
-----------------

For serious throughput, use a Linux/NVIDIA machine and the Docker setup:

.. code-block:: bash

   git clone -b 4.0 https://github.com/PufferAI/PufferTank.git
   cd PufferTank
   ./docker.sh test

Then build the target environment without ``--cpu``:

.. code-block:: bash

   puffer build breakout
   puffer train breakout

Common Commands
---------------

.. code-block:: bash

   .venv/bin/puffer build breakout
   .venv/bin/puffer train breakout --train.learning-rate 0.001
   .venv/bin/puffer train breakout --env.frameskip 3
   .venv/bin/puffer train breakout --vec.num-threads 4
   .venv/bin/puffer eval breakout --load-model-path latest
   .venv/bin/puffer sweep breakout
