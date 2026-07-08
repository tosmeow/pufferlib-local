Configuration
=============

Config Loading
--------------

Configuration lives in ``config/default.ini`` plus one environment-specific
``.ini`` file. ``pufferlib.pufferl.load_config(env_name)`` searches the config
tree for a matching ``[base] env_name``.

CLI option names mirror config sections:

.. code-block:: bash

   --train.learning-rate 0.001
   --env.frameskip 3
   --vec.num-threads 4
   --policy.hidden-size 64

The parser converts underscores to dashes, so ``learning_rate`` becomes
``--train.learning-rate``.

Important Sections
------------------

``[base]``
   Environment name, logging paths, checkpointing, rank/world-size fields.

``[vec]``
   Number of agents, buffers, and CPU threads.

``[env]``
   Environment-specific parameters consumed by ``ocean/<env>/binding.c``.

``[policy]``
   Model size and architecture knobs.

``[torch]``
   Names of model classes in ``pufferlib.models``.

``[train]``
   PPO-like training parameters, horizon, minibatch size, replay ratio, and optimizer settings.

``[sweep]``
   Protein sweep settings and hyperparameter distributions.

Breakout Example
----------------

.. literalinclude:: ../config/breakout.ini
   :language: ini
   :caption: config/breakout.ini
   :lines: 1-45
