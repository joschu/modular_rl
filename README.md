This repository implements several algorithms:

- Trust Region Policy Optimization [1]
- Proximal Policy Optimization (i.e., TRPO, but using a penalty instead of a constraint on KL divergence), where each subproblem is solved with either SGD or L-BFGS
- Cross Entropy Method

TRPO and PPO are implemented with neural-network value functions and use GAE [2].


This library is written in a modular way to allow for sharing code between TRPO and PPO variants, and to write the same code for different kinds of action spaces.

Dependencies:

- keras (1.0.1)
- theano (0.8.2)
- tabulate
- numpy
- scipy


To run the algorithms implemented here, you should put `modular_rl` on your `PYTHONPATH`, or run the scripts (e.g. `run_pg.py`) from this directory.

Good parameter settings can be found in the `experiments` directory.

You can learn about the various parameters by running one of the experiment scripts with the `-h` flag, but providing the (required) `env` and `agent` parameters. (Those parameters determine what other parameters are available.) For example, to see the parameters of TRPO,

    ./run_pg.py --env CartPole-v0 --agent modular_rl.agentzoo.TrpoAgent -h

To the the parameters of CEM,

    ./run_cem.py --env=Acrobot-v0 --agent=modular_rl.agentzoo.DeterministicAgent  --n_iter=2


[1] JS, S Levine, P Moritz, M Jordan, P Abbeel, "Trust region policy optimization." arXiv preprint arXiv:1502.05477 (2015).

[2] JS, P Moritz, S Levine, M Jordan, P Abbeel, "High-dimensional continuous control using generalized advantage estimation." arXiv preprint arXiv:1506.02438 (2015).

