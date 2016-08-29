## About

This repository implements several algorithms:

- Trust Region Policy Optimization [1]
- Proximal Policy Optimization (i.e., TRPO, but using a penalty instead of a constraint on KL divergence), where each subproblem is solved with either SGD or L-BFGS
- Cross Entropy Method

TRPO and PPO are implemented with neural-network value functions and use GAE [2].


This library is written in a modular way to allow for sharing code between TRPO and PPO variants, and to write the same code for different kinds of action spaces.

## Dependencies

- scipy
- numpy
- keras (1.0.1)
- theano (0.8.2)
- gym
- tabulate
- ffmpeg or avconv

## Install

Tested under Ubuntu 14.04 only.

#### Scipy stack (includes numpy)

https://www.scipy.org/install.html

#### Theano Requirements

Note: Do not install Theano through pip in this step.

http://deeplearning.net/software/theano/install.html

#### Gym

https://gym.openai.com/docs

#### ffmpeg or avconv

###### Ubuntu 14.04

```
sudo apt-get install libav-tools
```

###### Other Ubuntu Versions

```
sudo apt-get install ffmpeg
```

###### OS X

```
brew install ffmpeg
```


#### modular_rl

Install the correct versions of theano, keras and tabulate

```
sudo python setup.py develop --user
```

## Run

To run the algorithms implemented here, you should put `modular_rl` on your `PYTHONPATH`, or run the scripts (e.g. `run_pg.py`) from this directory.

Good parameter settings can be found in the `experiments` directory.

You can learn about the various parameters by running one of the experiment scripts with the `-h` flag, but providing the (required) `env` and `agent` parameters. (Those parameters determine what other parameters are available.) For example, to see the parameters of TRPO,

    ./run_pg.py --env CartPole-v0 --agent modular_rl.agentzoo.TrpoAgent -h

To the the parameters of CEM,

    ./run_cem.py --env=Acrobot-v0 --agent=modular_rl.agentzoo.DeterministicAgent  --n_iter=2


[1] JS, S Levine, P Moritz, M Jordan, P Abbeel, "Trust region policy optimization." arXiv preprint arXiv:1502.05477 (2015).

[2] JS, P Moritz, S Levine, M Jordan, P Abbeel, "High-dimensional continuous control using generalized advantage estimation." arXiv preprint arXiv:1506.02438 (2015).

## Troubleshooting

##### Theano warns "We did not found a dynamic library into the library_dir of the library we use for blas."

See [this stackoverflow answer](http://stackoverflow.com/questions/6789368/how-to-make-sure-the-numpy-blas-libraries-are-available-as-dynamically-loadable).

On Ubuntu 14.04 running the following has been known to resolve the issue

```
export THEANO_FLAGS=blas.ldflags="-L/usr/lib/ -lblas"
```

##### Theano fails to compile functions due to permissions issues

Theano seems to prefer being installed in a non-root location. The above setup.py
directions use --user for this purpose. You can run the following if you installed it separately.

```
sudo pip uninstall theano
pip install theano==0.8.2 --user
```