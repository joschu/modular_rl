#!/bin/bash
# Produce a coverage report
set -eux

rm -rf htmlcov .coverage
coverage run run_cem.py --snapshot_every=20 --max_pathlength=0 --env=Acrobot  --hid_sizes=5,5 --seed=2 --agent=modular_rl.agentzoo.DeterministicAgent  --n_iter=2
coverage run -a run_pg.py --snapshot_every=20 --max_pathlength=0 --n_iter=1 --cg_damping=0.1 --timesteps_per_batch=5 --lam=0.97 --agent=modular_rl.agentzoo.TrpoAgent --max_kl=0.01 --env=Pendulum --gamma=0.98   --hid_sizes=10,5
coverage html && open htmlcov/modular*.html