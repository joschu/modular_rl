#!/usr/bin/env python
"""
This script runs the cross-entropy method
"""

from rl_gym.envs import make
from modular_rl import *
import argparse, sys, cPickle

from tabulate import tabulate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    update_argument_parser(parser, GENERAL_OPTIONS)    
    parser.add_argument("--env",required=True)
    parser.add_argument("--agent",required=True)
    parser.add_argument("--plot",action="store_true")
    args,_ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])
    env, env_spec = make(args.env)
    env = FilteredEnv(env, ZFilter(env.observation_space.shape, clip=5), ZFilter((), demean=False, clip=10))
    agent_ctor = get_agent_cls(args.agent)
    update_argument_parser(parser, agent_ctor.options)
    update_argument_parser(parser, CEM_OPTIONS)
    args = parser.parse_args()
    cfg = args.__dict__
    agent = agent_ctor(env.observation_space, env.action_space, cfg)
    np.random.seed(args.seed)
    hdf, diagnostics = prepare_h5_file(args)

    if args.max_pathlength == 0: 
        args.max_pathlength = env_spec.timestep_limit

    COUNTER = 0
    def callback(stats):
        global COUNTER
        for (stat,val) in stats.items():
            diagnostics[stat].append(val)
        if args.plot:
            animate_rollout(env, agent, min(500, args.max_pathlength))
        print "*********** Iteration %i ****************" % COUNTER
        print tabulate(filter(lambda (k,v) : np.asarray(v).size==1, stats.items())) #pylint: disable=W0110
        COUNTER += 1
        if args.snapshot_every and ((COUNTER % args.snapshot_every==0) or (COUNTER==args.n_iter)): 
            hdf['/agent_snapshots/%0.4i'%COUNTER] = np.array(cPickle.dumps(agent,-1))
    run_cem_algorithm(env, agent, callback=callback, usercfg = cfg)

    hdf['env_id'] = env_spec.id
    hdf['ob_filter'] = np.array(cPickle.dumps(env.ob_filter, -1))
    try: hdf['env'] = np.array(cPickle.dumps(env, -1))
    except Exception: print "failed to pickle env" #pylint: disable=W0703
