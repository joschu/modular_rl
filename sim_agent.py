#!/usr/bin/env python
import argparse
import cPickle, h5py, numpy as np, time
from collections import defaultdict
import rl_gym

def animate_rollout(env, agent, n_timesteps,delay=.01):
    infos = defaultdict(list)
    ob = env.reset()
    if hasattr(agent,"reset"): agent.reset()
    env.render()
    for i in xrange(n_timesteps):
        a, _info = agent.act(ob)
        (ob, rew, done, info) = env.step(a)
        env.render()
        if done:
            print("terminated after %s timesteps"%i)
            break
        for (k,v) in info.items():
            infos[k].append(v)
        infos['ob'].append(ob)
        infos['reward'].append(rew)
        infos['action'].append(a)
        time.sleep(delay)
    return infos

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf")
    parser.add_argument("--max_pathlength",type=int)
    parser.add_argument("--snapname")
    args = parser.parse_args()

    hdf = h5py.File(args.hdf,'r')

    snapnames = hdf['agent_snapshots'].keys()
    print "snapshots:\n",snapnames
    if args.snapname is None: 
        snapname = snapnames[-1]
    elif args.snapname not in snapnames:
        raise ValueError("Invalid snapshot name %s"%args.snapname)
    else: 
        snapname = args.snapname

    _, spec = rl_gym.make(hdf["env_id"].value)

    agent = cPickle.loads(hdf['agent_snapshots'][snapname].value)
    agent.stochastic=False
    env = cPickle.loads(hdf['env'].value)

    max_pathlength = args.max_pathlength or spec.timestep_limit

    while True:
        infos = animate_rollout(env,agent,n_timesteps=max_pathlength, 
            delay=1.0/env.env.metadata['video.frames_per_second'])
        for (k,v) in infos.items():
            if k.startswith("reward"):
                print "%s: %f"%(k, np.sum(v))
        raw_input("press enter to continue")

if __name__ == "__main__":
    main()