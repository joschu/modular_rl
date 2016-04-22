#!/usr/bin/env python
import argparse, h5py, cPickle
from modular_rl.filtered_env import FilteredEnv
import sys, traceback
from rl_gym import scoreboard

parser = argparse.ArgumentParser()
parser.add_argument("hdf")
parser.add_argument("--maxprob", type=int, default=1)
args = parser.parse_args()

hdf = h5py.File(args.hdf,'r')
origagent = cPickle.loads(hdf['agent_snapshots'].values()[-1].value)
filtenv = cPickle.loads(hdf['env'].value)
assert isinstance(filtenv, FilteredEnv)

snapnames = hdf['agent_snapshots'].keys()
cPickle.loads(hdf['agent_snapshots'].values()[-1].value)
env_id = hdf["env_id"].value
print env_id

agent_name = "TRPO"
if args.maxprob: agent_name += "-maxprob"

class AgentWithFilter(object):
    def __init__(self, agent, obfilt):
        self.agent = agent
        self.obfilt = obfilt
    def act(self, o):
        a, info = self.agent.act(self.obfilt(o))
        if args.maxprob:            
            p = info['prob']
            if a.dtype.kind=='i':
                return p.argmax(), info
            else:
                return p[:len(p)//2].tolist(), info
        else:
            return a, info

agent = AgentWithFilter(origagent, filtenv.ob_filter)

for i in xrange(1):
    try:
        evaluation = scoreboard.evaluate(
            agent_callable=lambda ob_n,_,__ : [agent.act(ob)[0] for ob in ob_n],
            agent_name=agent_name,
            vectorization=100,
            env_id = env_id,
            training_dir = args.hdf + ".mondir"
            )
    except Exception: #pylint: disable=W0703
        print "failed trying again"
        print "**************"
        traceback.print_exc()
        print "**************"
    else:
        sys.exit(0)
sys.exit(1)

