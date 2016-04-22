from modular_rl import *
from rl_gym.spaces import Box, Discrete
from collections import OrderedDict
from keras.models import Sequential
from keras.layers.core import Dense

MLP_OPTIONS = [("hid_sizes", comma_sep_ints, [64,64], "Sizes of hidden layers of MLP")]

def make_mlps(ob_space, ac_space, cfg):
    assert isinstance(ob_space, Box)
    hid_sizes = cfg["hid_sizes"]
    if isinstance(ac_space, Box):
        outdim = ac_space.shape[0]
        probtype = DiagGauss(outdim)
    elif isinstance(ac_space, Discrete):
        outdim = ac_space.n
        probtype = Categorical(outdim)
    net = Sequential()
    for (i, layeroutsize) in enumerate(hid_sizes):
        inshp = dict(input_shape=ob_space.shape) if i==0 else {}
        net.add(Dense(layeroutsize, activation="tanh", **inshp))
    # net.add(Dense(64, activation="tanh"))
    if isinstance(ac_space, Box):
        net.add(Dense(outdim))
        Wlast = net.layers[-1].W
        Wlast.set_value(Wlast.get_value(borrow=True)*0.1)
        net.add(ConcatFixedStd())
    else:
        net.add(Dense(outdim, activation="softmax"))
        Wlast = net.layers[-1].W
        Wlast.set_value(Wlast.get_value(borrow=True)*0.1)
    policy = StochPolicyKeras(net, probtype)
    vfnet = Sequential()
    for (i, layeroutsize) in enumerate(hid_sizes):
        inshp = dict(input_shape=ob_space.shape) if i==0 else {}
        vfnet.add(Dense(layeroutsize, activation="tanh", **inshp))
    vfnet.add(Dense(1))
    baseline = NnVf(vfnet, dict(mixfrac=0.1))
    return policy, baseline

def make_deterministic_mlp(ob_space, ac_space, cfg):
    assert isinstance(ob_space, Box)
    hid_sizes = cfg["hid_sizes"]
    if isinstance(ac_space, Box):
        outdim = ac_space.shape[0]
        probtype = DiagGauss(outdim)
    elif isinstance(ac_space, Discrete):
        outdim = ac_space.n
        probtype = Categorical(outdim)
    net = Sequential()
    for (i, layeroutsize) in enumerate(hid_sizes):
        inshp = dict(input_shape=ob_space.shape) if i==0 else {}
        net.add(Dense(layeroutsize, activation="tanh", **inshp))
    inshp = dict(input_shape=ob_space.shape) if len(hid_sizes) == 0 else {}
    net.add(Dense(outdim, **inshp))
    Wlast = net.layers[-1].W
    Wlast.set_value(Wlast.get_value(borrow=True)*0.1)
    policy = StochPolicyKeras(net, probtype)
    return policy

class AgentWithPolicy(object):
    def __init__(self, policy):
        self.policy = policy
        self.stochastic = True
    def set_stochastic(self, stochastic):
        self.stochastic = stochastic
    def act(self, ob_no):
        return self.policy.act(ob_no, stochastic = self.stochastic)
    def get_flat(self):
        return self.policy.get_flat()
    def set_from_flat(self, th):
        return self.policy.set_from_flat(th)

class DeterministicAgent(AgentWithPolicy):
    options = MLP_OPTIONS
    def __init__(self, ob_space, ac_space, usercfg):
        cfg = update_default_config(self.options, usercfg)
        policy = make_deterministic_mlp(ob_space, ac_space, cfg)
        AgentWithPolicy.__init__(self, policy)
        self.set_stochastic(False)

class TrpoAgent(AgentWithPolicy):
    options = MLP_OPTIONS + TrpoUpdater.options
    def __init__(self, ob_space, ac_space, usercfg):
        cfg = update_default_config(self.options, usercfg)
        policy, self.baseline = make_mlps(ob_space, ac_space, cfg)
        self.updater = TrpoUpdater(policy, cfg)
        AgentWithPolicy.__init__(self, policy)
        AgentWithPolicy.__init__(self, policy)

class PpoLbfgsAgent(AgentWithPolicy):
    options = MLP_OPTIONS + PpoLbfgsUpdater.options
    def __init__(self, ob_space, ac_space, usercfg):
        cfg = update_default_config(self.options, usercfg)
        policy, self.baseline = make_mlps(ob_space, ac_space, cfg)
        self.updater = PpoLbfgsUpdater(policy, cfg)
        AgentWithPolicy.__init__(self, policy)

class PpoSgdAgent(AgentWithPolicy):
    options = MLP_OPTIONS + PpoSgdUpdater.options
    def __init__(self, ob_space, ac_space, usercfg):
        cfg = update_default_config(self.options, usercfg)
        policy, self.baseline = make_mlps(ob_space, ac_space, cfg)
        self.updater = PpoSgdUpdater(policy, cfg)
        AgentWithPolicy.__init__(self, policy)
