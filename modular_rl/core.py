import numpy as np, time, itertools, os
from collections import OrderedDict
from .misc_utils import *
from . import parallel_utils, distributions
concat = np.concatenate
import theano.tensor as T, theano
from importlib import import_module
import scipy.optimize
from .keras_theano_setup import floatX
from keras.layers.core import Layer
from tabulate import tabulate

FNOPTS = dict(allow_input_downcast=True, on_unused_input='ignore')        

# ================================================================
# Make agent 
# ================================================================

def get_agent_cls(name):
    p, m = name.rsplit('.', 1)
    mod = import_module(p)
    constructor = getattr(mod, m)
    return constructor


# ================================================================
# Stats 
# ================================================================

def add_episode_stats(stats, paths, state):
    reward_key = "reward_raw" if "reward_raw" in paths[0] else "reward"
    episoderewards = np.array([path[reward_key].sum() for path in paths])
    pathlengths = np.array([pathlength(path) for path in paths])
 
    stats["EpisodeRewards"] = episoderewards
    stats["EpisodeLengths"] = pathlengths
    stats["NumEpBatch"] = len(episoderewards)
    stats["EpRewMean"] = episoderewards.mean()
    stats["EpRewSEM"] = episoderewards.std()/np.sqrt(len(paths))
    stats["EpRewMax"] = episoderewards.max()
    stats["EpLenMean"] = pathlengths.mean()
    stats["EpLenMax"] = pathlengths.max()
    stats["RewPerStep"] = episoderewards.sum()/pathlengths.sum()
    stats["TimeElapsed"] = time.time() - state["tstart"]

def add_prefixed_stats(stats, prefix, d):
    for (k,v) in d.iteritems():
        stats[prefix+"_"+k] = v

# ================================================================
# Policy Gradients 
# ================================================================

def compute_advantage(vf, paths, gamma, lam):
    # Compute return, baseline, advantage
    for path in paths:
        path["return"] = discount(path["reward"], gamma)
        b = path["baseline"] = vf.predict(path)
        b1 = np.append(b, 0 if path["terminated"] else b[-1])
        deltas = path["reward"] + gamma*b1[1:] - b1[:-1] 
        path["advantage"] = discount(deltas, gamma * lam)
    alladv = np.concatenate([path["advantage"] for path in paths])    
    # Standardize advantage
    std = alladv.std()
    mean = alladv.mean()
    for path in paths:
        path["advantage"] = (path["advantage"] - mean) / std



PG_OPTIONS = [
    ("max_pathlength", int, 0, "maximum length of trajectories"),
    ("n_iter", int, 200, "number of batch"),
    ("parallel", int, 0, "collect trajectories in parallel"),
    ("timesteps_per_batch", int, 10000, ""),
    ("gamma", float, 0.99, "discount"),
    ("lam", float, 1.0, "lambda parameter from generalized advantage estimation"),
]

def run_policy_gradient_algorithm(env, policy, pol_updater, vf, usercfg=None, callback=None):
    cfg = update_default_config(PG_OPTIONS, usercfg)
    cfg.update(usercfg)
    print "policy gradient config", cfg

    if cfg["parallel"]:
        raise NotImplementedError

    state = {"tstart" : time.time()}
    for _ in xrange(cfg["n_iter"]):
        # Rollouts ========
        paths = get_paths(env, policy, cfg, state)
        compute_advantage(vf, paths, gamma=cfg["gamma"], lam=cfg["lam"])
        # VF Update ========
        vf_stats = vf.fit(paths)
        # Pol Update ========
        pol_stats = pol_updater(paths)
        # Stats ========
        stats = OrderedDict()
        add_episode_stats(stats, paths, state)
        add_prefixed_stats(stats, "vf", vf_stats)
        add_prefixed_stats(stats, "pol", pol_stats)
        if callback: callback(stats)

# ================================================================
# Cross-entropy method 
# ================================================================

def cem(f,th_mean,batch_size,n_iter,elite_frac, initial_std=1.0, extra_std=0.0, std_decay_time=1.0, pool=None):
    r"""
    Noisy cross-entropy method
    http://dx.doi.org/10.1162/neco.2006.18.12.2936
    http://ie.technion.ac.il/CE/files/papers/Learning%20Tetris%20Using%20the%20Noisy%20Cross-Entropy%20Method.pdf
    Incorporating schedule described on page 4 (also see equation below.)

    Inputs
    ------

    f : function of one argument--the parameter vector
    th_mean : initial distribution is theta ~ Normal(th_mean, initial_std)
    batch_size : how many samples of theta per iteration
    n_iter : how many iterations
    elite_frac : how many samples to select at the end of the iteration, and use for fitting new distribution
    initial_std : standard deviation of initial distribution
    extra_std : "noise" component added to increase standard deviation.
    std_decay_time : how many timesteps it takes for noise to decay

    \sigma_{t+1}^2 =  \sigma_{t,elite}^2 + extra_std * Z_t^2
    where Zt = max(1 - t / std_decay_time, 10 , 0) * extra_std.
    """
    n_elite = int(np.round(batch_size*elite_frac))

    th_std = np.ones(th_mean.size)*initial_std

    for iteration in xrange(n_iter):

        extra_var_multiplier = max((1.0-iteration/std_decay_time),0) # Multiply "extra variance" by this factor
        sample_std = np.sqrt(th_std + np.square(extra_std) * extra_var_multiplier)

        ths = np.array([th_mean + dth for dth in  sample_std[None,:]*np.random.randn(batch_size, th_mean.size)])
        if pool is None:
            ys = np.array([f(th) for th in ths])
        else:
            ys = np.array(pool.map(f, ths))
        assert ys.ndim==1
        elite_inds = ys.argsort()[-n_elite:]
        elite_ths = ths[elite_inds]
        th_mean = elite_ths.mean(axis=0)
        th_std = elite_ths.var(axis=0)
        yield {"ys":ys,"th":th_mean,"ymean":ys.mean(), "std" : sample_std}


CEM_OPTIONS = [
    ("batch_size", int, 200, "Number of episodes per batch"),
    ("n_iter", int, 200, "Number of iterations"),
    ("elite_frac", float, 0.2, "fraction of parameter settings used to fit pop"),
    ("initial_std", float, 1.0, "initial standard deviation for parameters"),
    ("extra_std", float, 0.0, "extra stdev added"),
    ("std_decay_time", float, -1.0, "number of timesteps that extra stdev decays over. negative => n_iter/2"),
    ("max_pathlength", int, 0, "maximum length of trajectories"),
    ("parallel", int, 0, "collect trajectories in parallel"),
]

def _cem_f(th):
    G = parallel_utils.G
    G.agent.set_from_flat(th)
    path = rollout(G.env, G.agent, G.max_pathlength)
    return path["reward"].sum()

def _seed_with_pid(_G):
    np.random.seed(os.getpid())

def run_cem_algorithm(env, agent, usercfg=None, callback=None):
    cfg = update_default_config(CEM_OPTIONS, usercfg)
    if cfg["std_decay_time"] < 0: cfg["std_decay_time"] = cfg["n_iter"] / 2 
    cfg.update(usercfg)
    print "cem config", cfg

    G = parallel_utils.G
    G.env = env
    G.agent = agent
    G.max_pathlength = cfg["max_pathlength"]
    if cfg["parallel"]:
        parallel_utils.init_pool()
        parallel_utils.apply_each(_seed_with_pid)
        pool = G.pool
    else:
        pool = None

    th_mean = agent.get_flat()

    for info in cem(_cem_f, th_mean, cfg["batch_size"], cfg["n_iter"], cfg["elite_frac"], 
        cfg["initial_std"], cfg["extra_std"], cfg["std_decay_time"], pool=pool):
        callback(info)
        ps = np.linspace(0,100,5)
        print tabulate([ps, np.percentile(info["ys"].ravel(),ps), np.percentile(info["std"].ravel(),ps)])

        agent.set_from_flat(info["th"])


def get_paths(env, agent, cfg, state):
    if "seed_iter" not in state:
        state["seed_iter"] = itertools.count()
    if cfg["parallel"]:
        raise NotImplementedError
    else:
        paths = do_rollouts_serial(env, agent, cfg["max_pathlength"], cfg["timesteps_per_batch"], state["seed_iter"])
    return paths

# ================================================================
# Stochastic policies 
# ================================================================

class StochPolicy(object):
    @property
    def probtype(self):
        raise NotImplementedError
    @property
    def trainable_variables(self):
        raise NotImplementedError
    @property
    def input(self):
        raise NotImplementedError
    def get_output(self, train):
        raise NotImplementedError
    def act(self, ob, stochastic=True):
        prob = self._act_prob(ob[None])
        if stochastic:
            return self.probtype.sample(prob)[0], {"prob" : prob[0]}
        else:
            return self.probtype.maxprob(prob)[0], {"prob" : prob[0]}
    def finalize(self):
        self._act_prob = theano.function([self.input], self.get_output(train=False), **FNOPTS)


class ProbType(object):
    def sampled_variable(self):
        raise NotImplementedError
    def prob_variable(self):
        raise NotImplementedError
    def likelihood(self, a, prob):
        raise NotImplementedError
    def loglikelihood(self, a, prob):
        raise NotImplementedError
    def kl(self, prob0, prob1):
        raise NotImplementedError
    def entropy(self, prob):
        raise NotImplementedError
    def maxprob(self, prob):
        raise NotImplementedError

class StochPolicyKeras(StochPolicy, EzPickle):
    def __init__(self, net, probtype):
        EzPickle.__init__(self, net, probtype)
        self._net = net
        self._probtype = probtype
        self.finalize()
    @property
    def probtype(self):
        return self._probtype
    @property
    def net(self):
        return self._net    
    @property
    def trainable_variables(self):
        return self._net.trainable_weights
    @property
    def variables(self):
        return self._net.get_params()[0]
    @property
    def input(self):
        return self._net.input
    def get_output(self, train):
        return self._net.get_output(train=train)
    def get_updates(self):
        self._net.get_output(train=True)
        return self._net.updates
    def get_flat(self):
        return flatten(self.net.get_weights())
    def set_from_flat(self, th):
        weights = self.net.get_weights()
        self._weight_shapes = [weight.shape for weight in weights]
        self.net.set_weights(unflatten(th, self._weight_shapes))

class Categorical(ProbType):
    def __init__(self, n):
        self.n = n
    def sampled_variable(self):
        return T.ivector('a')
    def prob_variable(self):
        return T.matrix('prob')
    def likelihood(self, a, prob):
        return prob[T.arange(prob.shape[0]), a]
    def loglikelihood(self, a, prob):
        return T.log(self.likelihood(a, prob))
    def kl(self, prob0, prob1):
        return (prob0 * T.log(prob0/prob1)).sum(axis=1)
    def entropy(self, prob0):
        return - (prob0 * T.log(prob0)).sum(axis=1)
    def sample(self, prob):
        return distributions.categorical_sample(prob)
    def maxprob(self, prob):
        return prob.argmax(axis=1)

class CategoricalOneHot(ProbType):
    def __init__(self, n):
        self.n = n
    def sampled_variable(self):
        return T.matrix('a')
    def prob_variable(self):
        return T.matrix('prob')
    def likelihood(self, a, prob):
        return (a * prob).sum(axis=1)
    def loglikelihood(self, a, prob):
        return T.log(self.likelihood(a, prob))
    def kl(self, prob0, prob1):
        return (prob0 * T.log(prob0/prob1)).sum(axis=1)
    def entropy(self, prob0):
        return - (prob0 * T.log(prob0)).sum(axis=1)
    def sample(self, prob):
        assert prob.ndim == 2
        inds = distributions.categorical_sample(prob)
        out = np.zeros_like(prob)
        out[np.arange(prob.shape[0]), inds] = 1
        return out
    def maxprob(self, prob):
        out = np.zeros_like(prob)
        out[prob.argmax(axis=1)] = 1

class DiagGauss(ProbType):
    def __init__(self, d):
        self.d = d
    def sampled_variable(self):
        return T.matrix('a')
    def prob_variable(self):
        return T.matrix('prob')
    def loglikelihood(self, a, prob):
        mean0 = prob[:,:self.d]
        std0 = prob[:, self.d:]
        # exp[ -(a - mu)^2/(2*sigma^2) ] / sqrt(2*pi*sigma^2)
        return - 0.5 * T.square((a - mean0) / std0).sum(axis=1) - 0.5 * T.log(2.0 * np.pi) * self.d - T.log(std0).sum(axis=1)
    def likelihood(self, a, prob):
        return T.exp(self.loglikelihood(a, prob))
    def kl(self, prob0, prob1):
        mean0 = prob0[:, :self.d]
        std0 = prob0[:, self.d:]
        mean1 = prob1[:, :self.d]
        std1 = prob1[:, self.d:]
        return T.log(std1 / std0).sum(axis=1) + ((T.square(std0) + T.square(mean0 - mean1)) / (2.0 * T.square(std1))).sum(axis=1) - 0.5 * self.d
    def entropy(self, prob):
        std_nd = prob[:, self.d:]
        return T.log(std_nd).sum(axis=1) + .5 * np.log(2 * np.pi * np.e) * self.d
    def sample(self, prob):
        mean_nd = prob[:, :self.d] 
        std_nd = prob[:, self.d:]
        return np.random.randn(prob.shape[0], self.d).astype(floatX) * std_nd + mean_nd
    def maxprob(self, prob):
        return prob[:, :self.d]

def test_probtypes():
    theano.config.floatX = 'float64'
    np.random.seed(0)

    prob_diag_gauss = np.array([-.2, .3, .4, -.5, 1.1, 1.5, .1, 1.9])
    diag_gauss = DiagGauss(prob_diag_gauss.size // 2)
    yield validate_probtype, diag_gauss, prob_diag_gauss

    prob_categorical = np.array([.2, .3, .5])
    categorical = Categorical(prob_categorical.size)
    yield validate_probtype, categorical, prob_categorical


def validate_probtype(probtype, prob):
    N = 100000
    # Check to see if mean negative log likelihood == differential entropy
    Mval = np.repeat(prob[None, :], N, axis=0)
    M = probtype.prob_variable()
    X = probtype.sampled_variable()
    calcloglik = theano.function([X, M], T.log(probtype.likelihood(X, M)), allow_input_downcast=True)
    calcent = theano.function([M], probtype.entropy(M), allow_input_downcast=True)
    Xval = probtype.sample(Mval)
    logliks = calcloglik(Xval, Mval)
    entval_ll = - logliks.mean()
    entval_ll_stderr = logliks.std() / np.sqrt(N)
    entval = calcent(Mval).mean()
    print entval, entval_ll, entval_ll_stderr
    assert np.abs(entval - entval_ll) < 3 * entval_ll_stderr # within 3 sigmas

    # Check to see if kldiv[p,q] = - ent[p] - E_p[log q]
    M2 = probtype.prob_variable()
    q = prob + np.random.randn(prob.size) * 0.1
    Mval2 = np.repeat(q[None, :], N, axis=0)
    calckl = theano.function([M, M2], probtype.kl(M, M2), allow_input_downcast=True)
    klval = calckl(Mval, Mval2).mean()
    logliks = calcloglik(Xval, Mval2)
    klval_ll = - entval - logliks.mean()
    klval_ll_stderr = logliks.std() / np.sqrt(N)
    print klval, klval_ll,  klval_ll_stderr
    assert np.abs(klval - klval_ll) < 3 * klval_ll_stderr # within 3 sigmas



# ================================================================
# Value functions 
# ================================================================

class Baseline(object):
    def fit(self, paths):
        raise NotImplementedError
    def predict(self, path):
        raise NotImplementedError

class TimeDependentBaseline(Baseline):
    def __init__(self):
        self.baseline = None
    def fit(self, paths):
        rets = [path["return"] for path in paths]
        maxlen = max(len(ret) for ret in rets)
        retsum = np.zeros(maxlen)
        retcount = np.zeros(maxlen)
        for ret in rets:
            retsum[:len(ret)] += ret
            retcount[:len(ret)] += 1
        retmean = retsum / retcount
        i_depletion = np.searchsorted(-retcount, -4)
        self.baseline = retmean[:i_depletion]
        pred = concat([self.predict(path) for path in paths])
        return {"EV" : explained_variance(pred, concat(rets))}
    def predict(self, path):
        if self.baseline is None:
            return np.zeros(pathlength(path))
        else:
            lenpath = pathlength(path)
            lenbase = len(self.baseline)
            if lenpath > lenbase:
                return concat([self.baseline, self.baseline[-1] + np.zeros(lenpath-lenbase)])
            else:
                return self.baseline[:lenpath]

class NnRegression(EzPickle):
    def __init__(self, net, mixfrac=1.0, maxiter=25):
        EzPickle.__init__(self, net, mixfrac, maxiter)
        self.net = net
        self.mixfrac = mixfrac

        x_nx = net.get_input()
        self.predict = theano.function([x_nx], net.get_output(train=False), **FNOPTS)

        ypred_ny = net.get_output(train=True)
        ytarg_ny = T.matrix("ytarg")
        var_list = net.trainable_weights
        l2 = 1e-3 * T.add(*[T.square(v).sum() for v in var_list])
        N = x_nx.shape[0]
        mse = T.sum(T.square(ytarg_ny - ypred_ny))/N
        symb_args = [x_nx, ytarg_ny]
        loss = mse + l2
        self.opt = LbfgsOptimizer(loss, var_list, symb_args, maxiter=maxiter, extra_losses={"mse":mse, "l2":l2})

    def fit(self, x_nx, ytarg_ny):
        nY = ytarg_ny.shape[1]
        ypredold_ny = self.predict(x_nx)
        out = self.opt.update(x_nx, ytarg_ny*self.mixfrac + ypredold_ny*(1-self.mixfrac))
        yprednew_ny = self.predict(x_nx)
        out["PredStdevBefore"] = ypredold_ny.std()
        out["PredStdevAfter"] = yprednew_ny.std()
        out["TargStdev"] = ytarg_ny.std()
        if nY==1: 
            out["EV_before"] =  explained_variance_2d(ypredold_ny, ytarg_ny)[0]
            out["EV_after"] =  explained_variance_2d(yprednew_ny, ytarg_ny)[0]
        else:
            out["EV_avg"] = explained_variance(yprednew_ny.ravel(), ytarg_ny.ravel())
        return out


class NnVf(object):
    def __init__(self, net, regression_params):
        self.reg = NnRegression(net, **regression_params)
    def predict(self, path):
        return self.reg.predict(path["observation"])[:,0]
    def fit(self, paths):
        ob_no = concat([path["observation"] for path in paths])
        vtarg_n1 = concat([path["return"] for path in paths]).reshape(-1,1)
        return self.reg.fit(ob_no, vtarg_n1)

class NnCpd(EzPickle):
    def __init__(self, net, probtype, maxiter=25):
        EzPickle.__init__(self, net, probtype, maxiter)
        self.net = net

        x_nx = net.get_input()

        prob = net.get_output(train=True)
        a = probtype.sampled_variable()
        var_list = net.trainable_weights

        loglik = probtype.loglikelihood(a, prob)

        self.loglikelihood = theano.function([a, x_nx], loglik, **FNOPTS)
        loss = - loglik.mean()
        symb_args = [x_nx, a]
        self.opt = LbfgsOptimizer(loss, var_list, symb_args, maxiter=maxiter)

    def fit(self, x_nx, a):
        return self.opt.update(x_nx, a)

class SetFromFlat(object):
    def __init__(self, var_list):
        
        theta = T.vector()
        start = 0
        updates = []
        for v in var_list:
            shape = v.shape
            size = T.prod(shape)
            updates.append((v, theta[start:start+size].reshape(shape)))
            start += size
        self.op = theano.function([theta],[], updates=updates,**FNOPTS)
    def __call__(self, theta):
        self.op(theta.astype(floatX))

class GetFlat(object):
    def __init__(self, var_list):
        self.op = theano.function([], T.concatenate([v.flatten() for v in var_list]),**FNOPTS)
    def __call__(self):
        return self.op() #pylint: disable=E1101

class EzFlat(object):
    def __init__(self, var_list):
        self.gf = GetFlat(var_list)
        self.sff = SetFromFlat(var_list)
    def set_params_flat(self, theta):
        self.sff(theta)
    def get_params_flat(self):
        return self.gf()

class LbfgsOptimizer(EzFlat):
    def __init__(self, loss,  params, symb_args, extra_losses=None, maxiter=25):
        EzFlat.__init__(self, params)
        self.all_losses = OrderedDict()
        self.all_losses["loss"] = loss        
        if extra_losses is not None:
            self.all_losses.update(extra_losses)
        self.f_lossgrad = theano.function(list(symb_args), [loss, flatgrad(loss, params)],**FNOPTS)
        self.f_losses = theano.function(symb_args, self.all_losses.values(),**FNOPTS)
        self.maxiter=maxiter

    def update(self, *args):
        thprev = self.get_params_flat()
        def lossandgrad(th):
            self.set_params_flat(th)
            l,g = self.f_lossgrad(*args)
            g = g.astype('float64')
            return (l,g)
        losses_before = self.f_losses(*args)
        theta, _, opt_info = scipy.optimize.fmin_l_bfgs_b(lossandgrad, thprev, maxiter=self.maxiter)
        del opt_info['grad']
        iprint(opt_info)
        self.set_params_flat(theta)
        losses_after = self.f_losses(*args)
        info = OrderedDict()
        for (name,lossbefore, lossafter) in zip(self.all_losses.keys(), losses_before, losses_after):
            info[name+"_before"] = lossbefore
            info[name+"_after"] = lossafter        
        return info

def numel(x):
    return T.prod(x.shape)

def flatgrad(loss, var_list):
    grads = T.grad(loss, var_list)
    return T.concatenate([g.flatten() for g in grads])


# ================================================================
# TRPO 
# ================================================================

class TrpoUpdater(EzFlat, EzPickle):
    
    options = [
        ("cg_damping", float, 1e-3, ""),
        ("max_kl", float, 1e-2, ""),
    ]

    def __init__(self, stochpol, usercfg):
        EzPickle.__init__(self, stochpol, usercfg)
        cfg = update_default_config(self.options, usercfg)
        print "TrpoUpdater", cfg

        self.stochpol = stochpol
        self.cfg = cfg

        probtype = stochpol.probtype
        params = stochpol.trainable_variables
        EzFlat.__init__(self, params)

        ob_no = stochpol.input
        act_na = probtype.sampled_variable()
        adv_n = T.vector("adv_n")

        # Probability distribution:
        prob_np = stochpol.get_output(train=True)
        oldprob_np = probtype.prob_variable()

        logp_n = probtype.loglikelihood(act_na, prob_np)
        oldlogp_n = probtype.loglikelihood(act_na, oldprob_np)
        N = ob_no.shape[0]

        # Policy gradient:
        surr = (-1.0 / N) * T.exp(logp_n - oldlogp_n).dot(adv_n)
        pg = flatgrad(surr, params)

        prob_np_fixed = theano.gradient.disconnected_grad(prob_np)
        kl_firstfixed = probtype.kl(prob_np_fixed, prob_np).sum()/N
        grads = T.grad(kl_firstfixed, params)
        flat_tangent = T.fvector(name="flat_tan")
        shapes = [var.get_value(borrow=True).shape for var in params]
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            tangents.append(T.reshape(flat_tangent[start:start+size], shape))
            start += size
        gvp = T.add(*[T.sum(g*tangent) for (g, tangent) in zipsame(grads, tangents)]) #pylint: disable=E1111
        # Fisher-vector product
        fvp = flatgrad(gvp, params)

        ent = probtype.entropy(prob_np).mean()
        kl = probtype.kl(oldprob_np, prob_np).mean()

        losses = [surr, kl, ent]
        self.loss_names = ["surr", "kl", "ent"]

        args = [ob_no, act_na, adv_n, oldprob_np]

        self.compute_policy_gradient = theano.function(args, pg, **FNOPTS)
        self.compute_losses = theano.function(args, losses, **FNOPTS)
        self.compute_fisher_vector_product = theano.function([flat_tangent] + args, fvp, **FNOPTS)

    def __call__(self, paths):
        cfg = self.cfg
        prob_np = concat([path["prob"] for path in paths])
        ob_no = concat([path["observation"] for path in paths])
        action_na = concat([path["action"] for path in paths])
        advantage_n = concat([path["advantage"] for path in paths])
        args = (ob_no, action_na, advantage_n, prob_np)

        thprev = self.get_params_flat()
        def fisher_vector_product(p):
            return self.compute_fisher_vector_product(p, *args)+cfg["cg_damping"]*p #pylint: disable=E1101,W0640
        g = self.compute_policy_gradient(*args)
        losses_before = self.compute_losses(*args)
        if np.allclose(g, 0):
            print "got zero gradient. not updating"
        else:
            stepdir = cg(fisher_vector_product, -g)
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / cfg["max_kl"])
            print "lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g)
            fullstep = stepdir / lm
            neggdotstepdir = -g.dot(stepdir)
            def loss(th):
                self.set_params_flat(th)
                return self.compute_losses(*args)[0] #pylint: disable=W0640
            success, theta = linesearch(loss, thprev, fullstep, neggdotstepdir/lm)
            print "success", success
            theta = thprev + fullstep
            self.set_params_flat(theta)
        losses_after = self.compute_losses(*args)

        out = OrderedDict()
        for (lname, lbefore, lafter) in zipsame(self.loss_names, losses_before, losses_after):
            out[lname+"_before"] = lbefore
            out[lname+"_after"] = lafter
        return out

def linesearch(f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
    """
    Backtracking linesearch, where expected_improve_rate is the slope dy/dx at the initial point
    """
    fval = f(x)
    print "fval before", fval
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac*fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate*stepfrac
        ratio = actual_improve/expected_improve
        print "a/e/r", actual_improve, expected_improve, ratio
        if ratio > accept_ratio and actual_improve > 0:
            print "fval after", newfval
            return True, xnew
    return False, x

def cg(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    Demmel p 312
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    fmtstr =  "%10i %10.3g %10.3g"
    titlestr =  "%10s %10s %10s"
    if verbose: print titlestr % ("iter", "residual norm", "soln norm")

    for i in xrange(cg_iters):
        if callback is not None:
            callback(x)
        if verbose: print fmtstr % (i, rdotr, np.linalg.norm(x))
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v*p
        r -= v*z
        newrdotr = r.dot(r)
        mu = newrdotr/rdotr
        p = r + mu*p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)
    if verbose: print fmtstr % (i+1, rdotr, np.linalg.norm(x))  # pylint: disable=W0631
    return x

# ================================================================
# PPO 
# ================================================================


class PpoLbfgsUpdater(EzFlat, EzPickle):

    options = [
        ("kl_target", float, 1e-2, ""),
        ("maxiter", int, 25, ""),
        ("reverse_kl", int, 0, "klnewold instead of kloldnew"),
        ("do_split", int, 0, "do train/test split")        
    ]

    def __init__(self, stochpol, usercfg):
        EzPickle.__init__(self, stochpol, usercfg)
        cfg = update_default_config(self.options, usercfg)
        print "PPOUpdater", cfg

        self.stochpol = stochpol
        self.cfg = cfg
        self.kl_coeff = 1.0
        kl_cutoff = cfg["kl_target"]*2.0

        probtype = stochpol.probtype
        params = stochpol.trainable_variables
        EzFlat.__init__(self, params)

        ob_no = stochpol.input
        act_na = probtype.sampled_variable()
        adv_n = T.vector("adv_n")
        kl_coeff = T.scalar("kl_coeff")

        # Probability distribution:
        prob_np = stochpol.get_output(train=True)
        oldprob_np = probtype.prob_variable()

        p_n = probtype.likelihood(act_na, prob_np)
        oldp_n = probtype.likelihood(act_na, oldprob_np)
        N = ob_no.shape[0]

        ent = probtype.entropy(prob_np).mean()
        if cfg["reverse_kl"]:
            kl = probtype.kl(prob_np, oldprob_np).mean()
        else:
            kl = probtype.kl(oldprob_np, prob_np).mean()


        # Policy gradient:
        surr = (-1.0 / N) * (p_n / oldp_n).dot(adv_n)
        pensurr = surr + kl_coeff*kl + 1000*(kl>kl_cutoff)*T.square(kl-kl_cutoff)
        g = flatgrad(pensurr, params)

        losses = [surr, kl, ent]
        self.loss_names = ["surr", "kl", "ent"]

        args = [ob_no, act_na, adv_n, oldprob_np]

        self.compute_lossgrad = theano.function([kl_coeff] + args, [pensurr, g], **FNOPTS)
        self.compute_losses = theano.function(args, losses, **FNOPTS)

    def __call__(self, paths):
        cfg = self.cfg
        prob_np = concat([path["prob"] for path in paths])
        ob_no = concat([path["observation"] for path in paths])
        action_na = concat([path["action"] for path in paths])
        advantage_n = concat([path["advantage"] for path in paths])

        N = ob_no.shape[0]
        train_stop = int(0.75 * N) if cfg["do_split"] else N
        train_sli = slice(0, train_stop)
        test_sli = slice(train_stop, None)

        train_args = (ob_no[train_sli], action_na[train_sli], advantage_n[train_sli], prob_np[train_sli])

        thprev = self.get_params_flat()
        def lossandgrad(th):
            self.set_params_flat(th)
            l,g = self.compute_lossgrad(self.kl_coeff, *train_args)
            g = g.astype('float64')
            return (l,g)

        train_losses_before = self.compute_losses(*train_args)
        if cfg["do_split"]: 
            test_args = (ob_no[test_sli], action_na[test_sli], advantage_n[test_sli], prob_np[test_sli])
            test_losses_before = self.compute_losses(*test_args)

        theta, _, opt_info = scipy.optimize.fmin_l_bfgs_b(lossandgrad, thprev, maxiter=cfg["maxiter"])
        del opt_info['grad']
        print opt_info
        self.set_params_flat(theta)
        train_losses_after = self.compute_losses(*train_args)
        if cfg["do_split"]: 
            test_losses_after = self.compute_losses(*test_args)        
        klafter = train_losses_after[self.loss_names.index("kl")]
        if klafter > 1.3*self.cfg["kl_target"]: 
            self.kl_coeff *= 1.5
            print "Got KL=%.3f (target %.3f). Increasing penalty coeff => %.3f."%(klafter, self.cfg["kl_target"], self.kl_coeff)
        elif klafter < 0.7*self.cfg["kl_target"]: 
            self.kl_coeff /= 1.5
            print "Got KL=%.3f (target %.3f). Decreasing penalty coeff => %.3f."%(klafter, self.cfg["kl_target"], self.kl_coeff)
        else:
            print "KL=%.3f is close enough to target %.3f."%(klafter, self.cfg["kl_target"])
        info = OrderedDict()
        for (name,lossbefore, lossafter) in zipsame(self.loss_names, train_losses_before, train_losses_after):
            info[name+"_before"] = lossbefore
            info[name+"_after"] = lossafter
            info[name+"_change"] = lossafter - lossbefore
        if cfg["do_split"]:
            for (name,lossbefore, lossafter) in zipsame(self.loss_names, test_losses_before, test_losses_after):
                info["test_"+name+"_before"] = lossbefore
                info["test_"+name+"_after"] = lossafter
                info["test_"+name+"_change"] = lossafter - lossbefore

        return info     


class PpoSgdUpdater(EzPickle):

    options = [
        ("kl_target", float, 1e-2, ""),
        ("epochs", int, 10, ""),
        ("stepsize", float, 1e-3, ""),
        ("do_split", int, 0, "do train/test split")
    ]

    def __init__(self, stochpol, usercfg):
        EzPickle.__init__(self, stochpol, usercfg)
        cfg = update_default_config(self.options, usercfg)
        print "PPOUpdater", cfg

        self.stochpol = stochpol
        self.cfg = cfg
        self.kl_coeff = 1.0
        kl_cutoff = cfg["kl_target"]*2.0

        probtype = stochpol.probtype
        params = stochpol.trainable_variables
        old_params = [theano.shared(v.get_value()) for v in stochpol.trainable_variables]

        ob_no = stochpol.input
        act_na = probtype.sampled_variable()
        adv_n = T.vector("adv_n")
        kl_coeff = T.scalar("kl_coeff")

        # Probability distribution:
        self.loss_names = ["surr", "kl", "ent"]

        def compute_losses(train):
            prob_np = stochpol.get_output(train=train) 
            oldprob_np = theano.clone(stochpol.get_output(train=train), replace=dict(zipsame(params, old_params)))
            p_n = probtype.likelihood(act_na, prob_np)
            oldp_n = probtype.likelihood(act_na, oldprob_np)
            N = ob_no.shape[0]
            ent = probtype.entropy(prob_np).mean()
            kl = probtype.kl(oldprob_np, prob_np).mean()
            # Policy gradient:
            surr = (-1.0 / N) * (p_n / oldp_n).dot(adv_n)
            losses = [surr, kl, ent] 
            return losses


        # training
        args = [ob_no, act_na, adv_n]
        train_losses = compute_losses(True)
        surr,kl = train_losses[:2]
        pensurr = surr + kl_coeff*kl + 1000*(kl>kl_cutoff)*T.square(kl-kl_cutoff)
        self.train = theano.function([kl_coeff]+args, train_losses, 
            updates=stochpol.get_updates()
            + adam_updates(pensurr, params, learning_rate=cfg.stepsize).items(), **FNOPTS)

        test_losses = compute_losses(False)

        self.test = theano.function(args, test_losses, **FNOPTS)
        self.update_old_net = theano.function([], [], updates = zip(old_params, params))

    def __call__(self, paths):
        cfg = self.cfg
        ob_no = concat([path["observation"] for path in paths])
        action_na = concat([path["action"] for path in paths])
        advantage_n = concat([path["advantage"] for path in paths])
        args = (ob_no, action_na, advantage_n)

        N = args[0].shape[0]
        batchsize = 128

        self.update_old_net()

        if cfg["do_split"]:
            train_stop = (int(.75*N)//batchsize) * batchsize
            test_losses_before = self.test(*[arg[train_stop:] for arg in args])

            print fmt_row(13, ["epoch"] 
                + self.loss_names
                + ["test_" + name for name in self.loss_names])
        else:
            train_stop = N
            print fmt_row(13, ["epoch"] 
                + self.loss_names)
        train_losses_before = self.test(*[arg[:train_stop] for arg in args])

        for iepoch in xrange(cfg["epochs"]):
            sortinds = np.random.permutation(train_stop)

            losses = []
            for istart in xrange(0, train_stop, batchsize):
                losses.append(  self.train(self.kl_coeff, *[arg[sortinds[istart:istart+batchsize]] for arg in args])  )
            train_losses = np.mean(losses, axis=0)
            if cfg.do_split:
                test_losses = self.test(*[arg[train_stop:] for arg in args])
                print fmt_row(13, np.concatenate([[iepoch], train_losses, test_losses]))
            else:
                print fmt_row(13, np.concatenate([[iepoch], train_losses]))
      

        klafter = train_losses[self.loss_names.index('kl')]
        if klafter > 1.3*self.cfg["kl_target"]: 
            self.kl_coeff *= 1.5
            print "Got KL=%.3f (target %.3f). Increasing penalty coeff => %.3f."%(klafter, self.cfg["kl_target"], self.kl_coeff)
        elif klafter < 0.7*self.cfg["kl_target"]: 
            self.kl_coeff /= 1.5
            print "Got KL=%.3f (target %.3f). Decreasing penalty coeff => %.3f."%(klafter, self.cfg["kl_target"], self.kl_coeff)
        else:
            print "KL=%.3f is close enough to target %.3f."%(klafter, self.cfg["kl_target"])

        info = {}
        for (name,lossbefore, lossafter) in zipsame(self.loss_names, train_losses_before, train_losses):
            info[name+"_before"] = lossbefore
            info[name+"_after"] = lossafter
            info[name+"_change"] = lossafter - lossbefore
        if cfg["do_split"]:
            for (name,lossbefore, lossafter) in zipsame(self.loss_names, test_losses_before, test_losses):
                info["test_"+name+"_before"] = lossbefore
                info["test_"+name+"_after"] = lossafter
                info["test_"+name+"_change"] = lossafter - lossbefore

        return info     


def adam_updates(loss, params, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8):

    all_grads = T.grad(loss, params)
    t_prev = theano.shared(np.array(0,dtype=floatX))
    updates = OrderedDict()

    t = t_prev + 1
    a_t = learning_rate*T.sqrt(1-beta2**t)/(1-beta1**t)

    for param, g_t in zip(params, all_grads):
        value = param.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

        m_t = beta1*m_prev + (1-beta1)*g_t
        v_t = beta2*v_prev + (1-beta2)*g_t**2
        step = a_t*m_t/(T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    updates[t_prev] = t
    return updates

# ================================================================
# Keras 
# ================================================================

class ConcatFixedStd(Layer):

    input_ndim = 2

    def __init__(self, **kwargs):
        Layer.__init__(self, **kwargs)

    def build(self):
        input_dim = self.input_shape[1]
        self.logstd = theano.shared(np.zeros(input_dim,floatX), name='{}_logstd'.format(self.name))
        self.trainable_weights = [self.logstd]

    @property
    def output_shape(self):
        return (self.input_shape[0], self.input_shape[1] * 2)

    def get_output(self, train=False):
        Mean = self.get_input(train)
        Std = T.repeat(T.exp(self.logstd)[None, :], Mean.shape[0], axis=0)
        return T.concatenate([Mean, Std], axis=1)

    def get_config(self):
        config = {'name': self.__class__.__name__}       
        base_config = Layer.get_config(self)
        return dict(list(base_config.items()) + list(config.items()))
