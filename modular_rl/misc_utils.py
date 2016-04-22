from __future__ import print_function
import h5py, atexit, numpy as np, os.path as osp, cPickle, time, scipy, sys, os, subprocess, urllib
from collections import defaultdict

# ================================================================
# Math utilities
# ================================================================

def discount(x, gamma):
    """
    computes discounted sums along 0th dimension of x.

    inputs
    ------
    x: ndarray
    gamma: float

    outputs
    -------
    y: ndarray with same shape as x, satisfying

        y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                where k = len(x) - t - 1

    """
    assert x.ndim >= 1
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

def explained_variance(ypred,y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def explained_variance_2d(ypred, y):
    assert y.ndim == 2 and ypred.ndim == 2
    vary = np.var(y, axis=0)
    out = 1 - np.var(y-ypred)/vary
    out[vary < 1e-10] = 0
    return out

def normal_kl(pdist, other_pdist):
    """
    Computes the KL divergence between two multivariate normal distributions.
    Expects pdist and other_pdist to be numpy matrices. Each row includes the
    mean and standard deviation concatenated together for a single sample.
    Returns a single KL divergence computed for each sample.
    """
    assert pdist.shape[1] % 2 == 0
    dim = pdist.shape[1] / 2
    mean, std = pdist[:, :dim], pdist[:, dim:]
    other_mean, other_std = other_pdist[:, :dim], other_pdist[:, dim:]
    numerator = np.square(mean - other_mean) + \
        np.square(std) - np.square(other_std)
    denominator = 2 * np.square(other_std) + 1e-8
    return np.sum(
        numerator / denominator + np.log(other_std) - np.log(std), axis=1)

def normal_entropy(pdist):
    """
    Computes the entropy of a multivariate normal distribution. Expects pdist
    to be a numpy matrix. Each row includes the mean and standard deviation
    concatenated together for a single sample.
    Returns a single entropy computed for each sample.
    """
    assert pdist.shape[1] % 2 == 0
    dim = pdist.shape[1] / 2
    _, std = pdist[:, :dim], pdist[:, dim:]
    return np.sum(std + np.log(np.sqrt(2 * np.pi * np.e)), axis=1)

concat = np.concatenate

def ind2onehot(inds, n_cls):
    inds = np.asarray(inds)
    out = np.zeros(inds.shape+(n_cls,), np.float32)
    out.flat[np.arange(inds.size)*n_cls + inds.ravel()] = 1
    return out

def decimate(x, period):
    npad = (-len(x)) % period
    ndecimated = int(np.ceil(len(x)/float(period)))
    x = np.concatenate([x, np.zeros(npad)],0)
    return x.reshape(ndecimated, period).sum(axis=1)

# ================================================================
# Filesystem and printing
# ================================================================

# terminal color codes
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight = False):
    """
    Return string surrounded by appropriate terminal color codes
    to print colorized text.
    Valid colors: gray, red, green, yellow, blue, magenta, cyan, white, crimson
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(unicode(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


MESSAGE_DEPTH = 0
class Message(object):
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        global MESSAGE_DEPTH  # pylint: disable=W0603
        print(colorize('\t'*MESSAGE_DEPTH + '=: ' + self.msg, 'magenta'))
        self.tstart = time.time()
        MESSAGE_DEPTH += 1

    def __exit__(self, etype, *args):
        global MESSAGE_DEPTH  # pylint: disable=W0603
        MESSAGE_DEPTH -= 1
        maybe_exc = "" if etype is None else " (with exception)"
        print(colorize('\t'*MESSAGE_DEPTH + "done%s in %.3f seconds" % (maybe_exc, time.time() - self.tstart), 'magenta'))

def iprint(*args):
    print('\t'*MESSAGE_DEPTH, *args)

# ================================================================
# RL specific
# ================================================================

def rollout(env, agent, max_pathlength):
    """
    Simulate the env and agent for max_pathlength steps
    """
    if hasattr(agent, "reset"): # XXX
        agent.reset()
    ob = env.reset()
    terminated = False

    data = defaultdict(list)
    for _ in xrange(max_pathlength):
        data["observation"].append(ob)
        action, agentinfo = agent.act(ob)
        data["action"].append(action)
        for (k,v) in agentinfo.iteritems():
            data[k].append(v)
        ob,rew,done,envinfo = env.step(action)
        data["reward"].append(rew)
        for (k,v) in envinfo.iteritems():
            data[k].append(v)
        if done:
            terminated = True
            break
    data = {k:np.array(v) for (k,v) in data.iteritems()}
    data["terminated"] = terminated
    return data


def do_rollouts_serial(env, agent, max_pathlength, n_timesteps, seed_iter):
    paths = []
    timesteps_sofar = 0
    while True:
        np.random.seed(seed_iter.next())
        path = rollout(env, agent, max_pathlength)
        paths.append(path)
        timesteps_sofar += pathlength(path)
        if timesteps_sofar > n_timesteps:
            break
    return paths

class LinearVF(object):
    coeffs = None
    def __init__(self, use_obs=True):
        self.use_obs = use_obs
    def _features(self, path):
        o = path["observation"].astype('float32')
        o = o.reshape(o.shape[0], -1)
        l = pathlength(path)
        al = np.arange(l).reshape(-1, 1)/100.0
        if self.use_obs:
            return np.concatenate([o, o**2, al, al**2, np.ones((l, 1))], axis=1)
        else:
            return np.concatenate([al, al**2, np.ones((l, 1))], axis=1)
    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["return"] for path in paths])
        self.coeffs = np.linalg.lstsq(featmat, returns)[0]
    def predict(self, path):
        return np.zeros(pathlength(path)) if self.coeffs is None else self._features(path).dot(self.coeffs)

def pathlength(path):
    return len(path["action"])

def animate_rollout(env, agent, n_timesteps,delay=.01):
    ob = env.reset()
    if hasattr(agent,"reset"): agent.reset()
    env.render()
    for i in xrange(n_timesteps):
        a, _info = agent.act(ob)
        (ob, _rew, done, _info) = env.step(a)
        env.render()
        if done:
            print("terminated after %s timesteps"%i)
            break
        time.sleep(delay)

# ================================================================
# Configuration
# ================================================================

def update_default_config(tuples, usercfg):
    """
    inputs
    ------
    tuples: a sequence of 4-tuples (name, type, defaultvalue, description)
    usercfg: dict-like object specifying overrides

    outputs
    -------
    dict2 with updated configuration
    """
    out = dict2()
    for (name,_,defval,_) in tuples:
        out[name] = defval
    if usercfg:
        for (k,v) in usercfg.iteritems():
            if k in out:
                out[k] = v
    return out

def update_argument_parser(parser, options, **kwargs):
    kwargs = kwargs.copy()
    for (name,typ,default,desc) in options:
        flag = "--"+name
        if flag in parser._option_string_actions.keys(): #pylint: disable=W0212
            print("warning: already have option %s. skipping"%name)
        else:
            parser.add_argument(flag, type=typ, default=kwargs.pop(name,default), help=desc or " ")
    if kwargs:
        raise ValueError("options %s ignored"%kwargs)

def comma_sep_ints(s):
    if s:
        return map(int, s.split(","))
    else:
        return []

IDENTITY = lambda x:x

GENERAL_OPTIONS = [
    ("seed",int,0,"random seed"),
    ("metadata",str,"","metadata about experiment"),
    ("outfile",str,"/tmp/a.h5","output file"),
    ("snapshot_every",int,0,"how often to snapshot"),
    ("load_snapshot",str,"","path to snapshot")
]

# ================================================================
# Load/save
# ================================================================


def prepare_h5_file(args):
    outfile_default = "/tmp/a.h5"
    fname = args.outfile or outfile_default
    if osp.exists(fname) and fname != outfile_default:
        raw_input("output file %s already exists. press enter to continue. (exit with ctrl-C)"%fname)
    hdf = h5py.File(fname,"w")
    hdf.create_group('params')
    for (param,val) in args.__dict__.items():
        try: hdf['params'][param] = val
        except (ValueError,TypeError):
            print("not storing parameter",param)
    diagnostics = defaultdict(list)
    print("Saving results to %s"%fname)
    def save():
        hdf.create_group("diagnostics")
        for (diagname, val) in diagnostics.items():
            hdf["diagnostics"][diagname] = val
    hdf["cmd"] = " ".join(sys.argv)
    atexit.register(save)

    return hdf, diagnostics

class Snapshottable(object):
    def load_snapshot(self, grp):
        raise NotImplementedError
    def save_snapshot(self, grp):
        raise NotImplementedError

def load_spec(s):
    return cPickle.loads(s)
def dump_spec(obj):
    return cPickle.dumps(obj)


# ================================================================
# Misc
# ================================================================

class dict2(dict):
    "dictionary-like object that exposes its keys as attributes"
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def train_val_test_slices(n, trainfrac, valfrac, testfrac):
    assert trainfrac+valfrac+testfrac==1.0
    ntrain = int(np.round(n*trainfrac))
    nval = int(np.round(n*valfrac))
    ntest = n - ntrain - nval
    return slice(0,ntrain), slice(ntrain,ntrain+nval), slice(ntrain+nval,ntrain+nval+ntest)

# helper methods to print nice table
def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim==0
        x = x.item()
    if isinstance(x, float): rep = "%g"%x
    else: rep = str(x)
    return " "*(l - len(rep)) + rep

def fmt_row(width, row, header=False):
    out = " | ".join(fmt_item(x, width) for x in row)
    if header: out = out + "\n" + "-"*len(out)
    return out

def get_and_create_download_dir():
    dirname = osp.expanduser("~/Data")
    os.makedirs(dirname)
    return dirname

def download_s3(url, datapath):
    "download and return path to file"
    subprocess.check_call("aws s3 cp %s %s"%(url, datapath),shell=True)
    datadir = osp.dirname(datapath)
    if not osp.exists(datapath):
        print("downloading %s to %s"%(url, datapath))
        if not osp.exists(datadir): os.makedirs(datadir)
        urllib.urlretrieve(url, datapath)
    return datapath

def download_web(url, datapath):
    "download and return path to file"
    datadir = osp.dirname(datapath)
    if not osp.exists(datapath):
        print("downloading %s to %s"%(url, datapath))
        if not osp.exists(datadir): os.makedirs(datadir)
        urllib.urlretrieve(url, datapath)
    return datapath


def fetch_dataset(url):
    fname = osp.basename(url)
    datadir = get_and_create_download_dir()
    datapath = osp.join(datadir, fname)
    if url.startswith("s3"):
        datapath = download_s3(url, datapath)
    else:
        datapath = download_web(url, datapath)
    fname = osp.basename(url)
    extension =  osp.splitext(fname)[-1]
    assert extension in [".npz", ".pkl"]
    if extension == ".npz":
        return np.load(datapath)
    elif extension == ".pkl":
        with open(datapath, 'rb') as fin:
            return cPickle.load(fin)
    else:
        raise NotImplementedError


def zipsame(*seqs):
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])
    return zip(*seqs)

def flatten(arrs):
    return np.concatenate([arr.flat for arr in arrs])

def unflatten(vec, shapes):
    i=0
    arrs = []
    for shape in shapes:
        size = np.prod(shape)
        arr = vec[i:i+size].reshape(shape)
        arrs.append(arr)
    return arrs

class EzPickle(object):
    """Objects that are pickled and unpickled via their constructor
    arguments.

    Example usage:

        class Dog(Animal, EzPickle):
            def __init__(self, furcolor, tailkind="bushy"):
                Animal.__init__()
                EzPickle.__init__(furcolor, tailkind)
                ...

    When this object is unpickled, a new Dog will be constructed by passing the provided
    furcolor and tailkind into the constructor. However, philosophers are still not sure
    whether it is still the same dog.

    This is generally needed only for environments which wrap C/C++ code, such as MuJoCo
    and Atari.
    """
    def __init__(self, *args, **kwargs):
        self._ezpickle_args = args
        self._ezpickle_kwargs = kwargs
    def __getstate__(self):
        return {"_ezpickle_args" : self._ezpickle_args, "_ezpickle_kwargs": self._ezpickle_kwargs}
    def __setstate__(self, d):
        out = type(self)(*d["_ezpickle_args"], **d["_ezpickle_kwargs"])
        self.__dict__.update(out.__dict__)

