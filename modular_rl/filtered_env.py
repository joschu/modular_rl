from gym import Env, spaces
import numpy as np
import scipy

class FilteredEnv(Env): #pylint: disable=W0223
    def __init__(self, env, ob_filter, rew_filter):
        self.env = env
        self.ob_filter = ob_filter
        self.rew_filter = rew_filter

        ob_space = self.env.observation_space
        shape = self.ob_filter.output_shape(ob_space)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape)
        self.action_space = self.env.action_space

    def _step(self, ac):
        ob, rew, done, info = self.env.step(ac)
        nob = self.ob_filter(ob) if self.ob_filter else ob
        nrew = self.rew_filter(rew) if self.rew_filter else rew
        info["reward_raw"] = rew
        return (nob, nrew, done, info)

    def _reset(self):
        ob = self.env.reset()
        return self.ob_filter(ob) if self.ob_filter else ob

    def _render(self, *args, **kw):
        self.env.render(*args, **kw)


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

class RGBImageToVector(object):
    def __init__(self, out_width=40, out_height=40):
        self.out_width = out_width
        self.out_height = out_height

    def __call__(self, obs):
        # obs is an M x N x 3 rgb image, want an (out_width x out_height,) vector
        grayscale = rgb2gray(obs)
        downsample = scipy.misc.imresize(grayscale, (self.out_width, self.out_height))
        flatten = downsample.reshape(self.out_width * self.out_height)
        return flatten

    def output_shape(self, x):
        return self.out_width * self.out_height
