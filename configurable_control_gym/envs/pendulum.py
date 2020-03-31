import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class PendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, force=10.0, length=1.0, mass=1.0):
        if isinstance(force, list):
            self.g_set = force
        else:
            self.g_set = None
            self.g = force

        if isinstance(length, list):
            self.l_set = length
        else:
            self.l_set = None
            self.l = length

        if isinstance(mass, list):
            self.m_set = mass
        else:
            self.m_set = None
            self.m = mass

        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.viewer = None

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.steps_in_top = 0

        self.seed()

    def _sample_parameter(self):
        if self.g_set is not None:
            set_index = self.np_random.randint(len(self.g_set))
            self.g = self.np_random.uniform(self.g_set[set_index][0],
                                            self.g_set[set_index][1])
        if self.l_set is not None:
            set_index = self.np_random.randint(len(self.l_set))
            self.l = self.np_random.uniform(self.l_set[set_index][0],
                                            self.l_set[set_index][1])
        if self.m_set is not None:
            set_index = self.np_random.randint(len(self.m_set))
            self.m = self.np_random.uniform(self.m_set[set_index][0],
                                            self.m_set[set_index][1])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        th, thdot = self.state # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])

        target = np.pi / 3.0
        _newth = newth
        if np.abs(_newth) > 2.0 * np.pi:
            _newth = np.sign(_newth) * (np.abs(_newth) - 2.0 * np.pi * (_newth // (2.0 * np.pi)))
        if np.abs(_newth) < target or (2.0 * np.pi - np.abs(_newth)) < target:
            self.steps_in_top += 1
        else:
            self.steps_in_top = 0

        info = {}
        info['success'] = self.steps_in_top >= 100

        return self._get_obs(), -costs, False, info

    def reset(self):
        self._sample_parameter()
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        self.steps_in_top = 0
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(self.l, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(gym.envs.classic_control.pendulum.__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
