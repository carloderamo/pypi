import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from mj_envs.hand_manipulation_suite import *

import gym
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import *
# from mushroom_rl.utils.viewer import ImageViewer


class MJEnv(Environment):
    """

    """
    def __init__(self, task_name, horizon=None, gamma=0.99,
        camera_id=0, use_pixels=False, pixels_width=64, pixels_height=64):
        """
        Constructor.

        Args:
             task_name (str): name of the task;
             horizon (int): the horizon;
             gamma (float): the discount factor;
             camera_id (int, 0): position of camera to render the environment;
             use_pixels (bool, False): if True, pixel observations are used
                rather than the state vector;
             pixels_width (int, 64): width of the pixel observation;
             pixels_height (int, 64): height of the pixel observation;

        """

        self.env = gym.make(task_name)

        self._camera_id = camera_id
        self._width = pixels_width
        self._height = pixels_height
        self._use_pixels = use_pixels

        # # get the default horizon
        # if horizon is None:
        #     horizon = self.env._step_limit
        #
        # # Hack to ignore dm_control time limit.
        # self.env._step_limit = np.inf

        # MDP properties
        action_space = self.env.action_space
        observation_space = self.env.observation_space
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        # self._viewer = ImageViewer((width_screen, height_screen), dt)
        # self._camera_id = camera_id

        super().__init__(mdp_info)

        self._state = None

    def reset(self, state=None):
        if state is None:
            return self.env.reset()
        else:
            raise NotImplementedError

    def step(self, action):
        return self.env.step(action)

    def pixels(self):
        img = self.env.sim.render(width=width, height=height, camera_name='vil_camera', depth=depth)
        return img[::-1,:,:]

    def render(self):
        pass
        # img = self.env.physics.render(self._viewer.size[1],
        #                               self._viewer.size[0],
        #                               self._camera_id)
        # self._viewer.display(img)

    def stop(self):
        pass
