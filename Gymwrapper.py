import logging
import itertools
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
from gym import error, spaces

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import BatchedStepResult
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel


# just for tennis and soccer!!!
class UnityEnv(gym.Env):

    def __init__(
            self, environment_filename
    ):
        engine_configuration_channel = EngineConfigurationChannel()
        self.env = UnityEnvironment(file_name=environment_filename, side_channels=[engine_configuration_channel])
        self.env.reset()

        self.brain_name = self.env.get_agent_groups()
        self.group_spec = self.env.get_agent_group_spec(self.brain_name[0])
        engine_configuration_channel.set_configuration_parameters(time_scale=3.0)
        self.group_name = self.brain_name

        # Set observation and action spaces
        if self.group_spec.is_action_discrete():
            self._action_space = []
            branches = self.group_spec.discrete_action_branches
            # if self.group_spec.action_shape == 1:
            for _ in range(2):
                self._action_space.append([spaces.Discrete(branches[i]) for i in range(len(branches))])
        else:
            high = np.array([1] * self.group_spec.action_shape)
            self._action_space = spaces.Box(-high, high, dtype=np.float32)

        high = np.array([np.inf] * self._get_vec_obs_size())

        self._observation_space = spaces.Box(-high, high, dtype=np.float32)

    def reset(self):
        o, r, d, info = [], [], [], []
        self.env.reset()
        for i in range(2):
            step_result = self.env.get_step_result(self.group_name[i])
            o.append(step_result.obs[0])
            r.append(step_result.reward[0])
            d.append(step_result.done[0])
        return o, r, d, info

    def step(self, action):
        o, r, d, info = [], [], [], []
        assert len(action) == 2
        self.env.set_actions(self.group_name[0], action[0])
        self.env.set_actions(self.group_name[1], action[1])
        self.env.step()
        for i in range(2):
            step_result = self.env.get_step_result(self.group_name[i])
            o.append(step_result.obs[0])
            r.append(step_result.reward[0])
            d.append(step_result.done[0])
        return o, r, d, info

    def render(self, mode="rgb_array"):
        return self.visual_obs

    def close(self) -> None:
        self.env.close()

    def seed(self, seed: Any = None) -> None:
        """Sets the seed for this env's random number generator(s).
        Currently not implemented.
        """
        return

    @property
    def metadata(self):
        return {"render.modes": ["rgb_array"]}

    @property
    def reward_range(self) -> Tuple[float, float]:
        return -float("inf"), float("inf")

    def _get_n_vis_obs(self) -> int:
        result = 0
        for shape in self.group_spec.observation_shapes:
            if len(shape) == 3:
                result += 1
        return result

    def _get_vis_obs_shape(self) -> Optional[Tuple]:
        for shape in self.group_spec.observation_shapes:
            if len(shape) == 3:
                return shape
        return None

    def _get_vis_obs_list(self, step_result: BatchedStepResult) -> List[np.ndarray]:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 4:
                result.append(obs)
        return result

    def _get_vector_obs(self, step_result: BatchedStepResult) -> np.ndarray:
        result: List[np.ndarray] = []
        for obs in step_result.obs:
            if len(obs.shape) == 2:
                result.append(obs)
        return np.concatenate(result, axis=1)

    def _get_vec_obs_size(self) -> int:
        result = 0
        for shape in self.group_spec.observation_shapes:
            if len(shape) == 1:
                result += shape[0]
        return result

    @property
    def spec(self):
        return None

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def number_agents(self):
        return self._n_agents
