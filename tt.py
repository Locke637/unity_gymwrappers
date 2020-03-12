import matplotlib.pyplot as plt
import numpy as np
import sys

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel

from Gymwrapper import UnityEnv


def main():
    env_name = "envs/Soccer/mlagent"  # Name of the Unity environment binary to launch

    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name, side_channels=[engine_configuration_channel])

    # env = UnityEnv(env_name)

    # Reset the environment
    env.reset()

    # Set the default brain to work with
    print(env.get_agent_groups())
    group_name_0 = env.get_agent_groups()[0]
    group_name_1 = env.get_agent_groups()[1]
    group_spec_0 = env.get_agent_group_spec(group_name_0)
    group_spec_1 = env.get_agent_group_spec(group_name_1)
    # print(group_spec_0.action_shape)

    # Set the time scale of the engine
    engine_configuration_channel.set_configuration_parameters(time_scale=3.0)

    for episode in range(10):
        env.reset()
        step = 0
        step_result_0 = env.get_step_result(group_name_0)
        step_result_1 = env.get_step_result(group_name_1)
        done = False
        episode_rewards = 0
        while not done:
            step += 1
            action_size_0 = group_spec_0.action_size
            action_size_1 = group_spec_1.action_size
            if group_spec_0.is_action_continuous():
                action_0 = np.random.randn(step_result_0.n_agents(), group_spec_0.action_size)

            if group_spec_0.is_action_discrete():
                branch_size = group_spec_0.discrete_action_branches
                action_0 = np.column_stack(
                    [np.random.randint(0, branch_size[i], size=(step_result_0.n_agents())) for i in
                     range(len(branch_size))])

            if group_spec_1.is_action_continuous():
                action_1 = np.random.randn(step_result_1.n_agents(), group_spec_1.action_size)

            if group_spec_1.is_action_discrete():
                branch_size = group_spec_1.discrete_action_branches
                action_1 = np.column_stack(
                    [np.random.randint(0, branch_size[i], size=(step_result_1.n_agents())) for i in
                     range(len(branch_size))])

            # print(group_name)
            print(step, action_0)
            env.set_actions(group_name_0, action_0)
            env.set_actions(group_name_1, action_1)
            env.step()
            step_result_0 = env.get_step_result(group_name_0)
            step_result_1 = env.get_step_result(group_name_1)
            # print(step_result_0.obs, step_result_1.obs)
            episode_rewards += step_result_0.reward[0]
            done = step_result_0.done[0]
        print("Total reward this episode: {}".format(episode_rewards))
        print(step)


if __name__ == '__main__':
    main()
