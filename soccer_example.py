import numpy as np
from colorama import Fore, Style

from Gymwrapper import UnityEnv


def main():
    """
    observation: [[agent_0 agent_1] [agent_0 agent_1]] np.array dim:264
    action: [[agent_0 agent_1] [agent_0 agent_1]] np.array agent dim:[Discrete(3), Discrete(3), Discrete(3)]
    example: [[[2 1 2] [1 2 1]] [[0 1 0] [2 1 0]]]
    :return:
    """
    env_name = "envs/Soccer/mlagent"  # Name of the Unity environment binary to launch

    env = UnityEnv(env_name)

    env.reset()
    print(env.action_space)
    # print(env.observation_space)

    for episode in range(100):
        episode_reward = [[0, 0]]
        step = 0
        obs_n, _, _, _ = env.reset()
        while True:
            step += 1
            action = []
            for _ in range(2):
                action_team = []
                for agent in range(2):
                    action_agent = []
                    for act in env.action_space[agent]:
                        action_agent.append(act.sample())
                    action_team.append(action_agent)
                # act = np.expand_dims(act, 0)
                action.append(action_team)
            action = np.array(action)

            o, r, d, _ = env.step(action)
            # print(Fore.GREEN, o)
            # print(Fore.RED, o[1])
            episode_reward[-1][0] += r[0]
            episode_reward[-1][1] += r[1]
            if d[0] and d[1]:
                print(step, episode_reward[-1])
                episode_reward.append([0, 0])
                break


if __name__ == '__main__':
    main()
