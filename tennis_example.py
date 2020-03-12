import numpy as np
from colorama import Fore, Style

from Gymwrapper import UnityEnv


def main():
    """
    observation: [[agent_0] [agent_1]] np.array dim:24
    action: [[agent_0] [agent_1]] np.array agent dim:Box(3, )
    example: [array([[0.729623, 0.13617396, 0.8946541]], dtype=float32), array([[-0.7937411 ,  0.00580084, -0.18735716]], dtype=float32)]
    :return:
    """
    env_name = "envs/Tennis/mlagent"  # Name of the Unity environment binary to launch

    env = UnityEnv(env_name)

    env.reset()
    # print(env.action_space)
    # print(env.observation_space)

    for episode in range(100):
        episode_reward = [[0, 0]]
        step = 0
        obs_n, _, _, _ = env.reset()
        while True:
            step += 1
            action = []
            for _ in range(2):
                act = env.action_space.sample()
                act = np.expand_dims(act, 0)
                action.append(act)
            o, r, d, _ = env.step(action)
            # print(action)
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
