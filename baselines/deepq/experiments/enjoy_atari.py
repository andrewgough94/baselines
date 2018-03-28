import gym
from baselines import deepq
import argparse

def main():


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--dir', help='model directory', default='')

    args = parser.parse_args()

    print("Args env: " + args.env)
    env = gym.make(args.env)
    env = deepq.wrap_atari_dqn(env)
    act = deepq.load(args.dir)


    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
