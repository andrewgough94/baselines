from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger
from baselines.common.atari_wrappers import make_atari

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--bufferSize', type=int, default=10000)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--learningRate', type=float, default=5e-4)
    parser.add_argument('--epsStart', type=float, default=1.0)
    parser.add_argument('--epsEnd', type=float, default=.05)
    parser.add_argument('--learningStart', type=int, default=int(1000))
    parser.add_argument('--targetNetworkUpdate', type=int, default=int(500))

    args = parser.parse_args()
    logger.configure()
    set_global_seeds(args.seed)
    env = make_atari(args.env)
    env = bench.Monitor(env, logger.get_dir())
    env = deepq.wrap_atari_dqn(env)
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=bool(args.dueling),
    )
    act = deepq.learn(
        env,
        q_func=model,
        lr=args.learningRate,
        max_timesteps=args.num_timesteps,
        buffer_size=args.bufferSize,
        exploration_fraction=args.epsStart,
        exploration_final_eps=args.epsEnd,
        train_freq=4,
        learning_starts= args.learningStart,
        target_network_update_freq=args.targetNetworkUpdate,
        gamma=0.99,
        prioritized_replay=bool(args.prioritized)
    )
    act.save()
    env.close()


if __name__ == '__main__':
    main()
