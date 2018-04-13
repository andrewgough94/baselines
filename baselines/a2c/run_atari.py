#!/usr/bin/env python3

from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.a2c.a2c import learn
#from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy GITHUB ISSUE: references ppo2 instead of a2c
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy

def train(env_id, num_timesteps, seed, policy, lrschedule, num_env):
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    # VecFrameStack
    # make_atari_env() : launches 'num_env' subprocess each with 'env_id' and for i in num_env: seed+=seed+i
    env = VecFrameStack(make_atari_env(env_id, num_env, seed), 4)
    print("~~~~~~~~~~~~~ run_atari: len(env): " + str(env.nstack))
    print("~~~~~~~~~~~~~ run_atari: str(env): " + str(env))
    # above prints : run_atari: str(env): <baselines.common.vec_env.vec_frame_stack.VecFrameStack object at 0x1c22ee06d8>
    print("_____________________________________________ policy: " + str(policy))
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule)
    env.close()

def main():
    parser = atari_arg_parser()
    # Below line is unnecessary because atari_arg_parser() handles env and steps to run for
    #parser.add_argument('--env', help='Atari Environment', default='BreakoutNoFrameskip-v0')
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    args = parser.parse_args()
    logger.configure()
    print("xxxxxxxxxxxxxxxxxxxxxxxx            : " + args.env)

    # train(...) initializes environments, and calls learn(...) with all arguments
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, lrschedule=args.lrschedule, num_env=1)

if __name__ == '__main__':
    main()
