import os
import os.path as osp
import gym
import time
import datetime
import joblib
import logging
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.common import tf_util

from baselines.a2c.utils import discount_with_dones
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.utils import cat_entropy, mse

class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):

        sess = tf_util.make_session()
        nact = ac_space.n
        nbatch = nenvs*nsteps

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        # Defines step_model function and train_model functions
        # Pass each model a copy of 'sess'
        print("Constructing model... STEP_MODEL & TRAIN_MODEL: constructing step_model policy | " + str(policy))
        step_model = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)

        # train_model takes in the mini-batch produced by 5 step_models, NOTE: reuse = true
        train_model = policy(sess, ob_space, ac_space, nenvs*nsteps, nsteps, reuse=True)

        # this neglogpac is still somewhat unknown, looks like it does softmax over policy layer of training model
        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)

        # policy gradient loss determined by advantage * neglogpac
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # value function loss is mse(tf.squeeze(train_model.vf), R)
        # ^ in english, mse(model value prediction, actual Reward)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))

        # entropy of policy
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))

        # total loss calculation?
        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef


        # params gets trainable variables from model (weights of network?)
        params = find_trainable_variables("model")

        # computes gradients (change of weights, or direction of weights) using 'loss' and 'params' above
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        # TODO: how many gradients are computed here, should be 16
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        # RMSProp pushes back new gradients over weights
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        # Trains the model,
        # TODO: What is 'masks' input param
        # TODO: How often does train_model (steps thru train_model) get run vs. step_model
        #   A: I think it does a 'train_model' for each mini-batch, which is currently 5 steps
        # Does a sess.run with train_model
        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            # td_map hooks up all inputs for train model?
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}

            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks

            # Policy Loss, Value Loss, and Policy Entropy calculations

            # Propagates losses backwards through the neural network?
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            ps = sess.run(params)
            make_path(osp.dirname(save_path))
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)

class Runner(object):

    # Run is passed a model and nsteps default to 5, runs both models?
    def __init__(self, env, model, nsteps=5, gamma=0.99):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc)
        self.obs = np.zeros((nenv, nh, nw, nc), dtype=np.uint8)
        self.nc = nc
        obs = env.reset()
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    # run() steps through 'nsteps' of each 'nenvs' environment, adds actions values
    # 'nsteps' is 5 actions set above
    def run(self):
        # initializes mini-batch arrays
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states

        # For each step n (5), the model steps through each environment without 'learning' anything, adds rewards
        for n in range(self.nsteps):
            actions, values, states, _ = self.model.step(self.obs, self.states, self.dones)

            #print("#######************###### ACTIONS PRINT: " + str(n))
            #print(str(actions))
            #print(str(values))

            # Records actions and values predicted from the model.step() call above
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            # Executes the actions predicted above
            print("RUNNER: self.env: " + str(self.env))
            obs, rewards, dones, _ = self.env.step(actions)
            print("RUNNER: len(obs): " + str(len(obs)))

            print("RUNNER: len(rewards): " + str(len(rewards)))


            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)

        #batch of steps to batch of rollouts, aggregates all observations, rewards, actions, values, dones, swaps axis?
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()


        #discount/bootstrap off value fn

        # For each (reward, dones, value) tuple in enumerate(zip(..,..,..) : add rewards to list, add dones to list,
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards

        # Todo: What are these values, print out, the original data is .flattened() to produce return vals
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values

def learn(policy, env, seed, nsteps=5, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    print('rockin ' + str(nenvs))
    ob_space = env.observation_space
    ac_space = env.action_space
    print('observation space: ' + str(ob_space))
    print('action space: ' + str(ac_space))

    # Initializes model with all arguments obtained from run_atari
    #   Model DOES NOT GET the env stack object
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)

    # Intializes a runner using the above model, an environment, and nsteps to run '5'
    # env is the VectorFrameStack object created in run_atari, holds 16 environments
    #   Runner DOES GET the env stack object
    #   Runner DOES get the model, which lacks the env stack object
    runner = Runner(env, model, nsteps=nsteps, gamma=gamma)


    file = open("testOutput.txt", "w")
    file.write(str(datetime.datetime.now()))

    nbatch = nenvs*nsteps
    tstart = time.time()
    i = 0

    prevAvgReward = 0

    # Todo: Figure out how frequently this is: loop 1 to 137,501
    for update in range(1, total_timesteps//nbatch+1):
        print("__________ Control loop goes from 1 -> " + str(total_timesteps//nbatch+1))

        print("_____________________ Super main loop, hits run, hits train: " + str(i))
        i += 1
        # runner.run(), steps model, returns observations, states, rewards, masks, actions, values for all agents?
        obs, states, rewards, masks, actions, values = runner.run()
        # 80 observations, 16 envs * 5 steps
        print("LEARNING FROM: len(obs): " + str(len(obs)))
        # Printing states: TypeError: object of type 'NoneType' has no len()
        #print("len(states): " + str(len(states)))
        print("LEARNING FROM: len(rewards): " + str(len(rewards)))

        # model.train(), trains model, takes all that above data, processes it through train_model
        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)

        if update % log_interval == 0 or update == 1:
            avgReward = 0
            rewardCount = 0
            for reward in rewards:
                # Prints 80 reward values? (5 training steps * 16 nenvs) = 80 reward values
                print(reward)
                avgReward += reward
                rewardCount += 1

            avgReward = avgReward / rewardCount
            ev = explained_variance(values, rewards)

            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("avgReward", float(avgReward))
            logger.record_tabular("explained_variance", float(ev))

            logger.dump_tabular()

            # If avg reward of this batch is greater than previous avg reward, save model
            if avgReward > prevAvgReward:
                logger.log("Saving model due to mean reward increase: {} -> {}".format(
                    prevAvgReward, avgReward))

                # Save model
                model.save()

                # Set prevAvgReward = avgReward
                prevAvgReward = avgReward



    file.close()
    env.close()
