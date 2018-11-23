import glob
import gym
from gym import wrappers
import itertools
import logging
import load_policy
import numpy as np
import tf_util
from tqdm import tqdm
import tensorflow as tf


slim = tf.contrib.slim
BATCH_SIZE = 32



class Setup(object):
    """Class responsible for startup and teardown ops."""
    def startup(self):
        """Perform all setup ops."""

        # logging setup
        logging.basicConfig(level=logging.DEBUG,
                            format="[%(asctime)s]%(levelname)s: %(message)s")

        # getting environment and model info
        env_models = {}
        for model in glob.glob("experts/*"):
            env = (model.strip().split("/")[1]).split(".")[0]
            env_models[env] = model
        return env_models

    def teardown(self):
        """Perform all tear down ops."""
        pass


class Policy():
    # policy hyperparameters
    LEARNING_RATE = 0.001
    BETA = 0.9

    def __init__(self, env, obsv_samples=None):
        if(obsv_samples is None):
            obsv_samples = np.array(
                [env.observation_space.sample() for _ in range(1000)])
        self.obs_mean = obsv_samples.mean(axis=0)
        self.obs_std = obsv_samples.std(axis=0)

        self.state = tf.placeholder(tf.float32,[None]+list(env.observation_space.shape))
        self.target_action= tf.placeholder(tf.float32,[None]+list(env.action_space.shape))

        normalized = (self.state - self.obs_mean) / self.obs_std
        net = slim.fully_connected(normalized, 50, scope='fc1', activation_fn=tf.nn.relu)
        net = slim.fully_connected(net, 50, scope='fc2', activation_fn=tf.nn.relu)
        self.policy = slim.fully_connected(net, env.action_space.shape[0], activation_fn=None, scope='policy')

        self.loss = tf.reduce_mean(tf.reduce_sum((self.policy-self.target_action)**2,axis=1))
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE,beta1=self.BETA)
        self.train_op = optimizer.minimize(self.loss)

    def predict(self, s):
        sess = tf.get_default_session()
        return sess.run(self.policy,
                        feed_dict={self.state:s})

    def update(self, s, a):
        sess = tf.get_default_session()
        loss,_ = sess.run([self.loss,self.train_op],
                          feed_dict={self.state: s,
                                     self.target_action: a})
        return loss

    def test_run(self, env, max_steps):
        obsvs = []
        actions = []
        reward = 0.

        obsv = env.reset()
        for steps in itertools.count():
            obsvs.append(obsv)
            logging.debug(f"obsvs: {obsvs}")
            actions.append(self.predict(np.expand_dims(obsv,axis=0))[0])
            logging.debug(f"actions: {actions}")
            obsv, r, done, _ = env.step(actions[-1])
            reward += r
            if steps >= max_steps or done:
                break

        experience = {'observations': np.stack(obsvs,axis=0),
                      'actions': np.squeeze(np.stack(actions,axis=0)),
                      'reward':reward}
        return experience



def gather_expert_experience(num_rollouts, env, policy_fn, max_steps):
    with tf.Session():
        tf_util.initialize()

        returns = []
        observations = []
        actions = []
        for i in tqdm(range(num_rollouts)):
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if steps >= max_steps:
                    break
            returns.append(totalr)

        expert_data = {'observations': np.stack(observations, axis=0),
                       'actions': np.squeeze(np.stack(actions, axis=0)),
                       'returns':np.array(returns)}
        return expert_data


def behavioral_cloning(env_name, expert_policy, num_rollouts, max_timesteps, num_epochs, save=None):
    tf.reset_default_graph()
    env = gym.make(env_name)
    max_steps = max_timesteps or env.spec.timestep_limit
    logging.info(f"""
        loading and building expert policy for:
        - environment: {env_name}
        - max steps: {max_steps}""")
    expert_policy_fn = load_policy.load_policy(expert_policy)
    logging.info("gathering experience...")
    data = gather_expert_experience(num_rollouts, env, expert_policy_fn, max_steps)
    logging.info(f"""
        Expert information:
        - reward mean: {np.mean(data['returns'])}
        - reward std: {np.std(data['returns'])}
    """)
    logging.info('building clone policy...')
    policy = Policy(env,data['observations'])

    with tf.Session():
        tf_util.initialize()

        for epoch in tqdm(range(num_epochs)):
            num_samples = data['observations'].shape[0]
            perm = np.random.permutation(num_samples)

            obsv_samples = data['observations'][perm]
            action_samples = data['actions'][perm]

            loss = 0.
            for k in range(0,obsv_samples.shape[0],BATCH_SIZE):
                loss += policy.update(obsv_samples[k:k+BATCH_SIZE],
                                     action_samples[k:k+BATCH_SIZE])
                logging.debug(f"loss: {loss}")
            new_exp = policy.test_run( env, max_steps )
            tqdm.write("Epoch %3d Loss %f Reward %f" %(epoch,loss/num_samples,new_exp['reward']))

        if(save is not None):
            env = wrappers.Monitor(env,save,force=True)

        results = []
        # for _ in tqdm(range(num_rollouts)):
            # results.append(policy.test_run( env, max_steps )['reward'])
        # logging.info("Reward mean & std of Cloned policy: %f(%f)"%(np.mean(results),np.std(results)))


def dagger(env_name, expert_policy):
    logging.info(f"env_name: {env_name} -- expert_policy: {expert_policy}")
    pass

if __name__ == '__main__':
    setup = Setup()
    env_models = setup.startup()

    i = 0
    # behavioral cloning runs
    for env, model in env_models.items():
        if i > 0:
            ret = behavioral_cloning(env_name=env, expert_policy=model, num_rollouts=2, max_timesteps=None, num_epochs=10)
            # break
        i += 1
    # # dagger runs
    # for env, model in env_models.items():
    #     ret = dagger(env_name=env, expert_policy=model)

    setup.teardown()