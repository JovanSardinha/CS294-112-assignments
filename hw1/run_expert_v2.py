from tqdm import tqdm
import tensorflow as tf
import numpy as np
import itertools
import gym
from gym import wrappers

import tf_util
import load_policy

slim = tf.contrib.slim

BATCH_SIZE = 32
LEARNING_RATE = 0.001
BETA = 0.9

class Policy():
    def __init__(self,env,obsv_samples=None):
        if( obsv_samples is None ):
            obsv_samples = np.array([env.observation_space.sample() for _ in range(1000)])
        self.obs_mean = obsv_samples.mean(axis=0)
        self.obs_std = obsv_samples.std(axis=0)

        self.state = tf.placeholder(tf.float32,[None]+list(env.observation_space.shape))
        self.target_action= tf.placeholder(tf.float32,[None]+list(env.action_space.shape))

        normalized = (self.state - self.obs_mean) / self.obs_std
        net = slim.fully_connected(normalized, 50, scope='fc1', activation_fn=tf.nn.relu)
        net = slim.fully_connected(net, 50, scope='fc2', activation_fn=tf.nn.relu)
        self.policy = slim.fully_connected(net, env.action_space.shape[0], activation_fn=None, scope='policy')

        self.loss = tf.reduce_mean(tf.reduce_sum((self.policy-self.target_action)**2,axis=1))
        #loss = tf.reduce_mean(tf.where(tf.abs(error) < 1.0, #Huber loss; gradient clipping
        #                               0.5 * tf.square(error),
        #                               tf.abs(error) - 0.5))

        optimizer = tf.train.AdamOptimizer(LEARNING_RATE,beta1=BETA) #works best;
        #optimizer = tf.train.MomentumOptimizer(LEARNING_RATE,BETA)
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

    def test_run(self,env,max_steps):
        obsvs = []
        actions = []
        reward = 0.

        obsv = env.reset()
        for steps in itertools.count() :
            obsvs.append(obsv)
            actions.append(self.predict(np.expand_dims(obsv,axis=0))[0])
            obsv, r, done, _ = env.step(actions[-1])
            reward += r
            if steps >= max_steps or done:
                break

        experience = {'observations': np.stack(obsvs,axis=0),
                      'actions': np.squeeze(np.stack(actions,axis=0)),
                      'reward':reward}
        return experience

def gather_expert_experience(num_rollouts,env,policy_fn,max_steps):
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

        expert_data = {'observations': np.stack(observations,axis=0),
                       'actions': np.squeeze(np.stack(actions,axis=0)),
                       'returns':np.array(returns)}
        return expert_data


def behavior_cloning(env_name='Hopper-v1',
         expert_policy_file='experts/Hopper-v1.pkl',
         num_rollouts=10,
         max_timesteps=None,
         num_epochs=100,
         save=None):
    tf.reset_default_graph()

    env = gym.make(env_name)
    max_steps = max_timesteps or env.spec.timestep_limit

    print('loading and building expert policy')
    expert_policy_fn = load_policy.load_policy(expert_policy_file)
    print('gather experience...')
    data = gather_expert_experience(num_rollouts,env,expert_policy_fn,max_steps)
    print("Expert's reward mean : %f(%f)"%(np.mean(data['returns']),np.std(data['returns'])))
    print('building cloning policy')
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
            new_exp = policy.test_run( env, max_steps )
            tqdm.write("Epoch %3d Loss %f Reward %f" %(epoch,loss/num_samples,new_exp['reward']))

        if( save is not None ):
            env = wrappers.Monitor(env,save,force=True)

        results = []
        for _ in tqdm(range(num_rollouts)):
            results.append(policy.test_run( env, max_steps )['reward'])
        print("Reward mean & std of Cloned policy: %f(%f)"%(np.mean(results),np.std(results)))
    return np.mean(data['returns']), np.std(data['returns']), np.mean(results), np.std(results)

def dagger(env_name='Hopper-v1',
         expert_policy_file='experts/Hopper-v1.pkl',
         num_rollouts=10,
         max_timesteps=None,
         num_epochs=100,
         save=None):
    tf.reset_default_graph()

    env = gym.make(env_name)
    max_steps = max_timesteps or env.spec.timestep_limit

    print('loading and building expert policy')
    expert_policy_fn = load_policy.load_policy(expert_policy_file)
    print('gather experience...')
    data = gather_expert_experience(num_rollouts,env,expert_policy_fn,max_steps)
    print("Expert's reward mean : %f(%f)"%(np.mean(data['returns']),np.std(data['returns'])))
    print('building cloning policy')
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

            new_exp = policy.test_run( env, max_steps )

            #Data Aggregation Steps. Supervision signal comes from expert policy.
            new_exp_len = new_exp['observations'].shape[0]
            expert_expected_actions = []
            for k in range(0,new_exp_len,BATCH_SIZE) :
                expert_expected_actions.append(expert_policy_fn(new_exp['observations'][k:k+BATCH_SIZE]))
            # Currently, I added new experience into original one. (No eviction)
            data['observations'] = np.concatenate((data['observations'],new_exp['observations']),
                                                  axis=0)
            data['actions'] = np.concatenate([data['actions']]+expert_expected_actions,
                                             axis=0)
            tqdm.write("Epoch %3d Loss %f Reward %f" %(epoch,loss/num_samples,new_exp['reward']))

        if( save is not None ):
            env = wrappers.Monitor(env,save,force=True)

        results = []
        for _ in tqdm(range(num_rollouts)):
            results.append(policy.test_run( env, max_steps )['reward'])
        print("Reward mean & std of Cloned policy with DAGGER: %f(%f)"%(np.mean(results),np.std(results)))
    return np.mean(data['returns']), np.std(data['returns']), np.mean(results), np.std(results)

if __name__ == "__main__":
    import os
    env_models = [('Ant-v2','experts/Ant-v2.pkl'),
                  ('HalfCheetah-v2','experts/HalfCheetah-v2.pkl'),
                  ('Hopper-v2','experts/Hopper-v2.pkl'),
                  ('Humanoid-v2','experts/Humanoid-v2.pkl'),
                  ('Reacher-v2','experts/Reacher-v2.pkl'),
                  ('Walker2d-v2','experts/Walker2d-v2.pkl'),]

    results = []
    for env,model in env_models :
        ex_mean, ex_std, bc_mean,bc_std = behavior_cloning(env_name=env,
                                                           expert_policy_file=model,
                                                           save=os.path.join(os.getcwd(),env,'bc'))
        _,_, da_mean,da_std = dagger(env_name=env,
                                expert_policy_file=model,
                                num_epochs=40,
                                save=os.path.join(os.getcwd(),env,'da'))
        results.append((env,ex_mean,ex_std,bc_mean,bc_std,da_mean,da_std))

    for env_name,ex_mean,ex_std,bc_mean,bc_std,da_mean,da_std in results :
        print('Env: %s, Expert: %f(%f), Behavior Cloning: %f(%f), Dagger: %f(%f)'%
              (env_name,ex_mean,ex_std,bc_mean,bc_std,da_mean,da_std))