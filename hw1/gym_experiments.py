import gym

def gym_experiments():
    env = gym.make('Ant-v2')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print("observations:", observation.shape)
            action = env.action_space.sample()
            print("action:", action)
            observation, reward, done, info = env.step(action)
            print("reward:", reward)
            print("info:", info)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break


if __name__ == "__main__":
    gym_experiments()