import gym

env = gym.make('Breakout-v0')
print(env.action_space)
print(env.observation_space)
print(env.unwrapped.get_action_meanings())
observation=env.reset()
for _ in range(1000):
    env.render()
    print(observation)

    env.step(env.action_space.sample()) # take a random action
env.close()