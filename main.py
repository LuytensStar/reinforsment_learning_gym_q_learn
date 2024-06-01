import gym

env = gym.make("Taxi-v3", render_mode='ansi')
env.reset()
print(env.render())

print("Action space {}".format(env.action_space))
print("State space {}".format(env.observation_space))

state = env.encode(7, 1, 2, 0)
print("State: ", state)

env.s = state
print(env.render())

print(env.P[328])