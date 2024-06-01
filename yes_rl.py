import random

import numpy as np
import gym
from IPython.display import clear_output
from time import sleep

env = gym.make("Taxi-v3", render_mode='ansi')

env.s = 328

q_table = np.zeros([env.observation_space.n, env.action_space.n])

alpha = 0.1
gamma = 0.6
epsilon = 0.1

all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()
    # print(state[0])
    if isinstance(state, tuple):
        state = state[0]

    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()

        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, truncated, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
                    #(1-0.1)*25 + 0.1*(10 + 0.6*32)
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    if i % 100 == 0:
        clear_output(wait=True)
        print(F"episode: {i}")

print("Training finished")
# np.save('q_trained.npy', q_table)

