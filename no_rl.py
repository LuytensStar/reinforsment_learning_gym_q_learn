#Solving problem without RL (Brute-force approach)
import gym
from IPython.display import clear_output
from time import sleep

env = gym.make("Taxi-v3", render_mode='ansi')

env.s = 328

epochs = 0
penalties, reward = 0, 0

frames = []

terminated = False

env.reset()

while not terminated:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if reward == -10:
        penalties+=1

    frames.append({
        'frame': env.render(),
        'state' : observation,
        'action': action,
        'reward': reward,
    })

    epochs+=1

print('Timesteps taken: {}'.format(epochs))
print('Penalties incurred{}'.format(penalties))


def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        # print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)


# print_frames(frames)
