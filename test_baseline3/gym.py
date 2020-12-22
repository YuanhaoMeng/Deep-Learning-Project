import gym
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

hist_rewards = []
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    hist_rewards.append(reward)

    # env.render()
    if done:
      obs = env.reset()


# Plot
plt.plot(hist_rewards)
plt.ylabel('Reward')
plt.savefig('Rewards.png')

env.close()