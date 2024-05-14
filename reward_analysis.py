import gym 
#from stable_baselines3 import PPO, A2C, SAC, TD3
from opencat_gym_env_render import OpenCatGymEnvrender
from stable_baselines3.common.env_util import make_vec_env
import os
import numpy as np


arr = np.array([1.5,4, 1.5,3, 1.5, 3, 1.5, 3])

env = OpenCatGymEnvrender()
env.reset()


episodes = 10

for ep in range(episodes):
    
    obs = env.reset()
    done = False

    while not done:
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())
        #print(reward) 

env.close()
