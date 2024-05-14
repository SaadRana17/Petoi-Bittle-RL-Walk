from threading import active_count
import time
import pybullet as p

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from opencat_gym_env_render import OpenCatGymEnvrender
# from opencat_gym_env import OpenCatGymEnv

# Create OpenCatGym environment from class
env = OpenCatGymEnvrender()
# env = OpenCatGymEnv()
#check_env(env)
model = PPO.load("/home/bilal/Documents/opencat-gym-training-v6/models_Elavation/PPO_4090000.zip")
# model = PPO.load("trained/PPO_1000000")
obs = env.reset()


while True:    
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render(mode="human")
    
    if done:
        obs = env.reset()
