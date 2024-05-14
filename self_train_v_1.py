import gym
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import os
# from opencat_gym_env import OpenCatGymEnv

from opencat_gym_env_render import OpenCatGymEnvrender

models_dir = "models_Elavation"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = OpenCatGymEnvrender()
env.reset()
#env = DummyVecEnv([lambda: env])  # Wrap the environment

# Define a custom MLP policy with a deeper architecture
# class CustomMlpPolicy(ActorCriticPolicy):
#     def __init__(self, *args, **kwargs):
#         super(CustomMlpPolicy, self).__init__(*args, **kwargs, net_arch=[256, 256, 256])

# algorithms = [A2C, SAC, TD3]

# for algo in algorithms:
#     # Create and train the model with the custom policy
#     #model = algo('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
#     model = algo('MlpPolicy', env, verbose=1, n_steps=2048, batch_size=64, learning_rate=2.5e-4, ent_coef=0, tensorboard_log=logdir)
#     TIMESTEPS = 1
#     for i in range(1, 500):
#         model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"{algo.__name__}")
#         model.save(f"{models_dir}/{algo.__name__}_{TIMESTEPS * i}")



model = PPO('MlpPolicy', env, verbose=1, n_steps=1024, batch_size=256, learning_rate=1e-4, ent_coef=0, tensorboard_log=logdir)


TIMESTEPS = 10000

for i in range(1, 1000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/PPO_{TIMESTEPS * i}")


env.close()
