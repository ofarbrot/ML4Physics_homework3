import sys
sys.path.append("/Users/synneeikeland/Documents/ML_course/ML4Physics_homework3/rl-ion-trap-tutorial")

import os
import numpy as np
from tqdm import tqdm

from wrappers import SRVWrapper
from utils import valid_srvs

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

# Settings
N_ENVS = 8
TOTAL_TIMESTEPS = 300_000
CHECKPOINT_DIR = "checkpoints"
MODEL_PATH = "srv_policy"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def make_env():
    def _f():
        srv = valid_srvs[np.random.randint(len(valid_srvs))]
        env = SRVWrapper(srv)
        return env
    return _f

# Create vectorized environment
env = DummyVecEnv([make_env() for _ in range(N_ENVS)])
env = VecNormalize(env, norm_obs=True, norm_reward=False)

policy_kwargs = dict(net_arch=[dict(pi=[256,256], vf=[256,256])])

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    batch_size=64,
    n_steps=512,
    policy_kwargs=policy_kwargs,
    verbose=1
)

checkpoint_callback = CheckpointCallback(
    save_freq=20_000 // N_ENVS,
    save_path=CHECKPOINT_DIR,
    name_prefix="ppo_srv"
)

model.learn(TOTAL_TIMESTEPS, callback=checkpoint_callback)

model.save(MODEL_PATH)
env.save(MODEL_PATH + "_vecnorm.pkl")
