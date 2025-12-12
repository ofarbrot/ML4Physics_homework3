import gym
import numpy as np
from ion_trap import IonTrapEnv

class SRVWrapper(gym.Env):
    """
    Wraps IonTrapEnv and augments observation with target SRV.
    Observation = [real(psi), imag(psi), normalized SRV]
    """

    def __init__(self, srv, max_steps=40):
        super().__init__()

        self.srv = np.array(srv, dtype=np.float32)
        self.srv_norm = self.srv / 3.0

        phases = {
            'pulse_angles': [np.pi/2],
            'pulse_phases': [np.pi/2],
            'ms_phases': [-np.pi/2]
        }

        self.env = IonTrapEnv(
        phases=phases,
        num_ions=3,
        goal=[srv.tolist()],
        max_steps=max_steps
        )


        self.num_amplitudes = 3**3  # = 27
        obs_dim = 2*self.num_amplitudes + 3

        self.action_space = gym.spaces.Discrete(self.env.num_actions)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(obs_dim,), dtype=np.float32
        )

    def reset(self):
        psi = self.env.reset().flatten()
        return self._build_obs(psi)

    def step(self, action):
        psi, reward, done = self.env.step(int(action))
        return self._build_obs(psi.flatten()), reward, done, {}

    def _build_obs(self, psi):
        psi_real = np.real(psi)
        psi_imag = np.imag(psi)
        return np.concatenate([psi_real, psi_imag, self.srv_norm]).astype(np.float32)
