import torch
import numpy as np
from ion_trap import IonTrapEnv
from stable_baselines3 import PPO

class model:

    def __init__(self):
        try:
            self.policy = PPO.load("srv_policy")
            self.has_policy = True
        except Exception as e:
            print("Warning: Cannot load PPO model:", e)
            self.policy = None
            self.has_policy = False

    def pred(self, samples):

        preds = []

        for srv in samples:
            srv_np = np.array(srv, dtype=np.float32)
            srv_norm = srv_np / 3.0

            env = IonTrapEnv(
                phases={
                    'pulse_angles': [np.pi/2],
                    'pulse_phases': [np.pi/2],
                    'ms_phases': [-np.pi/2]
                },
                num_ions=3,
                goal=[srv.tolist()],
                max_steps=10
            )

            psi = env.reset().flatten()
            obs = np.concatenate([np.real(psi), np.imag(psi), srv_norm]).astype(np.float32)

            seq = []

            for _ in range(10):

                if self.has_policy:
                    action, _ = self.policy.predict(obs, deterministic=True)
                    action = int(action)
                else:
                    action = np.random.randint(0, 7)

                seq.append(action)

                psi, reward, done = env.step(action)
                psi = psi.flatten()
                obs = np.concatenate([np.real(psi), np.imag(psi), srv_norm]).astype(np.float32)

                if reward == 1:
                    break

            preds.append(torch.tensor(seq, dtype=torch.long))

        return preds
