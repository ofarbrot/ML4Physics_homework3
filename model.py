# model.py
import torch
import numpy as np
from helpers_HW.HW3.ion_trap import IonTrapEnv



class QNetwork(torch.nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class model:
    def __init__(self):
        self.phases = {
            'pulse_angles': [np.pi / 2],
            'pulse_phases': [np.pi / 2],
            'ms_phases': [-np.pi / 2],
        }
        self.num_ions = 3
        self.max_steps = 10
        self.num_actions = 7

        # 27 complex amplitudes → 54 real + 3 SRV
        self.state_dim = 54 + 3

        self.q_net = QNetwork(self.state_dim, self.num_actions)

        # Load trained weights if available
        try:
            self.q_net.load_state_dict(torch.load("qnet.pt", map_location="cpu"))
            self.q_net.eval()
        except FileNotFoundError:
            print("Warning: qnet.pt not found, using random policy.")

    def pred(self, samples):
        sequences = []

        for srv in samples:
            env = IonTrapEnv(
                phases=self.phases,
                num_ions=self.num_ions,
                goal=[list(srv)],
                max_steps=self.max_steps,
            )

            state = env.reset()
            seq = []

            for _ in range(self.max_steps):
                obs = self.make_obs(state, srv)
                with torch.no_grad():
                    q_vals = self.q_net(obs)
                    action = torch.argmax(q_vals).item()

                state, reward, done = env.step(action)
                seq.append(action)

                if done:
                    break

            sequences.append(torch.tensor(seq, dtype=torch.long))

        return sequences

    def make_obs(self, state, srv):
        state = torch.tensor(state, dtype=torch.complex64).flatten()
        obs = torch.cat(
            [
                torch.real(state),
                torch.imag(state),
                torch.tensor(srv, dtype=torch.float32),
            ]
        )
        return obs.unsqueeze(0)
