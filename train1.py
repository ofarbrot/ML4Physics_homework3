# train.py
import torch
import random
import numpy as np
from collections import deque
from helpers_HW.HW3.ion_trap import IonTrapEnv
from model1 import QNetwork


# ------------------
# Hyperparameters
# ------------------
gamma = 0.99
lr = 1e-3
batch_size = 64
buffer_size = 10_000
episodes = 5000
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 0.995


# ------------------
# Replay Buffer
# ------------------
class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# ------------------
# Environment setup
# ------------------
phases = {
    'pulse_angles': [np.pi / 2],
    'pulse_phases': [np.pi / 2],
    'ms_phases': [-np.pi / 2],
}

num_ions = 3
num_actions = 7
state_dim = 54 + 3

q_net = QNetwork(state_dim, num_actions)
optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
buffer = ReplayBuffer(buffer_size)

epsilon = epsilon_start


def make_obs(state, srv):
    state = torch.tensor(state, dtype=torch.complex64).flatten()
    return torch.cat(
        [
            torch.real(state),
            torch.imag(state),
            torch.tensor(srv, dtype=torch.float32),
        ]
    )


# ------------------
# DQN update
# ------------------
def update(q_net, optimizer, batch, gamma):
    obs, actions, rewards, next_obs, dones = zip(*batch)

    obs = torch.stack(obs)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    next_obs = torch.stack(next_obs)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_vals = q_net(obs).gather(1, actions.unsqueeze(1)).squeeze()
    with torch.no_grad():
        max_next_q = q_net(next_obs).max(1).values
        target = rewards + gamma * (1 - dones) * max_next_q

    loss = torch.nn.functional.mse_loss(q_vals, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    

# ------------------
# Training loop
# ------------------
for ep in range(episodes):
    srv = torch.randint(1, 4, (3,))
    env = IonTrapEnv(
        phases=phases,
        num_ions=num_ions,
        goal=[list(srv)],
        max_steps=10,
    )

    state = env.reset()
    obs = make_obs(state, srv)
    if ep == 0:
        print("Obs shape:", obs.shape)

    for t in range(10):
        if random.random() < epsilon:
            action = random.randint(0, num_actions - 1)
        else:
            with torch.no_grad():
                action = torch.argmax(q_net(obs.unsqueeze(0))).item()

        next_state, reward, done = env.step(action)
        next_obs = make_obs(next_state, srv)

        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs

        if len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)
            update(q_net, optimizer, batch, gamma)

        if done:
            break

    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    if ep % 500 == 0:
        print(f"Episode {ep}, epsilon={epsilon:.3f}")

torch.save(q_net.state_dict(), "qnet.pt")
print("Training complete. Model saved to qnet.pt")





