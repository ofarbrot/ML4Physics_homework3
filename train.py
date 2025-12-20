from o_model import PS_agent

def train_ps(env, gamma, eta, episodes):

    ps_agent = PS_agent(gamma=gamma, eta=eta, 
                        num_states=env.observation_space.n, 
                        num_actions=env.num_actions)
     
    max_t = 99    
    time_steps = []
    rewards = []
    
    for t in range(episodes):
    
        s = env.reset()[0]
        r = 0
        done = False
    
        ps_agent.reset_gmatrix()
        
        for j in range(max_t):
    
            action = ps_agent.deliberate(s)        
            s1, r, done, _, _ = env.step(action)
            ps_agent.learn(r)

            s = s1
            if done == True:
                break
                
        time_steps.append(j + 1)
        rewards.append(r)
    
    return ps_agent.hmatrix, ps_agent.gmatrix, time_steps, rewards
























'''def train(env, num_episodes, eta, gamma, max_t=99):
    steps = []
    rewards = []
    
    for i in range(num_episodes):
        # Reset environment and initialize all variables
        s = env.reset()[0]
        r = 0
        done = False
        
        for j in range(max_t):
            # "Oberserve" state and choos action acording to policy
            a = np.argmax(Q[s,:] + np.random.randn(1, env.action_space.n)*(1./(i+1)))
            
            #Get new state and reward from environment
            state, reward, done = env.step(a)

            # Update policy/knowlage 
            None

            s = state
            if done == True:
                break
        
        steps.append(j + 1)
        rewards.append(r)
      
    return Q, steps, rewards'''




# Something chat, dont think it is very useful
'''
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from helpers_HW.HW3.utils import is_valid_srv
from helpers_HW.HW3.ion_trap import IonTrapEnv


# ---------- utilities ----------



# ---------- policy network + agent ----------

class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, srv_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + srv_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, g], dim=-1)
        return self.net(x)


class model():
    """
    Required interface:
      - must be called model
      - must have pred(self, samples) -> list[torch.Tensor] of gate-index sequences
    """
    def __init__(self):
        self.policy = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env_kwargs = None  # saved so pred() can build fresh envs

    def _ensure_init(self, env):
        if self.policy is not None:
            return
        D = env.dim ** env.num_ions
        state_dim = 2 * D
        srv_dim = env.num_ions
        self.policy = PolicyNet(state_dim, srv_dim, env.num_actions).to(self.device)

    @torch.no_grad()
    def pred(self, samples):
        """
        samples: iterable of target SRVs, e.g. [[3,3,3], [2,2,2], ...]
        returns: list of torch.LongTensor sequences of gate indices
        """
        if self.policy is None or self.env_kwargs is None:
            raise RuntimeError("Call training_algorithm(agent, env) before pred().")

        self.policy.eval()
        preds = []

        for target_srv in samples:
            env = IonTrapEnv(**self.env_kwargs)
            env.goal = [list(target_srv)]

            state = env.reset()
            seq = []
            done = False

            while not done:
                s = _state_to_features(state).to(self.device)
                g = _srv_to_features(target_srv).to(self.device)
                logits = self.policy(s, g)
                action = int(torch.argmax(logits).item())

                state, reward, done = env.step(action)
                seq.append(action)
                if len(seq) >= env.max_steps:
                    break

            preds.append(torch.tensor(seq, dtype=torch.long))

        return preds


# ---------- training algorithm (REINFORCE) ----------

def training_algorithm(agent: model, env, episodes: int = 5000, lr: float = 3e-4,
                       entropy_bonus: float = 0.01, print_every: int = 250):
    """
    Sparse terminal reward, so we use REINFORCE.
    Trains across valid SRVs by sampling a target SRV each episode.
    """
    agent._ensure_init(env)

    # save kwargs so agent.pred can recreate envs later
    agent.env_kwargs = {
        "phases": copy.deepcopy(env.phases),
        "num_ions": int(env.num_ions),
        "dim": int(env.dim),
        "goal": copy.deepcopy(env.goal),
        "max_steps": int(env.max_steps),
    }

    valid_srvs = infer_valid_srvs_for_env(env)

    opt = optim.Adam(agent.policy.parameters(), lr=lr)

    # moving baseline for variance reduction
    baseline = 0.0
    beta = 0.95
    running_success = 0.0

    agent.policy.train()

    for ep in range(1, episodes + 1):
        target_srv = valid_srvs[np.random.randint(len(valid_srvs))]
        env.goal = [target_srv]

        state = env.reset()
        done = False

        logps = []
        ents = []

        steps = 0
        while not done and steps < env.max_steps:
            steps += 1
            s = _state_to_features(state).to(agent.device)
            g = _srv_to_features(target_srv).to(agent.device)

            logits = agent.policy(s, g)
            dist = torch.distributions.Categorical(logits=logits)

            action = dist.sample()
            logps.append(dist.log_prob(action))
            ents.append(dist.entropy())

            state, reward, done = env.step(int(action.item()))

        R = float(reward)  # terminal reward 0/1

        baseline = beta * baseline + (1 - beta) * R
        adv = R - baseline

        loss = -(adv * torch.stack(logps).sum()) - entropy_bonus * torch.stack(ents).sum()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.policy.parameters(), 1.0)
        opt.step()

        running_success = 0.99 * running_success + 0.01 * R

        if ep % print_every == 0:
            print(f"ep={ep:5d}  success~{running_success:.3f}  baseline={baseline:.3f}  last_target={target_srv}")

    return agent'''
