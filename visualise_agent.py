# visualise_agent.py

from env import ContinuousMazeEnv
from DQN_model import Qnet
import torch

env = ContinuousMazeEnv(render_mode="human")
obs, _ = env.reset()

dqn = Qnet(dim_actions=4, dim_states=2)
dqn.load_state_dict(torch.load("dqn.pth"))

for _ in range(10):
    done = False
    while not done:
        env.render()
        action = dqn(torch.from_numpy(obs).float())
        obs, reward, done, _, _ = env.step(action.argmax().item())
    obs, _ = env.reset()

env.close()
