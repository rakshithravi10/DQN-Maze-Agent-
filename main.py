# Imports:
# --------
import torch
from env import ContinuousMazeEnv
from DQN_model import Qnet
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import ReplayBuffer, train

# User definitions:
# -----------------
train_dqn = True
test_dqn = True
render = True  # Use True for visual rendering during training/testing

# Define env attributes (environment specific)
dim_actions = 4  # up, down, left, right
dim_states = 2   # [x, y] position of agent

# Hyperparameters:
# ----------------
learning_rate = 0.003
gamma = 0.98 # Discount factor for future rewards
buffer_limit = 50_000
batch_size = 32
num_episodes = 12_000
max_steps = 200 # Max steps per episode

# Main:
# -----
if train_dqn:
    env = ContinuousMazeEnv(render_mode="human" if render else None)

    # Initialize the Q Net and the Q Target Net
    q_net = Qnet(dim_actions=dim_actions, dim_states=dim_states)
    q_target = Qnet(dim_actions=dim_actions, dim_states=dim_states)
    q_target.load_state_dict(q_net.state_dict())

    # Initialize the Replay Buffer
    memory = ReplayBuffer(buffer_limit=buffer_limit)

    print_interval = 10  # Printing every 10 episodes
    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
    rewards = []

    epsilon = 1.0
    consecutive_successes = 0

    for n_epi in range(num_episodes):
        epsilon = max(0.1, epsilon * 0.997)  # Slower decay
        s, _ = env.reset()
        done = False
        episode_reward = 0.0

        for _ in range(max_steps):
            a = q_net.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, _, _ = env.step(a)
            r = max(min(r, 50.0), -20.0)  # Clip reward
            memory.put((s, a, r, s_prime, 0.0 if done else 1.0))
            s = s_prime
            episode_reward += r
            if done:
                break

        if epsilon <= 0.1:
            if episode_reward >= 45.0:
                consecutive_successes += 1
            else:
                consecutive_successes = 0

        if consecutive_successes >= 100:
            print(f"✅ Agent converged at episode {n_epi}")
            break

        if memory.size() > 2000:
            train(q_net, q_target, memory, optimizer, batch_size, gamma)

        if n_epi % 5 == 0:
            q_target.load_state_dict(q_net.state_dict())

        if n_epi % 10 == 0 and n_epi != 0:
            print(f"n_episode :{n_epi}, Reward : {episode_reward:.2f}, "
                  f"Successes : {consecutive_successes}, eps : {epsilon:.3f}")

        rewards.append(episode_reward)

    env.close()
    torch.save(q_net.state_dict(), "dqn.pth")

    plt.plot(rewards, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.title('Training Curve')
    plt.legend()
    plt.savefig("training_curve.png")
    plt.show()

# Test:
if test_dqn:
    print("Testing the trained DQN: ")
    env = ContinuousMazeEnv(render_mode="human" if render else None)

    dqn = Qnet(dim_actions=dim_actions, dim_states=dim_states)
    dqn.load_state_dict(torch.load("dqn.pth"))

    for _ in range(10):
        s, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            action = dqn(torch.from_numpy(s).float())
            s_prime, reward, done, _, _ = env.step(action.argmax().item())
            s = s_prime
            episode_reward += reward
            if done:
                break

        print(f"Episode reward: {episode_reward}")
    env.close()
