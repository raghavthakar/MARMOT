import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import random
import itertools

from multiheaded_actor import MultiHeadActor

# np.random.seed(2024)
# torch.manual_seed(2024)
# random.seed(2024)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        hidden_size=128,
        num_heads=1,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):

        self.actor = MultiHeadActor(state_dim, action_dim, hidden_size, num_heads=num_heads)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def set_new_actor(self, new_actor):
        self.actor = copy.deepcopy(new_actor)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.total_it = 0

    def train(self, replay_buffer, batch_size=25, agent_id=-1):
        self.total_it += 1
        
        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        


        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target.clean_action(state=next_state, head=agent_id) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor.clean_action(state=state, head=agent_id)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return self.actor

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        assert self.size >= batch_size, "Batch size larger than experiences in the replay buffer" + str(self.size)
        return (
            torch.FloatTensor(self.state[ind]),
            torch.FloatTensor(self.action[ind]),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.not_done[ind])
        )

def add_experiences_to_replay_buffers(rollout_trajectory, rep_buffs, active_agents_indices):
    for agent_idx in active_agents_indices:
        replay_buffer = rep_buffs[agent_idx]
        # Iterate over the transitions for this agent.
        for transition in rollout_trajectory[agent_idx]:
            # Convert tensors to numpy arrays, if needed.
            state = transition['state']
            if torch.is_tensor(state):
                state = state.detach().cpu().numpy()

            action = transition['action']
            if torch.is_tensor(action):
                action = action.detach().cpu().numpy()

            next_state = transition['next_state']
            if torch.is_tensor(next_state):
                next_state = next_state.detach().cpu().numpy()

            # The reward is stored under 'local_reward'
            reward = transition['local_reward']

            # The buffer expects 'done' as 0 or 1, where done==True -> 1.
            done = 1 if transition['done'] else 0

            # Now add the transition to the replay buffer.
            replay_buffer.add(state, action, next_state, reward, done)

if __name__ == "__main__":
    import MORoverInterface

    interface = MORoverInterface.MORoverInterface("config/MORoverEnvConfig.yaml")
    td3 = TD3(state_dim=12, action_dim=interface.get_action_size(), max_action=1.0, num_heads=1)
    rep_buffs = [ReplayBuffer(state_dim=12, action_dim=interface.get_action_size(), max_size=50_000) for _ in range(1)]


    for i in range(3000):
        traj, g = interface.rollout(td3.actor, [0], noisy_action=True, noise_std=0.15)
        add_experiences_to_replay_buffers(traj, rep_buffs=rep_buffs, active_agents_indices=[0])

        for _ in range(25):
            td3.train(replay_buffer=rep_buffs[0], agent_id=0)
        
        

        traj, g = interface.rollout(td3.actor, [0])

        # if(i > 40 and g[0] != 1):
        #     print(traj[0][-1]["location"])
        #     exit()

        if(i % 5 == 0):
            print("Epoch:", i, "G:", g, "Final Location:", traj[0][-1]["location"])
    
    print("Finished Training")

    traj, g = interface.rollout(td3.actor, [0])
    print("G:", g)


    

        
    