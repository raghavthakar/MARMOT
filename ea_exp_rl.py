import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import random
import itertools
import nsgaii

from multiheaded_actor import MultiHeadActor
from td3 import ReplayBuffer, add_experiences_to_replay_buffers, TD3
import MORoverInterface

np.random.seed(2024)
torch.manual_seed(2024)
random.seed(2024)

interface = MORoverInterface.MORoverInterface("config/MORoverEnvConfig.yaml")
td3 = TD3(state_dim=12, action_dim=interface.get_action_size(), max_action=1.0, num_heads=1)
rep_buffs = [ReplayBuffer(state_dim=12, action_dim=interface.get_action_size(), max_size=50_000) for _ in range(1)]
nsga = nsgaii.NSGAII(alg_config_filename="config/MARMOTConfig.yaml", rover_config_filename="config/MORoverEnvConfig.yaml", 
                                replay_buffers=rep_buffs)

for i in range(120):
    # traj, g = interface.rollout(td3.actor, [0], noisy_action=True, noise_std=0.15)
    # add_experiences_to_replay_buffers(traj, rep_buffs=rep_buffs, active_agents_indices=[0])
    nsga.evolve()
    traj, g = interface.rollout(td3.actor, [0], noisy_action=True, noise_std=0.15)
    add_experiences_to_replay_buffers(traj, rep_buffs, [0])

    for _ in range(625):
        td3.train(replay_buffer=rep_buffs[0], agent_id=0)
    
    


    if(i % 5 == 0):
        traj, g = interface.rollout(td3.actor, [0])
        print("Epoch:", i, "G:", g, "Final Location:", traj[0][-1]["location"])

print("Finished Training")

traj, g = interface.rollout(td3.actor, [0])
print("G:", g)