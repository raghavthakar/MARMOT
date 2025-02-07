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
from td3 import ReplayBuffer, TD3
import MORoverInterface

np.random.seed(2024)
torch.manual_seed(2024)
random.seed(2024)

interface = MORoverInterface.MORoverInterface("config/MORoverEnvConfig.yaml")
td3 = TD3(state_dim=12, action_dim=interface.get_action_size(), max_action=1.0, num_heads=2, num_critics=2)
rep_buffs = [ReplayBuffer(state_dim=12, action_dim=interface.get_action_size(), max_size=50_000) for _ in range(2)]
nsga = nsgaii.NSGAII(alg_config_filename="config/MARMOTConfig.yaml", rover_config_filename="config/MORoverEnvConfig.yaml", 
                                replay_buffers=rep_buffs)

for i in range(1200):
    pop, roster_wise_team_combs, roster_wise_team_fits, chosen_roster, champion_indices = nsga.evolve()

    # if i > 10:
    #     td3.set_new_actor(chosen_roster)
    #     for _ in range(625):
    #         td3.train(replay_buffer=rep_buffs[0], agent_id=0)
    #         td3.train(replay_buffer=rep_buffs[1], agent_id=1)

    if(i % 5 == 0):
        traj, g = interface.rollout(chosen_roster, [0, 1])
        print("Epoch:", i, "G:", g, "Final Location:", traj[0][-1]["location"])

print("Finished Training")

traj, g = interface.rollout(td3.actor, [0, 1])
print("G:", g)