import yaml
import torch
import numpy as np

from multiheaded_actor import MultiHeadActor
from MORoverEnv import MORoverEnv

class MORoverInterface():
    def __init__(self, rover_config_filename):
        """
        Initialise the MOROverInterface class with its instance of the MOROverEnv Domain.
        Setup an internal reference to the rover config file
        """
        self.rover_env = MORoverEnv(rover_config_filename)
        with open(rover_config_filename, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
    
    # to perform a key-wise sum of two dicts
    def _keywise_sum(self, dict1, dict2):
        return {key: dict1.get(key, 0) + dict2.get(key, 0) for key in set(dict1) | set(dict2)}


    def rollout(self, mh_actor: MultiHeadActor, active_agents_indices: list, noisy_action=False):
        """
        Perform a rollout of a given multiheaded actor in the MORoverEnv domain.

        Parameters:
        - mh_actor (MultiHeadActor)
        - active_agents_indices (list): List of indices that specify which agents/heads in the MHA are active and will be a part of the rollout.
        - noisy_action (bool): Optional parameter to choose if action should be noisy or clean

        Returns:
        - rollout_trajectory (dict): Complete trajectory of the rollout with position, local reward, and action data of each agent.
        - global_reward (list): Reward vector that evaluates this MHA on each system-level objective.
        """

        ep_length = self.rover_env.get_ep_length()
        agent_locations = self.config['Agents']['starting_locs'] # set each agent to the starting location
        num_sensors = self.config['Agents']['num_sensors']
        observation_radii = self.config['Agents']['observation_radii']
        max_step_sizes = self.config['Agents']['max_step_sizes']
        
        cumulative_global_reward = {}  # Initialize cumulative global reward

        rollout_trajectory = {agent_idx : [] for agent_idx in active_agents_indices} # Initialise the episode's trajectory as a dict (of single agent trajectories)

        self.rover_env.reset() # reset the rover env

        for t in range(ep_length):
            observations = self.rover_env.generate_observations(agent_locations, num_sensors, observation_radii) # get each agent's observation at the current position
            observations_tensor = torch.tensor(observations, dtype=torch.float32) # Convert observations to a torch tensor (required for the actor model)

            local_rewards = self.rover_env.get_local_rewards(agent_locations) # get local rewards for this location configuration of agents
            
            observation_size = len(observations) // len(active_agents_indices) # size of each agent's observation

            agent_moves = []
            transitions = {} # transition = {'state' : [], 'action' : [], 'local_reward' : 0, 'next_state' : [], 'done' : False}

            for i, agent_idx in enumerate(active_agents_indices):
                # Extract the current observation for this agent
                agent_observation = observations_tensor[i].unsqueeze(0)  # Add batch dimension for the model

                if noisy_action:
                    action_tensor = mh_actor.noisy_action(observations_tensor[observation_size * i : observation_size * (i + 1)], active_agents_indices[i]) # add the agent's actions to the list
                else:
                    action_tensor = mh_actor.clean_action(observations_tensor[observation_size * i : observation_size * (i + 1)], active_agents_indices[i]) # add the agent's actions to the list

                action = action_tensor.squeeze(0).detach().numpy() # Convert action tensor to a numpy array without tracking gradient

                # Scale the action to comply with the agent's max step size
                norm = np.linalg.norm(action) # get the magnitude of the calculated move
                scaling_factor = (max_step_sizes[i] / norm) if norm > 0 else 0# the factor by which the moves should be scaled
                scaled_action = action * scaling_factor # multiply each member of the action by the scaling factor

                # Construct the transition dictionary for the current agent
                transitions[agent_idx] = {
                    'state': observations[observation_size * i : observation_size * (i + 1)],
                    'action': scaled_action,
                    'local_reward' : local_rewards[i],
                    'next_state': [],
                    'done': False
                }

                # Add scaled action to the list of agent moves
                agent_moves.append(scaled_action)
  
            agent_locations = self.rover_env.update_agent_locations(agent_locations, agent_moves, max_step_sizes) # get updated agent locations based on moves
            
            done = (t == ep_length - 1) # is the episode complete?

            # Get the global rweard and update the cumulative global reward
            global_reward = self.rover_env.get_global_rewards(rov_locations=agent_locations, timestep=t)
            cumulative_global_reward = self._keywise_sum(cumulative_global_reward, global_reward)

            # Prepare for next state's observations (after environment update)
            next_observations = self.rover_env.generate_observations(agent_locations, num_sensors, observation_radii)

            # Update each agent's transition dictionary with local reward, next state, and done
            for i, agent_idx in enumerate(active_agents_indices):
                transitions[agent_idx]['next_state'] = next_observations[observation_size * i : observation_size * (i + 1)]
                transitions[agent_idx]['done'] = done

                # Append the transition to the agent's trajectory
                rollout_trajectory[agent_idx].append(transitions[agent_idx])


        return rollout_trajectory, cumulative_global_reward