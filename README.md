# MARMOT: MultiAgent Rosters for Multi-Objective Coordination 
Code for the ROB 538: Multiagent Systems class project "Learning a Roster of Policies for Pareto-Optimal Coordination".

Authors: Raghav Thakar (thakarr@oregonstate.edu), and Siddarth Iyer (viswansi@oregonstate.edu).

Please read the paper for a thorough technical description of the project, as well as results from our experiments: [Learning a Roster of Policies for Pareto-Optimal Coordination](MARMOT_PrePrint.pdf).

## Description
This paper presents a novel approach to learning multiagent control policies that allow a team of agents to succeed in multi-objective coordination tasks. A key challenge in multi-objective settings is to account for _trade-offs_ among objectives, which generally generally give rise to several, _Pareto-optimal_ solutions instead of a single optimal solution. MARMOT explicitly addresses this challenge of learning multiple, equally optimal multiagent policies by learning a _roster_ of policies. Teams of agents formed by sampling subsets of policies from this roster may then demonstrate strikingly different behaviours, providing a wide coverage of trade-off performances among the objectives.

To achieve this, we leverage the Multiagent Evolutionary Reinforcement Learning (MERL) paradigm, which uses an evolutionary algorithm to train using the sparse, team-level global reward, while an off-policy reinforcement learning algorithm trains each agent using a dense, local reward.

## How to run this locally
1. Create a new conda virtual environment
2. Clone this repository
3. Install all the required dependencies listed in `environment.yml`
4. Navigate to the repository, and replace the paths in `MARMOT.py` with the paths to the config files in your system
5. Run the experiment by doing: `python MARMOT.py`
