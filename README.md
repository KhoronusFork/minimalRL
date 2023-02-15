# minimalRL-pytorch

Updated code of minimalRL-pytorch

Original code at: https://github.com/seungeunrho/minimalRL

Additional models from pull requests. Credit at the top of the file.

## Original version - for gym (CPU)
original_gym  
The code requires gym==0.19.0  
Please check the readme.md in the folder for more information.  

## Original version - adapted for gymnasium (CPU)
Updated the code to work with gymnasium package.  
Only ppo-continuous and sac are updated.  
Note: Tested with Pendulum-v1
'''batch
minimalRL/original_gymnasium$ python ppo-continuous.py  
'''

## DRL-code-pytorch - adapted for gymnasium (CPU)  
PPO-continuous
Note: Tested with HalfCheetah-v4  
'''batch
minimalRL/DRL-code-pytorch/5.PPO-continuous$ python PPO_continuous_main.py
'''

## MinimalRL
A different implementation of popular RL algorithms (may not converge).  
'''batch
minimalRL$ python ppo-continuous-improved.py
'''
Note: Tested with Pendulum-v1. It renders the environment after 130 episodes.  

### Render
dqn visualizes the render during the training (slow).

### Note
The algorithms in the root folders are tested with gymnasium and where possible execute on a GPU device.  
The algorithms are for a discrete space, if not indicated differently.  
Continuous space:  
- ddpg, ppo-continuous-improved, ppo-continuous, sac  
- ppo-continuous : Does not converge :/  
r2d2 : Contiguous error. TODO: Fix  

# From the original readme.md


Implementations of basic RL algorithms with minimal lines of codes! (PyTorch based)

* Each algorithm is complete within a single file.

* Length of each file is up to 100~150 lines of codes.

* Every algorithm can be trained within 30 seconds, even without GPU.

* Envs are fixed to "CartPole-v1". You can just focus on the implementations.



## Algorithms
1. [REINFORCE](https://github.com/seungeunrho/minimalRL/blob/master/REINFORCE.py) (67 lines)
2. [Vanilla Actor-Critic](https://github.com/seungeunrho/minimalRL/blob/master/actor_critic.py) (98 lines)
3. [DQN](https://github.com/seungeunrho/minimalRL/blob/master/dqn.py) (112 lines,  including replay memory and target network)
4. [PPO](https://github.com/seungeunrho/minimalRL/blob/master/ppo.py) (119 lines,  including GAE)
5. [DDPG](https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py) (145 lines, including OU noise and soft target update)
6. [A3C](https://github.com/seungeunrho/minimalRL/blob/master/a3c.py) (129 lines)
7. [ACER](https://github.com/seungeunrho/minimalRL/blob/master/acer.py) (149 lines)
8. [A2C](https://github.com/seungeunrho/minimalRL/blob/master/a2c.py) (188 lines)
9. [SAC](https://github.com/seungeunrho/minimalRL/blob/master/sac.py) (171 lines) added!! 
10. [PPO-Continuous](https://github.com/seungeunrho/minimalRL/blob/master/ppo-continuous.py) (161 lines) added!!
11. [Vtrace](https://github.com/seungeunrho/minimalRL/blob/master/vtrace.py) (137 lines) added!!


## Dependencies
1. PyTorch
2. OpenAI GYM

## Usage
```bash
# Works only with Python 3.
# e.g.
python3 REINFORCE.py
python3 actor_critic.py
python3 dqn.py
python3 ppo.py
python3 ddpg.py
python3 a3c.py
python3 a2c.py
python3 acer.py
python3 sac.py
```
