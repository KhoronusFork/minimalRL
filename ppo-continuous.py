import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

#Hyperparameters
learning_rate  = 0.0003
gamma           = 0.9
lmbda           = 0.9
eps_clip        = 0.2
K_epoch         = 10
rollout_len    = 3
buffer_size    = 30
minibatch_size = 32

class PPO(nn.Module):
    def __init__(self, device, input_size, output_size):
        super(PPO, self).__init__()
        self.data = []
        self.device = device
        
        self.fc1   = nn.Linear(input_size,128)
        self.fc_mu = nn.Linear(128,output_size)
        self.fc_std  = nn.Linear(128,output_size)
        self.fc_v = nn.Linear(128,output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        mu = 2.0*torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
        data = []

        for j in range(buffer_size):
            for i in range(minibatch_size):
                rollout = self.data.pop()
                s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

                for transition in rollout:
                    s, a, r, s_prime, prob_a, done = transition
                    
                    s_lst.append(s)
                    a_lst.append([a])
                    r_lst.append([r])
                    s_prime_lst.append(s_prime)
                    prob_a_lst.append([prob_a])
                    done_mask = 0 if done else 1
                    done_lst.append([done_mask])

                s_batch.append(s_lst)
                a_batch.append(a_lst)
                r_batch.append(r_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)
                done_batch.append(done_lst)


            vv = []
            for i in range(0, len(a_batch)):
                v = torch.stack(a_batch[i][0])
                vv.append(v)
            vvv = torch.stack(vv)

            vv1 = []
            for i in range(0, len(prob_a_batch)):
                v = torch.stack(prob_a_batch[i][0])
                vv1.append(v)
            vvv1 = torch.stack(vv)

                        #torch.tensor(a_batch, dtype=torch.float).to(self.device), \
                        #torch.stack(a_batch).to(self.device), \
                        #torch.tensor(prob_a_batch, dtype=torch.float).to(self.device)
            mini_batch = torch.tensor(np.array(s_batch), dtype=torch.float).to(self.device), \
                        vvv.to(self.device), \
                        torch.tensor(np.array(r_batch), dtype=torch.float).to(self.device), \
                        torch.tensor(np.array(s_prime_batch), dtype=torch.float).to(self.device), \
                        torch.tensor(done_batch, dtype=torch.float).to(self.device), \
                        vvv1.to(self.device)
            data.append(mini_batch)

        return data

    def calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_prob = mini_batch
            with torch.no_grad():
                td_target = r + gamma * self.v(s_prime) * done_mask
                delta = td_target - self.v(s)
            delta = torch.flip(delta, (0,)) #delta

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()

            vv = []
            for i in range(0, len(advantage_lst)):
                v = advantage_lst[i][0]
                vv.append(v)
            vvv = torch.stack(vv)
            advantage = vvv.to(self.device)
            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage))

        return data_with_adv

        
    def train_net(self):
        if len(self.data) == minibatch_size * buffer_size:
            data = self.make_batch()
            data = self.calc_advantage(data)

            for i in range(K_epoch):
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch
                    advantage = advantage.unsqueeze(1)

                    mu, std = self.pi(s, softmax_dim=1)
                    dist = Normal(mu, std)
                    log_prob = dist.log_prob(a)
                    ratio = torch.exp(log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))

                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target)

                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimization_step += 1
        

def action_space_dim(env):
    if type(env.action_space) == gym.spaces.discrete.Discrete:
        return env.action_space.n
    else:
        return env.action_space.shape[0]
def observation_space_dim(env):
    return env.observation_space.shape[0]

def main():
    #env = gym.make('Pendulum-v1', render_mode = 'rgb_array')
    env = gym.make('HalfCheetah-v4', render_mode = 'rgb_array')
    if torch.cuda.is_available():
        device= 'cuda:0'
    else:
        device = 'cpu'
    actionspace = action_space_dim(env)
    observationspace = observation_space_dim(env)
    print('actionspace:{}'.format(actionspace))
    print('observationspace:{}'.format(observationspace))
    model = PPO(device, observationspace, actionspace).to(device)
    score = 0.0
    print_interval = 20
    rollout = []

    import cv2
    for n_epi in range(10000):
        observation, info = env.reset()
        terminated = False

        iterations = 0
        while not terminated:
            if iterations >= 100:
                break
            iterations += 1
            for t in range(rollout_len):
                mu, std = model.pi(torch.from_numpy(observation).float().to(device))
                dist = Normal(mu, std)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                observation_prime, reward, terminated, truncated, info = env.step(action.cpu().numpy())
                rollout.append((observation, action, reward/10.0, observation_prime, log_prob, terminated))
                if len(rollout) == rollout_len:
                    model.put_data(rollout)
                    rollout = []

                observation = observation_prime
                score += reward
                if terminated:
                    break

                # Render into buffer.
                if n_epi > 260:
                    frame = env.render()
                    cv2.imshow('frame', frame)
                    cv2.waitKey(1)


            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}, opt step: {}".format(n_epi, score/print_interval, model.optimization_step))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()