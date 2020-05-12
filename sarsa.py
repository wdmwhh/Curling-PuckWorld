#!/usr/bin/env python
# -*- coding: utf-8 -*-

    
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim

from curling import Curling
from utils import plot, egreedy_strategy


# s = (pos, target_pos, v)

class QFunc(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, s):
        x = self.predictor(s)
        return x
    
    
    def fit(self, s1_batch, 
                  a1_batch,
                  target, 
                  epochs=1,
                  learning_rate=1e-4):
                  
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        if epochs < 1:
            epochs = 1

        for t in range(epochs):
            q_s1 = self.forward(s1_batch)
            q_s1_a = q_s1.gather(1, a1_batch.unsqueeze(1)).squeeze()  
            loss = criterion(q_s1_a, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return loss.item()
    

class Memory(object):

    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []
        self.pos = 0

    def push(self, item):
        if len(self.memory) < self.capacity:
            self.memory.append(())
        self.memory[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    
def main():
    target_update = 2
    capacity = 50000
    batch_size = 100
    eps = 0.9
    min_eps = 0.2
    eps_factor = 0.3
    lr = 1e-3
    action = [(5, 5), (-5, 5), (5, -5), (-5, -5)]
    gamma = 0.9
    num_epoch = 5000
    num_step = 300
    num_action = len(action)

    policy_NN = QFunc(6, 32, num_action) 
    target_NN = QFunc(6, 32, num_action) 
    target_NN.load_state_dict(policy_NN.state_dict())
    target_NN.eval()
    # criterion = nn.SmoothL1Loss()
    # optimizer = optim.RMSprop(policy_NN.parameters()) 
    memory = Memory(capacity)
    
    loss_list = []
    reward_list = []
    for epoch in range(num_epoch):
        if epoch == 4000:
            lr *= 0.1
        curling = Curling()
        loss_tot = 0
        reward_tot = 0
        with torch.no_grad():
            q_s = policy_NN(torch.tensor(curling.state, dtype=torch.float32).flatten())
        a_pre = egreedy_strategy(q_s, num_action, eps)
        for step in range(num_step):
            s1 = curling.state
            with torch.no_grad():
                q_s = policy_NN(torch.tensor(s1, dtype=torch.float32).flatten())
            a1 = a_pre
            r = curling.reward(curling.position)
            reward_tot += r
            curling.action(*action[a1])
            for interval in range(10):
                curling.cnt += 1
                curling.move(0.01)
            s2 = curling.state
            # eps *= eps_decay
            slope = (min_eps - 1.) / (num_epoch * eps_factor)
            eps = max(min_eps, slope * epoch + 1.)
            with torch.no_grad():
                q_s = policy_NN(torch.tensor(s2, dtype=torch.float32).flatten())
            a2 = egreedy_strategy(q_s, num_action, eps)
            a_pre = a2

            # update weight
            
            memory.push((s1, a1, r, s2, a2))
            if step==num_step-1:
                memory.push((s2, a2, 
                    curling.reward(curling.position), ((-1., )*2, )*3, 0))
            
            if len(memory) < batch_size: continue

            with torch.enable_grad():
                s1_batch, a1_batch, r_batch, s2_batch, a2_batch = zip(*memory.sample(batch_size))
                s1_batch = torch.tensor(s1_batch, dtype=torch.float32).flatten(1)
                a1_batch = torch.tensor(a1_batch)
                r_batch = torch.tensor(r_batch, dtype=torch.float32)
                s2_batch = torch.tensor(s2_batch, dtype=torch.float32).flatten(1)
                a2_batch = torch.tensor(a2_batch)
                with torch.no_grad():
                    q_s2 = target_NN(s2_batch)
                    q_s2_a = q_s2.gather(1, a2_batch.unsqueeze(1)).squeeze().detach()  
                    q_s2_a *= s2_batch[:, 0]>0

                loss = policy_NN.fit(s1_batch,
                                   a1_batch,
                                   r_batch + q_s2_a * gamma, 
                                   learning_rate = lr,
                                   epochs = 3)
        
                loss_tot += loss
               

        if (epoch+1) % target_update == 0:
            target_NN.load_state_dict(policy_NN.state_dict())
        print('Epoch', epoch+1, ': loss', loss_tot/num_step, ', reward', reward_tot)
        loss_list.append(loss_tot/num_step)
        reward_list.append(reward_tot)
    
    torch.save(policy_NN.state_dict(), 'sarsa_policy.pth')
    curling = Curling()
    episodic = [curling.position]
    for step in range(num_step):
        s = curling.state
        with torch.no_grad():
            q_s = policy_NN(torch.tensor(s, dtype=torch.float32).flatten())
        a = egreedy_strategy(q_s, num_action) 
        curling.action(*action[a])
        for interval in range(10):
            curling.cnt += 1
            curling.move(0.01)
        episodic.append(curling.position)
    
    x = list(range(1, num_epoch+1))
    plt.figure(figsize=(10, 20))
    plt.subplot(121)
    plt.xlabel('epoch')
    plt.xticks(list(range(0, num_epoch+1, 500)))
    plt.ylabel('loss')
    plt.plot(x, loss_list, color="r",linestyle = "-")
    plt.subplot(122)
    plt.xlabel('epoch')
    plt.xticks(list(range(0, num_epoch+1, 500)))
    plt.ylabel('reward')
    plt.plot(x, reward_list, color="y",linestyle = "-")
    plt.tight_layout()
    plt.savefig('sarsa_curves.png')
    plt.show()
    
    
    # plt.figure(figsize=(10, 20))
    # plt.subplot(121)
    # plt.xlabel('epoch')
    # plt.xticks(list(range(0, num_epoch+1, 500)))
    # plt.ylabel('loss')
    # plt.plot(x, loss_20, color="r",linestyle = "-")
    # plt.subplot(122)
    # plt.xlabel('epoch')
    # plt.xticks(list(range(0, num_epoch+1, 500)))
    # plt.ylabel('reward')
    # plt.plot(x, reward_20, color="y",linestyle = "-")
    # plt.tight_layout()
    # plt.savefig('sarsa_curves_20average.png')
    # plt.show()
    plot(episodic, curling.target)
    

if __name__ == '__main__':
    main()