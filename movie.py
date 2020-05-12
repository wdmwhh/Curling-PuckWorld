# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:30:15 2020

@author: 张
"""

import os
import matplotlib.pyplot as plt
import torch

from dqn import QFunc

from curling import Curling
from utils import egreedy_strategy


env_w, env_h = 100.0, 100.0
num_step = 300
num_action = 4
action = [(5, 5), (-5, 5), (5, -5), (-5, -5)]
policy_NN = QFunc(6, 32, 4) 
policy_NN.load_state_dict(torch.load('sarsa_policy.pth'))
fig = plt.figure(0)
fig.suptitle("Curling")

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1,1,1)
ax.set_aspect(1)
major_locator = plt.MultipleLocator(2)
ax.xaxis.set_major_locator(major_locator)
ax.yaxis.set_major_locator(major_locator)
ax.tick_params(axis='both', which='major', labelsize=7)
ax.grid(linewidth=2)
ln, = ax.plot([], [], 'r-', animated=False)
ax.set_xlim(0, env_w)
ax.set_ylim(0, env_h)


frame_number = 1
for _ in range(10):
    curling = Curling()
    target = curling.target
    p_target = ax.scatter(target[0], target[1], marker='*', s=100, color="b")
    for step in range(num_step):
        position = curling.position
        p_position = ax.scatter(position[0], position[1], marker='p', s=200, color="r")
        fig.savefig('images/_tmp%04d.png' % frame_number)
        frame_number += 1
        p_position.remove()
        s = curling.state
        with torch.no_grad():
            q_s = policy_NN(torch.tensor(s, dtype=torch.float32).flatten())
        a = egreedy_strategy(q_s, num_action) 
        curling.action(*action[a])
        for interval in range(10):
            curling.cnt += 1
            curling.move(0.01)
    p_target.remove()

print(os.system("ffmpeg -framerate 10 -i images/_tmp%04d.png  -c:v libx264 -r 10 -pix_fmt yuv420p sarsa_out.mp4"))
