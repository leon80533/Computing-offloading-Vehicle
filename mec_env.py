import math
import random
import numpy as np
from random import choice
import queue
import torch
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
import gym


class Env():
    def __init__(self, num_uv, num_vfs, time, max_ddl):

        self.n_uv = num_uv
        self.n_vfs = num_vfs
        self.duration = 0.1
        self.time_limit = time
        self.action_space = None
        self.n_actions = 1 + self.n_vfs
        self.n_features = 1 + 1 + 2 + 1 + self.n_vfs
        self.state_dim = self.n_uv * (1 + 1 + 2 + 1 + self.n_vfs)
        loc_MEC = [500, 500 * self.n_vfs]

        # UV計算能力：2.5 Gigacycles/s  * duration
        self.comp_cap_uv = 2.5 * np.ones(self.n_uv) * self.duration
        # VF計算能力： 15 Gigacycles/s  * duration
        self.comp_cap_vfs = 15 * np.ones(self.n_vfs) * self.duration
        # 傳輸速率： Mbps * duration
        # self.tran_cap_uv = 18 * np.ones([self.n_uv, self.n_vfs]) * self.duration

        # Processing density (CPU cycle per bit)
        self.comp_density = 0.297 * np.ones([self.n_uv])
        # self.device = torch.device('cpu')

        # 容忍task delay
        self.max_delay = max_ddl

        # Action: 0:local ; 1:VFS_1 ; 2:VFS_2; ...
        self.n_action_space = 1 + self.n_vfs
        # State: 每個uv * 需要處理的task + 現在的task + vehicle coordinate + 是否offload + 已經compute off laoding 的對應的vfs
        # state[0]: accumulated task, state[1]: task, statestate[2:4] = coord, state[4]: whether offload, state[5:n+1]: vfs
        self.state_space_dim = (self.n_uv, 1 + 1 + 2 + 1 + self.n_vfs)

        self.state = np.zeros(self.state_space_dim)

        # Time count
        self.time_count = 0
        self.done = False

    def create_task(self):
        # 隨機生成task: 每個時間有0.5的prob生成task，其大小在2~5M之间
        task_prob = 0.5
        task_size = np.arange(2, 5, 0.1)
        task_list = np.zeros((self.n_uv, 3))
        for i in range(len(task_list)):
            if np.random.random() < task_prob:
                task_list[i, 0] = choice(task_size)
                task_list[i, 1] = self.time_count
                task_list[i, 2] = self.time_count + self.max_delay
        return task_list

    def reset(self):
        # location of uv
        self.task = self.create_task()
        loc_uv_list = np.random.randint(0, 1001, size=[self.n_uv, 2])

        # cumulated task
        self.state[:, 0] = 0
        # current task
        self.state[:, 1] = self.task[:, 0]
        # uv location
        self.state[:, 2:4] = loc_uv_list

        self.state[:, 5:] = self.comp_cap_vfs

        self.time_count = 0

        self.Queue_uv_comp = [queue.Queue() for i in range(self.n_uv)]
        self.Queue_vfs_comp = [[queue.Queue()
                                for i in range(self.n_vfs)]] * self.n_uv

        # current processing task 0: size, 1: time
        self.task_on_process_local = np.zeros((self.n_uv, 2))
        self.task_on_process_vfs = np.zeros((self.n_uv, self.n_vfs, 2))
        self.complete_task = 0
        self.drop_uv = 0
        self.drop_vf = 0
        self.done = False
        # self.action_space = Discrete(self.)
        return self.state

    def step(self, action):

        reward = 0
        reward_penalty = 20

        # init Update (queue task)
        for i, j in enumerate(action):
            if j == 0:
                # put task into accumulated task
                self.state[i, 1] += self.state[i, 0]

                # 放入uv queue task 大小及時間
                self.Queue_uv_comp[i].put((self.state[i, 0], self.task[i, 1]))
                # Not offloading = 0
                self.state[i, 4] = 0
            if j != 0:
                self.state[i, j+5] = self.state[i, 0]
                self.state[i, 4] = 1

                # 放入vfs_queue task 大小及時間
                self.Queue_vfs_comp[i][j -
                                       1].put((self.state[i, 0], self.task[i, 1]))

        for index in range(self.n_uv):
            # task done in current duration
            uv_comp_cap = self.comp_cap_uv[index]
            uv_comp_density = self.comp_density[index]
            # task deal in this vehicle in this time slot
            uv_task_deal = uv_comp_cap/uv_comp_density

            # empty current task list but task_queue is not empyt
            if not np.all(self.task_on_process_local[index]) \
                    and (not self.Queue_uv_comp[index].empty()):
                cur_process = self.Queue_uv_comp[index].get()
                self.task_on_process_local[index] = cur_process

            # process the task:
            while not self.Queue_uv_comp[index].empty():

                remain_task = self.task_on_process_local[index][0] - \
                    uv_task_deal
                time_exceed = self.task_on_process_local[index][1] + \
                    self.max_delay > self.time_count

                if remain_task < 0 and not time_exceed:

                    # reward = tm if local
                    reward += self.max_delay
                    uv_task_deal = -remain_task
                    # 完成的task + 1
                    self.complete_task += 1
                    cur_process = self.Queue_uv_comp[index].get()
                    self.task_on_process_local[index] = cur_process

                if remain_task > 0 and not time_exceed:

                    self.task_on_process_local[index][0] = remain_task
                    break

                if time_exceed:

                    self.drop_uv += 1
                    cur_process = self.Queue_uv_comp[index].get()
                    self.task_on_process_local[index] = cur_process
                    reward -= reward_penalty

        # count task num on each VFS
        for vfs_index in range(self.n_vfs):
            for uv_index in range(self.n_uv):
                task_on_vfs = np.zeros(self.n_vfs)
                if (not self.Queue_vfs_comp[uv_index][vfs_index].empty())\
                        or self.task_on_process_vfs[uv_index][vfs_index][0] > 0:
                    task_on_vfs[vfs_index] += 1

        # VFS
        for uv_index in range(self.n_vfs):
            uv_comp_density = self.comp_cap_vfs[uv_index]

            for vfs_index in range(self.n_vfs):
                vfs_comp_cap = self.comp_cap_vfs[vfs_index]
                num_task_on_vf = task_on_vfs[vfs_index]
                # task deal in vfs in current time slot
                vfs_task_deal = vfs_comp_cap / \
                    uv_comp_density / max(num_task_on_vf, 1)

                if not np.all(self.task_on_process_vfs[uv_index][vfs_index]) \
                        and (not self.Queue_vfs_comp[uv_index][vfs_index].empty()):
                    cur_process = self.Queue_vfs_comp[uv_index][vfs_index].get(
                    )
                    self.task_on_process_vfs[uv_index][vfs_index] = cur_process

                while not self.Queue_vfs_comp[uv_index][vfs_index].empty():

                    remain_task = self.task_on_process_vfs[uv_index][vfs_index][0] - vfs_task_deal
                    time_exceed = self.task_on_process_vfs[uv_index][vfs_index][1] + \
                        self.max_delay < self.time_count

                    if remain_task < 0 and not time_exceed:
                        reward += self.max_delay
                        vfs_task_deal = -remain_task
                        self.complete_task += 1
                        cur_process = self.Queue_vfs_comp[uv_index][vfs_index].get(
                        )
                        self.task_on_process_vfs[uv_index][vfs_index] = cur_process

                    if remain_task > 0 and not time_exceed:
                        self.task_on_process_vfs[uv_index][vfs_index][0] = remain_task
                        break

                    if time_exceed:
                        self.drop_vf += 1
                        cur_process = self.Queue_vfs_comp[uv_index][vfs_index].get(
                        )
                        self.task_on_process_vfs[uv_index][vfs_index] = cur_process
                        reward -= reward_penalty

        # new task generate in current time slot
        self.task = self.create_task()
        self.state[:, 0] = self.task[:, 0]

        self.time_count += 1
        if self.time_count >= self.time_limit:
            self.done = True
        return self.state, reward, self.done
