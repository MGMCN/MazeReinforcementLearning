import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input, hidden, output):
        super(Net, self).__init__()
        self.il = nn.Linear(input, hidden)
        # 目前没激活层 先试试看吧!
        # self.hl0 = nn.Linear(hidden, hidden)
        # self.hl1 = nn.Linear(hidden, hidden * 2)
        # self.hl2 = nn.Linear(hidden * 2, hidden * 2)
        # self.hl3 = nn.Linear(hidden * 2, hidden)
        #
        self.ol = nn.Linear(hidden, output)

    def forward(self, x):
        x = self.il(x)
        x = F.relu(x)  # 作用 ？？？
        # x = self.hl0(x)
        # x = F.relu(x)
        # x = self.hl1(x)
        # x = F.relu(x)
        # x = self.hl2(x)
        # x = F.relu(x)
        # x = self.hl3(x)
        # x = F.relu(x)
        x = self.ol(x)
        return x


class DeepQNetwork():
    def __init__(self, actions, input, hidden=20, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=200, memory_size=200, batch_size=32, e_greedy_increment=None,
                 ):
        self.actions = actions
        self.actions_cnt = len(actions)
        self.input = input
        self.hidden = hidden
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter  # 替换神经网络需要迭代的次数
        self.memory_size = memory_size  # 可以保存多少sample
        self.memory_counter = 0
        self.batch_size = batch_size  # 从sample集合中,每次训练sample样品数
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, input * 2 + 2))  # size = 6 <- 输入俩坐标 + action + reward + 输出俩坐标

        self.loss_func = nn.MSELoss()  # 这个以后可以尝试换下其他的

        self._build_net()

    def _build_net(self):
        self.q_eval = Net(self.input, self.hidden, self.actions_cnt)
        self.q_target = Net(self.input, self.hidden, self.actions_cnt)
        self.optimizer = torch.optim.RMSprop(self.q_eval.parameters(), lr=self.lr)  # RMSprop可以自己了解下是啥哟

    def store_transition(self, state, action, reward, next_state):
        if self.memory_counter == self.memory_size:
            self.memory_counter = 0
        self.memory[self.memory_counter, :] = np.array(
            [state[0], state[1], action, reward, next_state[0], next_state[1]])
        self.memory_counter += 1

    def choose_action(self, observation_state):
        input = torch.Tensor([[observation_state[0], observation_state[1]]])
        if np.random.uniform() < self.epsilon:
            actions_value = self.q_eval(input)
            values = actions_value.data.numpy()
            action = np.argmax(values)
        else:
            action = np.random.randint(0, self.actions_cnt)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next = self.q_target(torch.Tensor(batch_memory[:, -self.input:]))  # 老的
        q_eval = self.q_eval(torch.Tensor(batch_memory[:, :self.input]))

        q_target = torch.Tensor(q_eval.data.numpy().copy())

        batch_index = np.arange(self.batch_size, dtype=np.int32)  # 仅仅是 0-31 index下标
        eval_act_index = batch_memory[:, self.input].astype(int)  # 较老的但也不是很老的做了什么动作
        reward = torch.Tensor(batch_memory[:, self.input + 1])  # 取出来较老的但也不是很老的奖励是多少
        q_target[batch_index, eval_act_index] = reward + self.gamma * torch.max(q_next, 1)[0]  # 算老的reward

        loss = self.loss_func(q_eval, q_target)  # 和新的比较差了多少,你会发现其他没采取动作的地方相减为0
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
