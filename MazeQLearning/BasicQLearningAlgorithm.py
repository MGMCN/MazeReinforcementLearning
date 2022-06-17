import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,e_greedy_increment=0.1):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.e_greedy_increment = e_greedy_increment
        self.q_learning_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        # self.cnt = 0  # check the learning table for 10000 steps

    def check_state_exist(self, state):
        if state not in self.q_learning_table.index:
            self.q_learning_table.loc[state, :] = 0

    def change_epsilon(self):
        if abs(self.epsilon-0.9) < 0.0000000001:
            return
        self.epsilon += self.e_greedy_increment

    def choose_action(self, observation_state):
        self.check_state_exist(observation_state)
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_learning_table.loc[observation_state, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learning(self, state, action, reward, next_state, done):
        self.check_state_exist(next_state)
        q_predict_reward = self.q_learning_table.loc[state, action]
        if not done:  # trap or end or just road
            q_target_reward = reward + self.gamma * self.q_learning_table.loc[next_state, :].max()
        else:
            q_target_reward = reward
        self.q_learning_table.loc[state, action] += (self.lr * (q_target_reward - q_predict_reward))
        # self.cnt += 1
        # if self.cnt % 10 == 0:
        #     print(self.q_learning_table)
