import keras
from keras.layers import Dense, Input
from keras.layers import Lambda
from keras.layers import Concatenate
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam as adam
from keras import backend as K  # Add this import statement
from keras.optimizers import Adam
############################
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
import random
import math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
#############################
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sumtree import sumtree
##########################
# Main fucntions
def plot_distance_error(rewards):
    # Plotting
    best_reward = rewards[0]
    worst_reward = rewards[0]
    for i in range (len(rewards)):
        label = str("episode"+str(i))
        if len(best_reward) > len(rewards[i]):
            best_reward = rewards[i]
        if len(worst_reward) < len(rewards[i]):
            worst_reward = rewards[i]
        plt.plot(rewards[i], label=label)
        # Add labels to points
        for j, reward in enumerate(rewards[i]):
            plt.text(j, reward, f'{reward:.2f}', ha='left', va='bottom', fontsize=8, color='black')

    # Adding labels and legend
    plt.xlabel('Episodes')
    plt.ylabel('Distance error')
    plt.title('Distance error vs Episodes')
    plt.legend()
    # Show plot
    #plt.show()
    return best_reward,worst_reward

def search_or_add_state(state_lst,New_state):
    """
    Search for state information and return the state ID if it exists.
    If the state doesn't exist, add it to the state list and return the new state ID.

    Parameters:
        New_state (numpy.ndarray): Information about the state.
        state_list (numpy.ndarray): Array containing state information.

    Returns:
        int: State ID.
    """
    for i, state in enumerate(state_lst):
        if np.array_equal(state, New_state):
            return state_lst,i  # Return existing state ID

        # If state doesn't exist, add it to state list
    state_lst.append(New_state.tolist())


    return state_lst,len(state_lst) - 1  # Return new state ID
def plot_rewards(algorithm_rewards, labels):
    """
    Plot rewards for different algorithms.

    Parameters:
    - algorithm_rewards (list of lists): List containing rewards for each algorithm.
    - labels (list of strings): List containing labels for each algorithm.
    """
    # Plotting
    for rewards, label in zip(algorithm_rewards, labels):
        plt.plot(rewards, label=label)
    # Adding labels and legend
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Returns of DQN Algorithm')
    plt.legend()
    # Show plot
    plt.show()
def print_variable(x):
    """
    print the type and the value of X

    Parameters:
    - X: the variable for printing

    """
    print("/////////////")
    print(" x = ",type(x))
    print(x)
    print("/////////////")
def save_list(file_name, data):
    """
    Save a list to a text file.

    Parameters:
        file_name (str): Name of the file to save the list to.
        data (list): List to be saved.
    """
    with open(file_name, 'w') as f:
        for item in data:
            f.write(str(item) + '\n')
#####################
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = [0] * (2 * capacity - 1)
        self.data = [None] * capacity
        self.write_index = 0
        self.full = False
    def __len__(self):
        return len(self.data)
    def total(self):
        return self.tree[0]
    def add(self, priority, data):
        self.data[self.write_index] = data
        self.update(self.write_index, priority)
        self.write_index += 1
        if self.write_index >= self.capacity:
            self.write_index = 0
            self.full = True
    def update(self, index, priority):
        tree_index = index + self.capacity - 1
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    def get_leaf(self, value):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            if value <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                value -= self.tree[left_child_index]
                parent_index = right_child_index
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]
    def sample(self, value):
        leaf_index, priority, data = self.get_leaf(value)
        return leaf_index, priority, data
    def update_priorities(self, indices, priorities):
        for i, p in zip(indices, priorities):
            self.update(i, p)
# Double DQNAgent code
class PEX_Double_DQNAgent:
    def __init__(self, state_size, action_size, no_layer, Learning_rate, no_neurons,buffer_size=2000, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, abs_err_upper=1.0):
        self.state_size = state_size
        self.total_reward_lst = []
        self.action_size = action_size
        self.alpha = alpha
        #self.memory = sumtree.SumTree(capacity = buffer_size,alpha =alpha)# Initialize SumTree for prioritized experience replay
        self.memory = SumTree(capacity=buffer_size)  # Initialize SumTree for prioritized experience replay
        self.batch_size = batch_size
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.abs_err_upper = abs_err_upper
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.no_layers = no_layer
        self.epsilon_decay = 0.995
        self.learning_rate = Learning_rate
        self.no_neurons_per_layer = no_neurons
        self.no_epoch = no_epoch
        self.model = self._build_model()
        self.distance_error = 100
    def reward_cal(self, state_id, action, distance_error, next_state_id,collusion):
        new_reward_step = 1 - 0.5 * (abs(distance_error) - abs(self.distance_error))
        if collusion > 0:
            new_reward_step = new_reward_step - 200
        if abs(self.distance_error) <= abs(distance_error):
            new_reward_step = new_reward_step - 20
        else:
            new_reward_step = new_reward_step + 5
        return new_reward_step
    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.no_neurons_per_layer, input_dim=self.state_size, activation='relu'))
        for element in range(self.no_layers):
            model.add(Dense(self.no_neurons_per_layer, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        return model
    def remember(self, state, action, reward, next_state, done):
        max_p = np.max(self.memory.tree[-self.memory.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.memory.add(max_p, (state, action, reward, next_state, done))
    def act(self, state, state_lst):
        y = np.random.uniform(0, 1)
        if y <= self.epsilon:
            chosen_action = np.random.uniform(self.action_size)
            chosen_action = int(chosen_action)
            return chosen_action
        predict_state = state_lst[state]
        predict_state = np.array(predict_state)
        predict_state = predict_state.reshape(1, -1)
        act_values = self.model.predict(predict_state)
        return np.argmax(act_values[0])
    def replay(self):
        # print("self.memory",type(self.memory))
        # print(self.memory)
        tree_idx, batch, ISWeights_mb = self.memory.sample(self.batch_size)
        # print(self.memory.sample(self.batch_size))
        states, targets = [], []
        # print("ISWeights_mb",type(ISWeights_mb))
        # print(ISWeights_mb)
        batch = int(batch)
        # print("batch",type(batch),batch)
        if batch!= 0:
            for i in range(int(len(ISWeights_mb)/5)):
                # print(int((len(ISWeights_mb) / 5)))
                # print(ISWeights_mb)
                state = ISWeights_mb[0]
                action = ISWeights_mb[1]
                reward = ISWeights_mb[2]
                next_state = ISWeights_mb[3]
                done = ISWeights_mb[4]
                target = reward
                # print("state, action, reward, next_state, done")
                # print(state, action, reward, next_state, done)
                if not done:
                    next_state = next_state.reshape(1, -1)
                    best_action = np.argmax(self.model.predict(next_state)[0])
                    target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                state = state.reshape(1, -1)
                # print(state)
                target_f = self.model.predict(state)
                target_f[0][action] = target
                states.append(state)
                targets.append(target_f)
            history = self.model.fit(np.vstack(states), np.vstack(targets), epochs=100, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            return history.history['loss']
