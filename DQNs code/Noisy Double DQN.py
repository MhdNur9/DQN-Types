import keras
import tensorflow as tf
from keras.initializers import RandomUniform
import keras.backend as K
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
##########################
# Main fucntions
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
###
# Double DQNAgent code
class Noisy_Double_DQNAgent:
    def __init__(self, state_size, action_size, no_layer, Learning_rate, no_neurons, no_epoch):
        self.state_size = state_size
        self.total_reward_lst = []
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.no_layers = no_layer
        self.epsilon_decay = 0.995
        self.learning_rate = Learning_rate
        self.no_neurons_per_layer = no_neurons
        self.no_epoch = no_epoch
        self.model = self._build_model()
        self.target_model = self._build_model()
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
        def noisy_layer(x):
            noise = tf.random.normal(shape=tf.keras.backend.int_shape(x), mean=0., stddev=0.1)
            return x + noise
        inputs = Input(shape=(self.state_size,))
        x = Dense(self.no_neurons_per_layer, activation='relu')(inputs)
        for _ in range(self.no_layers):
            x = Lambda(noisy_layer, output_shape=tf.keras.backend.int_shape(x))(x)  # Add Noisy layer
            x = Dense(self.no_neurons_per_layer, activation='relu')(x)
        outputs = Dense(self.action_size, activation='linear')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer=adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
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
    def replay(self, batch_size, state_lst):
        minibatch = random.sample(list(self.memory), min(len(self.memory), batch_size))
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = next_state.reshape(1, -1)
                best_action = np.argmax(self.model.predict(next_state)[0])
                target = reward + self.gamma * self.target_model.predict(next_state)[0][best_action]
            state = state.reshape(1, -1)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            states.append(state)
            targets.append(target_f)

        history = self.model.fit(np.vstack(states), np.vstack(targets), epochs=100, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return history.history['loss']  # Return the loss value for this replay

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
