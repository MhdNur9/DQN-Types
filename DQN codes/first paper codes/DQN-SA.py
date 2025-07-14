import keras
from keras.layers import Dense, Input, Attention
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
class DQN_attention_Agent:
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
        self.distance_error = 100
    def reward_cal(self, state_id, action, distance_error, next_state_id,collusion):
        new_reward_step = 1 - 0.5 * (abs(distance_error) - abs(self.distance_error))
        if collusion > 0:
            new_reward_step=new_reward_step-200
            #print(new_reward_step)
        if abs(self.distance_error) <= abs(distance_error):
            new_reward_step = new_reward_step - 20
        else:
            new_reward_step = new_reward_step + 5
        return new_reward_step
    def print_val(self,state_id, action, distance_error, next_state_id,collusion,reward):
        print("state_id ",state_id)
        print("action ", action)
        print("distance_error ", distance_error)
        print("next_state_id ", next_state_id)
        print("collusion ", collusion)
        print("reward ", reward)
    def _build_attention_layers(self):
        attention_layers = []
        for _ in range(self.no_layers*5):
            attention_layer = Attention()
            attention_layers.append(attention_layer)
        return attention_layers
    def _build_model(self):
        state_input = Input(shape=(1, self.state_size))  # Adjusted input shape
        attention_layers = self._build_attention_layers()
        attention_outputs = []
        for attention_layer in (attention_layers*5):
            # Pass the same input twice to mimic [query, value] format
            attention_output = attention_layer([state_input, state_input])
            attention_outputs.append(attention_output)
        merged_attention = Concatenate()(attention_outputs)
        x = Dense(self.no_neurons_per_layer, activation='relu')(merged_attention)
        for _ in range(self.no_layers - 1):
            x = Dense(self.no_neurons_per_layer, activation='relu')(x)
        output = Dense(self.action_size, activation='linear')(x)
        model = Model(inputs=state_input, outputs=output)
        print(model)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        return model
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state,state_lst):
        y = np.random.uniform(0, 1)
        if y <= self.epsilon:
            chosen_action = np.random.uniform(self.action_size)
            chosen_action = int(chosen_action)
            return chosen_action
        predict_state = state_lst[state]
        predict_state = np.array(predict_state)
        predict_state = predict_state.reshape(1, 1, -1)
        act_values = self.model.predict(predict_state)
        return np.argmax(act_values[0])
    def replay(self, batch_size,state_lst):
        minibatch = random.sample(list(self.memory), min(len(self.memory), batch_size))
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Reshape the array to have shape (1, 5)
                next_state = next_state.reshape(1, 1, -1)

                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            state = state.reshape(1, 1, -1)
            target_f = self.model.predict(state)
            target_f[0][0][action] = target
            # print("////////////////")
            # print(state)
            # print("**********")
            # print(target_f)
            # print("////////////////")
            states.append(state)
            targets.append(target_f)
        history=self.model.fit(np.vstack(states), np.vstack(targets), epochs=100, verbose=0)


        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon *= self.epsilon_decay
        return history.history['loss']  # Return the loss value for this replay
