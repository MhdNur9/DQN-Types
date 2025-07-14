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
# RobotArmEnv code
class RobotArmEnv:
    def __init__(self, basket_position=(0, 0),stopping_thresholds=1,wall1_position=(0, 0), arm_lengths=(1, 1), throw_velocity=2, initial_joint1_angle=np.pi/2, initial_joint2_angle=np.pi/2, initial_throw_angle=0,joint1_ranges_max=np.pi,joint1_ranges_min=0,joint2_ranges_max=np.pi,joint2_ranges_min=0):
        self.basket_position = np.array(basket_position)
        self.stopping_thresholds=stopping_thresholds
        self.arm_lengths = arm_lengths
        self.joint_ranges = (0, np.pi)  # Joint angle ranges
        self.joint1_ranges = (joint1_ranges_min,joint1_ranges_max)  # Joint1 angle ranges
        self.joint2_ranges = (joint2_ranges_min,joint2_ranges_max)  # Joint2 angle ranges
        self.action_space = 10  # 10 actions: increase/decrease joint1/joint2, throw, increase/decrease throwing angle
        self.observation_space_dim = 2  # 2 dimensions:# [state_idx,current_x_coordinator,current_y_coordinator,throwing_moment_velocity,throwing_moment_theta,Theta1_selected,Theta2_selected,error_in_distance]
        self.current_joint1_angle = initial_joint1_angle
        self.current_joint2_angle = initial_joint2_angle
        self.current_end_effector_x_position=self.get_end_effector_position()[0]
        self.current_end_effector_y_position =self.get_end_effector_position()[1]
        self.current_throw_angle = initial_throw_angle
        self.throw_velocity = throw_velocity
        self.throw_trajectory = 1
        self.error_based_adjusmtent=0.1
        self._throw()
        self.wall1=np.array(wall1_position)
    def reset(self):
        self.big_negative_reward = -100
        self.target_is_reached_reward=1000
        self.current_joint1_angle = np.pi/3
        self.current_joint2_angle = np.pi/3
        self.current_throw_angle = 0
        self.throw_trajectory = self._throw_trajectory()
        return self._get_state()
    def step(self, action):
        # Update joint angles or throw
        error_based_adjustment=self._error_based_adjustment()
        old_Theta1 = self.current_joint1_angle
        old_Theta2 = self.current_joint2_angle
        old_throwing_angle=self.current_throw_angle
        if action == 0:
            self.current_joint1_angle = np.clip(self.current_joint1_angle + error_based_adjustment, self.joint1_ranges[0], self.joint1_ranges[1])
        elif action == 1:
            self.current_joint1_angle = np.clip(self.current_joint1_angle - error_based_adjustment, self.joint1_ranges[0], self.joint1_ranges[1])
        elif action == 2:
            self.current_joint2_angle = np.clip(self.current_joint2_angle + error_based_adjustment, self.joint_ranges[0], self.joint_ranges[1])
        elif action == 3:
            self.current_joint2_angle = np.clip(self.current_joint2_angle - error_based_adjustment, self.joint_ranges[0], self.joint_ranges[1])
        elif action == 4:
            self.current_joint1_angle = np.clip(self.current_joint1_angle + error_based_adjustment, self.joint1_ranges[0], self.joint1_ranges[1])
            self.current_joint2_angle = np.clip(self.current_joint2_angle + error_based_adjustment, self.joint_ranges[0], self.joint_ranges[1])
        elif action == 5:
            self.current_joint1_angle = np.clip(self.current_joint1_angle + error_based_adjustment, self.joint1_ranges[0], self.joint1_ranges[1])
            self.current_joint2_angle = np.clip(self.current_joint2_angle - error_based_adjustment, self.joint_ranges[0], self.joint_ranges[1])
        elif action == 6:
            self.current_joint1_angle = np.clip(self.current_joint1_angle - error_based_adjustment, self.joint1_ranges[0], self.joint1_ranges[1])
            self.current_joint2_angle = np.clip(self.current_joint2_angle + error_based_adjustment, self.joint_ranges[0], self.joint_ranges[1])
        elif action == 7:
            self.current_joint1_angle = np.clip(self.current_joint1_angle - error_based_adjustment, self.joint1_ranges[0], self.joint1_ranges[1])
            self.current_joint2_angle = np.clip(self.current_joint2_angle - error_based_adjustment, self.joint_ranges[0], self.joint_ranges[1])
        elif action == 8:
            self.current_throw_angle = np.clip(self.current_throw_angle + 0.2, -np.pi/2, np.pi/2)
        elif action == 9:
            self.current_throw_angle = np.clip(self.current_throw_angle - 0.2, -np.pi/2, np.pi/2)
        self.current_end_effector_x_position=self.get_end_effector_position()[0]
        self.current_end_effector_y_position =self.get_end_effector_position()[1]


        if self.check_wall_collision() == 1:
            self.current_joint1_angle = old_Theta1
            self.current_joint2_angle = old_Theta2
            self.current_throw_angle = old_throwing_angle
            self.current_end_effector_x_position = self.get_end_effector_position()[0]
            self.current_end_effector_y_position = self.get_end_effector_position()[1]
        if self.current_end_effector_y_position< 0.15:
            self.current_joint1_angle = old_Theta1
            self.current_joint2_angle = old_Theta2
            self.current_throw_angle = old_throwing_angle
            self.current_end_effector_x_position = self.get_end_effector_position()[0]
            self.current_end_effector_y_position = self.get_end_effector_position()[1]
        self.throw_trajectory = self._throw_trajectory()
        throw_position = self.throw_trajectory[-1]
        # Calculate Distance Error
        distance_error = -np.linalg.norm(self.basket_position - throw_position)
        if abs(distance_error) < abs(env.stopping_thresholds):
            done = True
        else:
            done = False

        return self._get_state(), distance_error, done,self.check_wall_collision(), {}

        return self._get_state(), 0, done, {}
    def _get_state(self):
        # get_state=np.array([self.get_end_effector_position()[0],self.get_end_effector_position()[1], self.current_throw_angle, self.throw_velocity,
        #                  self.current_joint1_angle, self.current_joint2_angle, self._calculate_distance_error()])
        get_state = np.array(
            [self.get_end_effector_position()[0], self.get_end_effector_position()[1], self.current_throw_angle,
             self.current_joint1_angle, self.current_joint2_angle])
        return get_state
    def _throw(self):
        # Calculate throwing position
        end_effector_position = np.array(self.get_end_effector_position())
        direction_vector = np.array([np.cos(self.current_joint1_angle + self.current_joint2_angle + self.current_throw_angle),
                                     np.sin(self.current_joint1_angle + self.current_joint2_angle + self.current_throw_angle)])
        throw_vector = end_effector_position + self.throw_velocity * direction_vector
        return throw_vector
    def _print_state(self):
        print(self.current_joint1_angle, self.current_joint2_angle, self.current_throw_angle,
              self.current_end_effector_x_position, self.current_end_effector_y_position)
    def _throw_trajectory(self):
        v_0 = self.throw_velocity
        time = 0  # 10 time steps
        #Theta = self.current_joint1_angle + self.current_joint2_angle + self.current_throw_angle
        Theta = self.current_throw_angle
        x_initial_position, y_initial_position = self.get_end_effector_position()
        vx_0 = v_0 * np.cos(Theta)
        vy_0 = v_0 * np.sin(Theta)
        horizontal_coordinator = vx_0 * time - x_initial_position
        vertical_coordinator = vy_0 * time - (0.5 * 9.8 * (time ** 2)) + y_initial_position
        trajectory=[]
        collision=0

        while vertical_coordinator >0:
            vx_0 = v_0 * np.cos(Theta)
            vy_0 = v_0 * np.sin(Theta)
            if abs(horizontal_coordinator - self.wall1[0]) < 0.01:
                if vertical_coordinator < self.wall1[1]:
                    collision=1
            if collision == 1:
                horizontal_coordinator = -self.wall1[0]
            else:
                horizontal_coordinator = vx_0 * time - x_initial_position
            vertical_coordinator = vy_0 * time - (0.5 * 9.8 * (time ** 2)) + y_initial_position
            horizontal_coordinator=-horizontal_coordinator
            trajectory.append([horizontal_coordinator, vertical_coordinator])

            time = time + 0.01
        return np.array(trajectory)
    def get_end_effector_position(self):
        T1_0 = np.array([[np.cos(self.current_joint1_angle), -np.sin(self.current_joint1_angle), 0, 0],
                         [np.sin(self.current_joint1_angle), np.cos(self.current_joint1_angle), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T2_1 = np.array(
            [[np.cos(self.current_joint2_angle), -np.sin(self.current_joint2_angle), 0, self.arm_lengths[0]],
             [np.sin(self.current_joint2_angle), np.cos(self.current_joint2_angle), 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]])
        T3_2 = np.array([[np.cos(0), -np.sin(0), 0, self.arm_lengths[1]],
                         [np.sin(0), np.cos(0), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T4_3 = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T4_0 = np.dot(np.dot(np.dot(T1_0, T2_1), T3_2), T4_3)
        x = T4_0[:3, 3]
        x=[x[0], x[1]]
        return x
    def get_joints_position(self):
        T1_0 = np.array([[np.cos(self.current_joint1_angle), -np.sin(self.current_joint1_angle), 0, 0],
                         [np.sin(self.current_joint1_angle), np.cos(self.current_joint1_angle), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T2_1 = np.array(
            [[np.cos(self.current_joint2_angle), -np.sin(self.current_joint2_angle), 0, self.arm_lengths[0]],
             [np.sin(self.current_joint2_angle), np.cos(self.current_joint2_angle), 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]])
        T2_0 = np.dot(T1_0, T2_1)
        robot_link2 = T2_0[:3, 3]
        robot_link2_x = robot_link2[0]
        robot_link2_y = robot_link2[1]

        T3_2 = np.array([[np.cos(0), -np.sin(0), 0, self.arm_lengths[1]],
                         [np.sin(0), np.cos(0), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T3_0 = np.dot(np.dot(T1_0, T2_1), T3_2)
        robot_link3 = T3_0[:3, 3]
        robot_link3_x = robot_link3[0]
        robot_link3_y = robot_link3[1]

        T4_3 = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        T4_0 = np.dot(np.dot(np.dot(T1_0, T2_1), T3_2), T4_3)
        x = T4_0[:3, 3]
        robot_link4_x = x[0]
        robot_link4_y = x[1]

        x=[0,0,robot_link2_x,robot_link2_y,robot_link3_x,robot_link3_y]
        return x
    def plot(self):
        plt.figure(figsize=(8, 6))
        plt.plot([0, self.arm_lengths[0] * np.cos(self.current_joint1_angle)], [0, self.arm_lengths[0] * np.sin(self.current_joint1_angle)], 'r-', markersize=15)
        plt.plot([self.arm_lengths[0] * np.cos(self.current_joint1_angle), self.arm_lengths[0] * np.cos(self.current_joint1_angle) + self.arm_lengths[1] * np.cos(self.current_joint1_angle + self.current_joint2_angle)],
                 [self.arm_lengths[0] * np.sin(self.current_joint1_angle), self.arm_lengths[0] * np.sin(self.current_joint1_angle) + self.arm_lengths[1] * np.sin(self.current_joint1_angle + self.current_joint2_angle)], 'b-', markersize=15)
        plt.plot(self.basket_position[0], self.basket_position[1], 'go', markersize=35)
        plt.plot(0, 0, 'ko', markersize=8)  # Plot the base of the robot arm
        plt.plot([ self.wall1[0] , self.wall1[0]], [ 0 , self.wall1[1]], marker='o')
        plt.plot()
        if self.throw_trajectory is not None:
            plt.plot(self.throw_trajectory[:, 0], self.throw_trajectory[:, 1], 'g--', label='Throw trajectory')
        plt.axis('equal')
        #plt.xlim(-1, 1)
        #plt.ylim(-0.005, 1)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Robot Arm')
        plt.legend()
        plt.grid()
        plt.show()
    def _error_based_adjustment(self):
        # Calculate adjustment based on error distance
        if self.throw_trajectory is None:
            return 0
        error_distance = np.linalg.norm(self.basket_position - self.throw_trajectory[-1])
        error_distance = 3 * (abs(error_distance) / 5)
        return error_distance  # Adjust by 10% of error distance
    def _calculate_distance_error(self):
        # Calculate adjustment based on error distance
        if self.throw_trajectory is None:
            return 0
        error_distance = np.linalg.norm(self.basket_position - self.throw_trajectory[-1])
        return -error_distance  # Adjust by 10% of error distance
    def check_wall_collision(self):
        """
        Check if the robot body has hit the wall.

        Parameters:
            robot_position (tuple): Current position of the robot (x, y).
            wall_boundaries (tuple): Boundaries of the wall (min_x, max_x, min_y, max_y).

        Returns:
            int: 1 if the robot body hits the wall, 0 otherwise.
        """
        Data=self.get_joints_position()
        m1 = (Data[3] - Data[1]) / (Data[2] - Data[0])
        m2 = (Data[5] - Data[3]) / (Data[4] - Data[2])
        if self.wall1[1] >= ((m1 * self.wall1[0]) - (m1 * Data[2]) + Data[3]) and ((m1 * self.wall1[0]) - (m1 * Data[2]) + Data[3]) > 0:
            #print("1st link hit by the wall")
            return 1  # Robot body hits the wall
        elif self.wall1[1] >= ((m2 * self.wall1[0]) - (m2 * Data[4]) + Data[5]) and ((m2 * self.wall1[0]) - (m2 * Data[4]) + Data[5]) > 0:
            #print("2st link hit by the wall")
            return 1  # Robot body hits the wall
        else:
            return 0  # Robot body does not hit the wall
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
# Create the environment
env = RobotArmEnv(basket_position=(-0.6, 0), arm_lengths=(0.2, 0.2), throw_velocity=2,wall1_position=(-0.9,0.3),stopping_thresholds=0.03)
env_agent_Double_DQN = env
testing_env = env # for testing only
state_size = env._get_state()
# Defined the parameters
Learning_rate = 0.001
no_layer = 8
num_episodes = 100
no_neurons = 256
no_epoch = 500
no_states = state_size.shape[0] # no. information per state not the state id
state_lst=[]
no_actions = 10
noise_scale = 0.1
# Training loop
batch_size = 32
time_range = 500
total_reward = 0
env = env_agent_Double_DQN
# Create an instance of the DQNAgent
dqn_agent = Noisy_Double_DQNAgent(no_states,no_actions,no_layer,Learning_rate,no_neurons,no_epoch)
state_lst = []########### Double DQN Agent
print("Double DQN Agent")
total_reward = 0
successful_trials = []
losses =[]
for episode in range(num_episodes):
    print("Episode ",episode)
    state = env.reset()
    time = 0
    temp_index = 0
    done = False
    state_lst,state_idx=search_or_add_state(state_lst, state)
    while (not done):
        old_state = state
        state_lst,old_state_idx = search_or_add_state(state_lst, old_state)
        action = dqn_agent.act(old_state_idx,state_lst)
        next_state, distance_error, done, collusion, _ = env.step(action)
        state_lst,next_state_idx = search_or_add_state(state_lst, next_state)
        reward = dqn_agent.reward_cal(old_state_idx,action,distance_error,next_state_idx,collusion)
        reward = reward if not done else -10
        dqn_agent.remember(state,action,reward,next_state,done)
        total_reward += reward
        state = next_state
        #dqn_agent.print_val(old_state_idx,action,distance_error,next_state_idx,collusion,reward)
        temp_index = temp_index+1
        time += 1
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(episode, num_episodes, time, dqn_agent.epsilon))
            successful_trials.append(temp_index)
            break
    if len(dqn_agent.memory) > batch_size:

        loss = dqn_agent.replay(batch_size, state_lst)
    print("Total_reward = ",total_reward)
    dqn_agent.total_reward_lst.append(total_reward)
Double_dqn_agent = dqn_agent

# Use trained agent
state1 = testing_env.reset()
testing_reward = 0
testing_total_reward = 0
state_lst, state_idx = search_or_add_state(state_lst, state1)
done = False
while not done:
    action = dqn_agent.act(state_idx,state_lst)
    next_state, distance_error, done, collusion, _ = testing_env.step(action)
    state_lst, new_state_idx = search_or_add_state(state_lst, next_state)
    reward = dqn_agent.reward_cal(old_state_idx, action, distance_error, next_state_idx, collusion)
    total_reward += reward
    state = next_state

algorithm_rewards = [
     Double_dqn_agent.total_reward_lst   # Algorithm 1 rewards
]
#save_list(total_rewards_table_file, algorithm_rewards)
print("algorithm_rewards",algorithm_rewards[0][-1])
print(algorithm_rewards)
labels = ['Noisy Double DQN']
plot_rewards(algorithm_rewards, labels)
print("successful_trials = ")
print(successful_trials)
print("Losses = ")
print(loss)