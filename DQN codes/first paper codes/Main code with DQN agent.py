import keras
from keras.layers import Dense, Input, Lambda, Concatenate
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras import backend as K
############################
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

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

def analyze_and_plot(data):
    """
    Finds the lowest and highest numbers in the given data,
    and plots a histogram of the data with numbers displayed on top of each bar.

    Args:
    - data (list of int): A list of numerical data to analyze.

    Returns:
    - tuple: The lowest and highest numbers in the data.
    """
    if not data:
        print("The data list is empty.")
        return None, None

    # Find the smallest and largest numbers
    min_value = min(data)
    max_value = max(data)

    # Print the results
    print(f"Lowest number: {min_value}")
    print(f"Biggest number: {max_value}")

    # Plot the histogram
    counts, bins, patches = plt.hist(data, bins=10, color='skyblue', edgecolor='black')

    # Display numbers on top of each bar
    for count, bin_edge in zip(counts, bins):
        plt.text(bin_edge + (bins[1] - bins[0]) / 2, count, str(int(count)),
                 ha='center', va='bottom', fontweight='bold')

    plt.title('Histogram of Needed Data for ')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    return min_value, max_value
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
        #plt.plot(rewards[i], label=label)
        # Add labels to points
        # for j, reward in enumerate(rewards[i]):
        #     plt.text(j, reward, f'{reward:.2f}', ha='left', va='bottom', fontsize=8, color='black')

    # Adding labels and legend
    # plt.xlabel('Episodes')
    # plt.ylabel('Distance error')
    # plt.title('Distance error vs Episodes')
    # plt.legend()
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
def jacobian_func(A):
    """
    Compute the Jacobian matrix for the robot arm at the given joint angles.

    Parameters:
    angles (array): An array containing the joint angles [theta1, theta2].

    Returns:
    np.array: The Jacobian matrix.
    """
    T1, T2, T3, l1, l2, l3 = A

    a1 = -l1 * np.sin(T1) - l2 * np.sin(T1 + T2) - l3 * np.sin(T1 + T2 + T3)
    a2 = -l2 * np.sin(T1 + T2) - l3 * np.sin(T1 + T2 + T3)
    a3 = -l3 * np.sin(T1 + T2 + T3)
    b1 = l1 * np.cos(T1) + l2 * np.cos(T1 + T2) + l3 * np.cos(T1 + T2 + T3)
    b2 = l2 * np.cos(T1 + T2) + l3 * np.cos(T1 + T2 + T3)
    b3 = l3 * np.cos(T1 + T2 + T3)

    results = np.array([[a1, a2, a3], [b1, b2, b3]])
    return results

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
        self.current_joint3_angle = initial_throw_angle
        self.current_joint1_speed = throw_velocity
        self.current_joint2_speed = throw_velocity
        self.current_joint3_speed = throw_velocity
        self.current_end_effector_x_position=self.get_end_effector_position()[0]
        self.current_end_effector_y_position =self.get_end_effector_position()[1]

        A = np.array([self.current_joint1_angle, self.current_joint2_angle, self.current_joint3_angle, self.arm_lengths[0], self.arm_lengths[1], self.arm_lengths[2]])
        print("Init A angles = ",A)
        R2 = jacobian_func(A)
        print("Init R2 = ", R2)
        A = np.array([self.current_joint1_speed, self.current_joint2_speed, self.current_joint3_speed])
        print("Init A Speed = ",A)
        results = np.dot(R2, A)
        print("Init Results speed = ",results)
        self.current_end_effector_x_speed = results[0]
        self.current_end_effector_y_speed = results[1]
        self.throw_trajectory = 1
        self.error_based_adjusmtent=0.1
        self.wall1=np.array(wall1_position)
    def reset(self):
        self.big_negative_reward = -100
        self.target_is_reached_reward=1000
        self.current_joint1_angle = np.pi/3
        self.current_joint2_angle = 0
        self.current_joint3_angle = np.pi/3
        self.throw_trajectory = self._throw_trajectory()
        return self._get_state()
    def step(self, action):
        # Update joint angles or throw
        error_based_adjustment=self._error_based_adjustment()
        old_Theta1 = self.current_joint1_angle
        old_Theta2 = self.current_joint2_angle
        old_Theta3 = self.current_joint3_angle
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
            self.current_joint3_angle = np.clip(self.current_joint3_angle + 0.2, -np.pi/2, np.pi/2)
        elif action == 9:
            self.current_joint3_angle = np.clip(self.current_joint3_angle - 0.2, -np.pi/2, np.pi/2)
        self.current_end_effector_x_position=self.get_end_effector_position()[0]
        self.current_end_effector_y_position =self.get_end_effector_position()[1]


        if self.check_wall_collision() == 1:
            self.current_joint1_angle = old_Theta1
            self.current_joint2_angle = old_Theta2
            self.current_joint3_angle = old_Theta3
            self.current_end_effector_x_position = self.get_end_effector_position()[0]
            self.current_end_effector_y_position = self.get_end_effector_position()[1]
        if self.current_end_effector_y_position< 0.15:
            self.current_joint1_angle = old_Theta1
            self.current_joint2_angle = old_Theta2
            self.current_joint3_angle = old_Theta3
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
            [self.get_end_effector_position()[0], self.get_end_effector_position()[1], self.current_joint1_angle, self.current_joint2_angle, self.current_joint3_angle])
        #print("get_state",get_state)
        return get_state

    def _print_state(self):
        print(self.current_joint1_angle, self.current_joint2_angle, self.current_joint3_angle,
              self.current_end_effector_x_position, self.current_end_effector_y_position)
    def _throw_trajectory(self):
        A = np.array(
            [self.current_joint1_angle, self.current_joint2_angle, self.current_joint3_angle, self.arm_lengths[0],
             self.arm_lengths[1], self.arm_lengths[2]])
        R2 = jacobian_func(A)
        A = np.array([self.current_joint1_speed, self.current_joint2_speed, self.current_joint3_speed])
        results = np.dot(R2, A)
        self.current_end_effector_x_speed = results[0]
        vx_0 = self.current_end_effector_x_speed
        self.current_end_effector_y_speed = results[1]
        vy_0 = self.current_end_effector_y_speed

        time = 0  # 10 time steps
        x_initial_position, y_initial_position = self.get_end_effector_position()
        horizontal_coordinator = vx_0 * time - x_initial_position
        vertical_coordinator = vy_0 * time - (0.5 * 9.8 * (time ** 2)) + y_initial_position
        trajectory=[]
        collision=0

        while vertical_coordinator >0:
            if abs(horizontal_coordinator - self.wall1[0]) < 0.01:
                if vertical_coordinator < self.wall1[1]:
                    collision=1
            if collision == 1:
                horizontal_coordinator = -self.wall1[0]
            else:
                horizontal_coordinator = vx_0 * time + x_initial_position
            vertical_coordinator = vy_0 * time - (0.5 * 9.8 * (time ** 2)) + y_initial_position

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
        return error_distance
    def _calculate_distance_error(self):
        # Calculate adjustment based on error distance
        if self.throw_trajectory is None:
            self.throw_trajectory = [-100]
            print(self.throw_trajectory[-1])
            raise Exception("Sorry, no numbers below zero")
            return 0
        print("self.basket_position",self.basket_position)
        print("self.throw_trajectory", self.throw_trajectory)
        print("self.throw_trajectory[-1]", self.throw_trajectory[-1])
        print("np.linalg.norm(self.basket_position - self.throw_trajectory[-1])",np.linalg.norm(self.basket_position - self.throw_trajectory[-1]))
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
# DQNAgent code
class DQNAgent:

    def __init__(self, state_size, action_size, no_layer, Learning_rate, no_neurons, no_epoch):
        self.state_size = state_size
        self.total_reward_lst = []
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.no_layers = no_layer
        self.epsilon_decay = 0.995
        self.learning_rate = Learning_rate
        self.no_neurons_per_layer = no_neurons
        self.no_epoch = no_epoch
        self.lr_decay = 0.9
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
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        for element in range(self.no_layers):
            model.add(Dense(self.no_neurons_per_layer, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.add(Dense(self.action_size, activation='linear'))
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
        # print("predict_state in DQN = ",predict_state)
        predict_state = np.array(predict_state)
        predict_state = predict_state.reshape(1, -1)
        act_values = self.model.predict(predict_state)
        return np.argmax(act_values[0])
    def replay(self, batch_size):
        minibatch = random.sample(list(self.memory), min(len(self.memory), batch_size))
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Reshape the array to have shape (1, 5)
                next_state = next_state.reshape(1, -1)
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            state = state.reshape(1, -1)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            states.append(state)
            targets.append(target_f)

        history=self.model.fit(np.vstack(states), np.vstack(targets), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def replay1(self, batch_size):
        minibatch = random.sample(list(self.memory), min(len(self.memory), batch_size))
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            # print("reward = ",reward)
            if not done:
                # Reshape the array to have shape (1, 5)
                next_state = next_state.reshape(1, -1)
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            state = state.reshape(1, -1)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            states.append(state)
            targets.append(target_f)
            # print("State = ",state)
            # print("target_f = ", target_f)
            # Find the index of the maximum value
            max_index = np.argmax(target_f)

            # Set all values to 0 except the max one
            target_f = np.zeros_like(target_f)
            target_f[0, max_index] = target  # The original value of the maximum element
            # print("New target_f = ", target_f)
            # raise Exception("Sorry, no numbers below zero")

        history=self.model.fit(np.vstack(states), np.vstack(targets), epochs=100, verbose=0)

        # Apply learning rate decay after each training step
        self.learning_rate *= self.lr_decay  # Update the learning rate
        self.model.optimizer.learning_rate = self.learning_rate  # Set the new learning rate

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return history.history['loss'][0]  # Return the loss value for this replay

    def replay2(self, batch_size):
        minibatch = random.sample(list(self.memory), min(len(self.memory), batch_size))
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            # print("reward = ",reward)
            if not done:
                # Reshape the array to have shape (1, 5)
                next_state = next_state.reshape(1, -1)
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            state = state.reshape(1, -1)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            states.append(state)
            targets.append(target_f)
            # print("State = ",state)
            # print("target_f = ", target_f)
            # Find the index of the maximum value
            max_index = np.argmax(target_f)

            # Set all values to 0 except the max one
            target_f = np.zeros_like(target_f)
            target_f[0, max_index] = target  # The original value of the maximum element
            # print("New target_f = ", target_f)
            # raise Exception("Sorry, no numbers below zero")

        history=self.model.fit(np.vstack(states), np.vstack(targets), epochs=100, verbose=0)

        # # Apply learning rate decay after each training step
        # self.learning_rate *= self.lr_decay  # Update the learning rate
        # self.model.optimizer.learning_rate = self.learning_rate  # Set the new learning rate

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return history.history['loss']  # Return the loss value for this replay
#######
# Create the environment
env = RobotArmEnv(basket_position=(-0.4, 0), arm_lengths=(0.2, 0.2,0.0), throw_velocity=2,wall1_position=(-0.9,0.3),stopping_thresholds=0.2)
env_agent_Double_DQN = env
state_size = env._get_state()
# Defined the parameters
Learning_rate = 0.00001
no_layer = 256
num_episodes = 10
no_neurons = 512
no_epoch = 500
no_states = state_size.shape[0] # no. information per state not the state id
state_lst=[]
no_actions = 10
noise_scale = 0.1
# Create an instance of the DQNAgent
dqn_agent=DQNAgent(no_states,no_actions,no_layer,Learning_rate,no_neurons,no_epoch)
# Training loop
batch_size = 32
time_range = 100
total_reward = 0
error_lst_episode = []
error_episode = []
episode_done_lst =[]
history_data_check = []
successful_trials = []
losses =[]
########### Normal DQN Agent
for episode in range(num_episodes):
    print("Episode ",episode)
    error_episode = []
    state = env.reset()
    time = 0
    temp_index = 0
    done = False
    state_lst,state_idx=search_or_add_state(state_lst, state)
    # for time in range (time_range):
    while not done:
        old_state = state
        time = time + 1
        # print("*******************")
        # print("Old state = ",old_state)
        # print("state_lst =",state_lst)
        state_lst,old_state_idx = search_or_add_state(state_lst, old_state)
        action = dqn_agent.act(old_state_idx,state_lst)
        next_state, distance_error, done, collusion, _ = env.step(action)
        state_lst,next_state_idx = search_or_add_state(state_lst, next_state)
        # print("*******************")
        # print("next_state_idx = ", next_state_idx)
        # print("state_lst =", state_lst)
        reward = dqn_agent.reward_cal(old_state_idx,action,distance_error,next_state_idx,collusion)
        dqn_agent.remember(state,action,reward,next_state,done)
        history_data_check.append([state,action,reward,next_state,done])
        total_reward += reward
        state = next_state
        #dqn_agent.print_val(old_state_idx,action,distance_error,next_state_idx,collusion,reward)
        error_episode.append(distance_error)
        temp_index = temp_index+1

        if done:
            episode_done_lst.append(time)
            env.plot()
            print("episode: {}/{}, score: {}, e: {:.2}".format(episode, num_episodes, time, dqn_agent.epsilon))
            error_lst_episode.append(error_episode)
            successful_trials.append(temp_index)
            loss = dqn_agent.replay2(batch_size)
            losses.append([loss])
            break
    # if len(dqn_agent.memory) > batch_size:
    #     loss = dqn_agent.replay2(batch_size)
    #     losses.append([loss])
    print("Total_reward = ",total_reward)
    dqn_agent.total_reward_lst.append(total_reward)

env.plot()

algorithm_rewards = [
    dqn_agent.total_reward_lst # Algorithm 1 rewards
]
#save_list(total_rewards_table_file, algorithm_rewards)
print("algorithm_rewards")
print(algorithm_rewards)
labels = ['Normal DQN']
plot_rewards(algorithm_rewards, labels)
data_compare_dict =[]
data_compare_lst =[]
print("episode_done_lst",episode_done_lst)

plt.plot(loss)
plt.title("DQN Training Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.grid()
plt.show()
print("***************")
print("successful_trials = ",successful_trials)
print("size = ")
print(analyze_and_plot(successful_trials))
print("results")
print(len(dqn_agent.memory))



