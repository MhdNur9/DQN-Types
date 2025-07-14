
class PEX_Double_DQNAgent:
    def __init__(self, state_size, action_size, no_layer, Learning_rate, no_neurons, buffer_size=2000, alpha=0.6,
                 beta=0.4, beta_increment_per_sampling=0.001, abs_err_upper=1.0):
        self.state_size = state_size
        self.total_reward_lst = []
        self.action_size = action_size
        self.alpha = alpha
        # self.memory = sumtree.SumTree(capacity = buffer_size,alpha =alpha)# Initialize SumTree for prioritized experience replay
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
        self.target_model = self._build_model()
        self.distance_error = 100

    def reward_cal(self, state_id, action, distance_error, next_state_id, collusion):
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
        if batch != 0:
            for i in range(int(len(ISWeights_mb) / 5)):
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
                    target = reward + self.gamma * self.target_model.predict(next_state)[0][best_action]
                state = state.reshape(1, -1)
                # print(state)
                target_f = self.model.predict(state)
                target_f[0][action] = target
                states.append(state)
                targets.append(target_f)
            history = self.model.fit(np.vstack(states), np.vstack(targets), epochs=10, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            return history.history['loss']

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

