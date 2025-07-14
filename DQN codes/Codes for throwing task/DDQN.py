from keras import backend as K

class Double_DQNAgent:
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
        model = Sequential()
        model.add(Dense(self.no_neurons_per_layer, input_dim=self.state_size, activation='relu'))
        for element in range(self.no_layers):
            model.add(Dense(self.no_neurons_per_layer, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
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
