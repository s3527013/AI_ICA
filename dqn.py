import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, environment, learning_rate=0.001, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 batch_size=64):
        self.environment = environment
        self.state_size = environment.grid_size * 2 + environment.num_locations  # Simplified state
        self.action_size = environment.num_locations
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=2000)

        self.model = DQN(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def _flatten_state(self, state):
        current_pos, remaining_locations = state
        flat_state = np.zeros(self.state_size)
        flat_state[0:2] = current_pos
        for loc in remaining_locations:
            flat_state[2 + loc] = 1
        return flat_state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.environment.get_possible_actions())

        state = self._flatten_state(state)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)

        possible_actions = self.environment.get_possible_actions()
        q_values = q_values[0].numpy()

        valid_q_values = {action: q_values[action] for action in possible_actions}
        if not valid_q_values:
            return None
        return max(valid_q_values, key=valid_q_values.get)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = self._flatten_state(state)
            state = torch.FloatTensor(state)

            target = reward
            if not done:
                next_state_flat = self._flatten_state(next_state)
                next_state_flat = torch.FloatTensor(next_state_flat)
                target = reward + self.gamma * torch.max(self.model(next_state_flat)).item()

            q_values = self.model(state)
            q_values[action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), q_values)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
