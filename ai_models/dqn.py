"""
This module implements a Deep Q-Learning (DQN) agent for solving reinforcement learning problems.
It includes the Q-Network, a Replay Buffer for experience replay, and the main DQNAgent class.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """
    A simple feed-forward neural network used as the function approximator for Q-values.
    The network takes a state as input and outputs the estimated Q-value for each possible action.
    """

    def __init__(self, state_size: int, action_size: int):
        """
        Initializes the Q-Network.

        Args:
            state_size (int): The dimensionality of the state space.
            action_size (int): The number of possible actions.
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The output tensor of Q-values for each action.
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """
    A simple replay buffer to store and sample experiences (transitions).
    This helps to break the correlation between consecutive experiences and stabilize training.
    """

    def __init__(self, capacity: int):
        """
        Initializes the Replay Buffer.

        Args:
            capacity (int): The maximum number of experiences to store in the buffer.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Adds a new experience to the buffer.

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The resulting next state.
            done (bool): Whether the episode has terminated.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list:
        """
        Randomly samples a batch of experiences from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            list: A list of sampled experience tuples.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """Returns the current number of experiences in the buffer."""
        return len(self.buffer)


class DQNAgent:
    """
    The main agent class that implements the Deep Q-Learning algorithm.
    It uses a Q-Network for function approximation, a target network for stable learning,
    and a replay buffer to store experiences.
    """

    def __init__(self, state_size: int, action_size: int, learning_rate: float = 1e-4, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 buffer_size: int = 10000, batch_size: int = 64, target_update_freq: int = 10):
        """
        Initializes the DQN Agent.

        Args:
            state_size (int): The dimensionality of the state space.
            action_size (int): The number of possible actions.
            learning_rate (float): The learning rate for the optimizer.
            gamma (float): The discount factor for future rewards.
            epsilon (float): The initial exploration rate.
            epsilon_decay (float): The rate at which epsilon decays.
            epsilon_min (float): The minimum value for epsilon.
            buffer_size (int): The capacity of the replay buffer.
            batch_size (int): The size of each training batch.
            target_update_freq (int): The frequency (in steps) to update the target network.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Set the device to GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the main Q-network and the target network
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is only for inference

        # Setup optimizer and loss function
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Initialize the replay buffer and a counter for target network updates
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.update_counter = 0

    def choose_action(self, state, possible_actions: list) -> int:
        """
        Chooses an action using an epsilon-greedy policy.

        Args:
            state: The current state of the environment.
            possible_actions (list): A list of valid actions to take from the current state.

        Returns:
            int: The chosen action.
        """
        if not possible_actions:
            return None

        # Exploration: choose a random action from the possible actions
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(possible_actions)
        
        # Exploitation: choose the best action based on Q-network's estimates
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor).cpu().numpy().flatten()

                # Mask out invalid actions by setting their Q-values to negative infinity
                masked_q_values = np.full(self.action_size, -np.inf)
                masked_q_values[possible_actions] = q_values[possible_actions]

                return np.argmax(masked_q_values)

    def update_model(self):
        """
        Updates the Q-network by sampling a batch from the replay buffer and performing a training step.
        """
        # Do not update if the buffer doesn't have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch of experiences
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert batch to PyTorch tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # --- Calculate the loss ---
        # 1. Get the Q-values for the actions that were actually taken
        current_q_values = self.q_network(states).gather(1, actions)

        # 2. Get the maximum Q-value for the next states from the target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            # 3. Compute the target Q-value using the Bellman equation
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # 4. Compute the Mean Squared Error loss between current and target Q-values
        loss = self.loss_fn(current_q_values, target_q_values)

        # --- Perform the optimization step ---
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # --- Update the target network ---
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def add_experience(self, state, action, reward, next_state, done):
        """
        Adds a new experience to the replay buffer. This is a convenience method.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def decay_epsilon(self):
        """
        Decays the exploration rate (epsilon) to shift from exploration to exploitation over time.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
