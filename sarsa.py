import numpy as np
import random
from collections import defaultdict

class SarsaAgent:
    def __init__(self, action_space, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        SARSA Agent.

        Args:
            action_space (list): A list of all possible actions.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Initial exploration rate.
            epsilon_decay (float): Rate at which epsilon decays.
            epsilon_min (float): Minimum value for epsilon.
        """
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.q_table = defaultdict(lambda: 0.0)

    def choose_action(self, state, possible_actions):
        """
        Choose an action using an epsilon-greedy policy.
        """
        if not possible_actions:
            return None

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(possible_actions)
        else:
            q_values = {action: self.q_table.get((state, action), 0.0) for action in possible_actions}
            max_q_value = max(q_values.values())
            best_actions = [action for action, q in q_values.items() if q == max_q_value]
            return random.choice(best_actions)

    def update_q_table(self, state, action, reward, next_state, next_action):
        """
        Update the Q-table using the SARSA update rule.

        Args:
            state: The state before the action.
            action: The action taken.
            reward: The reward received.
            next_state: The state after the action.
            next_action: The action to be taken in the next state.
        """
        old_value = self.q_table.get((state, action), 0.0)
        
        # Get Q-value of the next state and next action
        next_q_value = self.q_table.get((next_state, next_action), 0.0)

        # SARSA update rule
        new_value = old_value + self.alpha * (reward + self.gamma * next_q_value - old_value)
        self.q_table[(state, action)] = new_value

    def decay_epsilon(self):
        """
        Decay the exploration rate (epsilon).
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
