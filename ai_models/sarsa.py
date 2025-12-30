"""
This module implements the SARSA (State-Action-Reward-State-Action) agent,
an on-policy temporal-difference (TD) learning algorithm.
"""

import random
from collections import defaultdict


class SarsaAgent:
    """
    A reinforcement learning agent that learns a policy using the SARSA algorithm.

    SARSA is an on-policy algorithm, meaning it learns the Q-values for the policy
    it is currently following (including the exploration steps). This is in contrast
    to Q-Learning, which is off-policy.
    """

    def __init__(self, action_space: list, alpha: float = 0.1, gamma: float = 0.9,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        """
        Initializes the SARSA Agent.

        Args:
            action_space (list): A list of all possible actions in the environment.
            alpha (float): The learning rate, determining how much new information overrides old information.
            gamma (float): The discount factor, valuing future rewards.
            epsilon (float): The initial exploration rate for the epsilon-greedy policy.
            epsilon_decay (float): The rate at which epsilon decays after each episode.
            epsilon_min (float): The minimum value that epsilon can decay to.
        """
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # The Q-table stores the estimated value of taking an action in a given state.
        # It is a dictionary mapping (state, action) pairs to a Q-value.
        self.q_table = defaultdict(lambda: 0.0)

    def choose_action(self, state, possible_actions: list) -> int:
        """
        Chooses an action from a list of possible actions using an epsilon-greedy policy.

        With probability epsilon, it chooses a random action (exploration).
        Otherwise, it chooses the action with the highest Q-value (exploitation).

        Args:
            state: The current state of the environment.
            possible_actions (list): A list of valid actions for the current state.

        Returns:
            int: The chosen action, or None if no actions are possible.
        """
        if not possible_actions:
            return None

        # Exploration
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(possible_actions)
        
        # Exploitation
        else:
            # Get Q-values for all possible actions
            q_values = {action: self.q_table.get((state, action), 0.0) for action in possible_actions}
            
            # Find the maximum Q-value among possible actions
            max_q_value = max(q_values.values())
            
            # Get all actions that have the maximum Q-value
            best_actions = [action for action, q in q_values.items() if q == max_q_value]
            
            # Choose one of the best actions randomly (to break ties)
            return random.choice(best_actions)

    def update_q_table(self, state, action, reward, next_state, next_action):
        """
        Updates the Q-value for a given state-action pair using the SARSA update rule.

        The rule is: Q(s,a) <- Q(s,a) + alpha * [R + gamma * Q(s',a') - Q(s,a)]

        Args:
            state: The state before the action was taken.
            action: The action that was taken.
            reward: The reward received from the action.
            next_state: The resulting state.
            next_action: The action that will be taken in the next state, according to the current policy.
        """
        # Current Q-value estimate
        old_value = self.q_table.get((state, action), 0.0)

        # Get the Q-value of the next state and the *next* action (this is the key part of SARSA)
        next_q_value = self.q_table.get((next_state, next_action), 0.0)

        # Apply the SARSA update rule
        new_value = old_value + self.alpha * (reward + self.gamma * next_q_value - old_value)
        self.q_table[(state, action)] = new_value

    def decay_epsilon(self):
        """
        Decays the exploration rate (epsilon) after each episode.
        This encourages the agent to explore less and exploit more as it learns.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
