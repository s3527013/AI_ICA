"""
This module implements the Q-Learning agent, an off-policy temporal-difference (TD)
learning algorithm.
"""

import random
from collections import defaultdict


class QLearningAgent:
    """
    A reinforcement learning agent that learns a policy using the Q-Learning algorithm.

    Q-Learning is an off-policy algorithm, which means it learns the value of the
    optimal policy independently of the agent's actions. It does this by always
    choosing the maximum Q-value for the next state when updating its Q-table,
    regardless of which action was actually taken.
    """

    def __init__(self, action_space: list, alpha: float = 0.1, gamma: float = 0.9,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        """
        Initializes the Q-Learning Agent.

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
        # A defaultdict is used so that new states are automatically initialized with a Q-value of 0.0.
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

        # Exploration phase
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(possible_actions)
        
        # Exploitation phase
        else:
            # Get Q-values for all possible actions from the current state
            q_values = {action: self.q_table.get((state, action), 0.0) for action in possible_actions}
            
            # Find the maximum Q-value among the possible actions
            max_q_value = max(q_values.values())
            
            # Collect all actions that have this maximum Q-value
            best_actions = [action for action, q in q_values.items() if q == max_q_value]
            
            # Randomly choose one of the best actions to break ties
            return random.choice(best_actions)

    def update_q_table(self, state, action, reward, next_state, next_possible_actions: list = None):
        """
        Updates the Q-value for a given state-action pair using the Q-Learning update rule.

        The rule is: Q(s,a) <- Q(s,a) + alpha * [R + gamma * max_a' Q(s',a') - Q(s,a)]

        Args:
            state: The state before the action was taken.
            action: The action that was taken.
            reward: The reward received from the action.
            next_state: The resulting state.
            next_possible_actions (list, optional): All possible actions from the next state.
        """
        # Get the current Q-value estimate for the state-action pair
        old_value = self.q_table.get((state, action), 0.0)

        # Find the maximum Q-value for the next state among all possible next actions
        if next_possible_actions:
            next_max = max([self.q_table.get((next_state, a), 0.0) for a in next_possible_actions], default=0.0)
        else:
            # If there are no possible next actions, it's a terminal state
            next_max = 0.0

        # Apply the Q-Learning update rule
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[(state, action)] = new_value

    def decay_epsilon(self):
        """
        Decays the exploration rate (epsilon) after each episode.
        This encourages the agent to explore less and exploit more as it learns.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
