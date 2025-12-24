import random
from collections import defaultdict


class QLearningAgent:
    def __init__(self, action_space, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 state_space=None):
        """
        Q-Learning Agent.

        Args:
            action_space (list): A list of all possible actions.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Initial exploration rate.
            epsilon_decay (float): Rate at which epsilon decays.
            epsilon_min (float): Minimum value for epsilon.
            state_space: Not used in this implementation as the Q-table is dynamic,
                         but kept for potential future compatibility.
        """
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Use a defaultdict for a dynamically growing Q-table
        self.q_table = defaultdict(lambda: 0.0)

    def choose_action(self, state, possible_actions):
        """
        Choose an action using an epsilon-greedy policy.

        Args:
            state: The current state of the environment.
            possible_actions (list): A list of valid actions from the current state.

        Returns:
            The chosen action.
        """
        if not possible_actions:
            return None

        if random.uniform(0, 1) < self.epsilon:
            # Exploration: choose a random action from possible actions
            return random.choice(possible_actions)
        else:
            # Exploitation: choose the best action from the Q-table
            q_values = {action: self.q_table.get((state, action), 0.0) for action in possible_actions}
            max_q_value = max(q_values.values())
            # In case of a tie, randomly choose among the best actions
            best_actions = [action for action, q in q_values.items() if q == max_q_value]
            return random.choice(best_actions)

    def update_q_table(self, state, action, reward, next_state, next_possible_actions=None):
        """
        Update the Q-table using the Bellman equation.

        Args:
            state: The state before the action.
            action: The action taken.
            reward: The reward received.
            next_state: The state after the action.
            next_possible_actions (list, optional): Possible actions from the next state.
        """
        # Get current Q-value
        old_value = self.q_table.get((state, action), 0.0)

        # Get the maximum Q-value for the next state
        if next_possible_actions:
            next_max = max([self.q_table.get((next_state, a), 0.0) for a in next_possible_actions], default=0.0)
        else:
            # This is a terminal state
            next_max = 0.0

        # Calculate the new Q-value
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[(state, action)] = new_value

    def decay_epsilon(self):
        """
        Decay the exploration rate (epsilon).
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
