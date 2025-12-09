import random


class QLearningAgent:
    def __init__(self, environment, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.environment = environment
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def get_action(self, state):
        possible_actions = self.environment.get_possible_actions()
        if not possible_actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        else:
            q_values = [self.get_q_value(state, a) for a in possible_actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(possible_actions, q_values) if q == max_q]
            return random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        old_q_value = self.get_q_value(state, action)

        next_possible_actions = self.environment.get_possible_actions()
        if not next_possible_actions:
            next_max_q = 0.0
        else:
            next_q_values = [self.get_q_value(next_state, a) for a in next_possible_actions]
            next_max_q = max(next_q_values)

        new_q_value = old_q_value + self.alpha * (reward + self.gamma * next_max_q - old_q_value)
        self.q_table[(state, action)] = new_q_value
