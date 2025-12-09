import numpy as np

class DeliveryEnvironment:
    def __init__(self, num_locations=50, grid_size=50):
        self.num_locations = num_locations
        self.grid_size = grid_size
        self.locations = self._generate_locations()
        self.start_pos = self.locations[0]
        self.current_pos = self.start_pos
        self.remaining_locations = set(range(1, self.num_locations))
        self.time_step = 0

    def _generate_locations(self):
        return np.random.randint(0, self.grid_size, size=(self.num_locations, 2))

    def reset(self):
        self.current_pos = self.start_pos
        self.remaining_locations = set(range(1, self.num_locations))
        self.time_step = 0
        return self._get_state()

    def _get_state(self):
        return tuple(self.current_pos), tuple(sorted(list(self.remaining_locations)))

    def step(self, action):
        if action not in self.remaining_locations:
            raise ValueError("Invalid action")

        next_pos = self.locations[action]
        distance = np.linalg.norm(self.current_pos - next_pos)
        reward = -distance
        self.current_pos = next_pos
        self.remaining_locations.remove(action)
        self.time_step += 1
        
        done = len(self.remaining_locations) == 0
        return self._get_state(), reward, done

    def get_possible_actions(self):
        return list(self.remaining_locations)
