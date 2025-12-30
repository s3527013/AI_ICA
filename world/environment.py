"""
This module defines the DeliveryEnvironment class, which simulates the delivery
problem for reinforcement learning agents.
"""

import numpy as np

from .osm_client import OSMClient, OSMNXClient


class DeliveryEnvironment:
    """
    A simulation environment for the delivery route optimization problem (a variant of the TSP).

    This class manages the state of the simulation, including the locations, distance matrix,
    and the agent's current position. It provides methods for agents to interact with the
    environment by taking actions and receiving rewards.
    """

    def __init__(self, addresses: list = None, locations: np.ndarray = None, city_name: str = None,
                 distance_metric: str = 'network'):
        """
        Initializes the delivery environment.

        Args:
            addresses (list, optional): A list of address strings to be geocoded.
            locations (np.ndarray, optional): A NumPy array of (latitude, longitude) coordinates.
            city_name (str, optional): The name of the city, required for the 'network' distance metric.
            distance_metric (str): The metric to use for calculating distances ('network' or 'haversine').
        """
        self.distance_metric = distance_metric
        self.city_name = city_name
        self.osmnx_client = None
        self.nodes = None  # To store the OSMnx graph nodes corresponding to locations

        if locations is not None:
            self.locations = np.array(locations)
            self.addresses = [f"Location {i}" for i in range(len(locations))]
        elif addresses is not None:
            self.addresses = addresses
            print("Geocoding addresses to coordinates...")
            geocoding_client = OSMClient()
            self.locations = self._get_locations_from_addresses(geocoding_client)
            print(f"  ✓ Successfully geocoded {len(self.locations)} out of {len(addresses)} addresses.")
        else:
            raise ValueError("You must provide either 'addresses' or 'locations'.")

        self.num_locations = len(self.locations)

        if self.num_locations > 0:
            self.distance_matrix = self._get_distance_matrix()
            self._sanitize_distance_matrix()
        else:
            self.distance_matrix = np.array([])

        # Initialize the state of the environment
        self.start_pos_index = 0
        self.current_pos_index = self.start_pos_index
        self.remaining_locations = set(range(1, self.num_locations))

    def _sanitize_distance_matrix(self):
        """Replaces infinite distances with a large finite penalty."""
        if self.distance_matrix is None:
            return

        # Find all finite values to determine a suitable penalty
        finite_vals = self.distance_matrix[np.isfinite(self.distance_matrix)]

        if len(finite_vals) > 0:
            max_dist = np.max(finite_vals)
            # Penalty should be significantly larger than any real path
            penalty = max_dist * 10.0
        else:
            # Fallback penalty if all distances are infinite (unlikely)
            penalty = 1e9

        self.distance_matrix[np.isinf(self.distance_matrix)] = penalty

    def get_state_size(self) -> int:
        """Returns the size of the vectorized state representation."""
        # State is a concatenation of a one-hot vector for current position and a binary mask for remaining locations
        return self.num_locations * 2

    def _get_locations_from_addresses(self, client: OSMClient) -> np.ndarray:
        """Converts a list of address strings to a NumPy array of coordinates."""
        locations = []
        for address in self.addresses:
            lat, lng = client.get_coordinates(address)
            if lat is not None and lng is not None:
                locations.append((lat, lng))
            else:
                print(f"    - Warning: Could not geocode address '{address}'. It will be skipped.")
        return np.array(locations)

    def _get_distance_matrix(self) -> np.ndarray:
        """Calculates the distance matrix based on the chosen metric."""
        if self.distance_metric == 'network':
            if not self.city_name:
                raise ValueError("A 'city_name' must be provided for the 'network' distance metric.")
            self.osmnx_client = OSMNXClient(self.city_name)
            matrix, self.nodes = self.osmnx_client.get_distance_matrix(self.locations)
            return matrix
        elif self.distance_metric == 'haversine':
            return self._haversine_distance_matrix()
        else:
            print(f"Warning: Unknown distance metric '{self.distance_metric}'. Defaulting to 'haversine'.")
            return self._haversine_distance_matrix()

    def _haversine_distance_matrix(self) -> np.ndarray:
        """Calculates the distance matrix using the Haversine (great-circle) formula."""
        num_locs = self.num_locations
        matrix = np.zeros((num_locs, num_locs))
        # Convert degrees to radians
        locs_rad = np.radians(self.locations)
        for i in range(num_locs):
            for j in range(i + 1, num_locs):
                lat1, lon1 = locs_rad[i]
                lat2, lon2 = locs_rad[j]
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                
                # Haversine formula
                a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                distance = 6371000 * c  # Earth radius in meters
                
                matrix[i, j] = distance
                matrix[j, i] = distance
        return matrix

    def reset(self, vectorized: bool = False):
        """Resets the environment to its initial state."""
        self.current_pos_index = self.start_pos_index
        self.remaining_locations = set(range(1, self.num_locations))
        return self._get_state(vectorized)

    def _get_state(self, vectorized: bool = False):
        """Returns the current state of the environment."""
        if vectorized:
            # For DQN: a one-hot vector for current location and a binary mask for remaining locations
            current_loc_one_hot = np.zeros(self.num_locations)
            current_loc_one_hot[self.current_pos_index] = 1
            remaining_mask = np.zeros(self.num_locations)
            for loc_idx in self.remaining_locations:
                remaining_mask[loc_idx] = 1
            return np.concatenate([current_loc_one_hot, remaining_mask])
        else:
            # For tabular methods: a tuple of current position and sorted remaining locations
            return self.current_pos_index, tuple(sorted(list(self.remaining_locations)))

    def step(self, action_index: int):
        """
        Executes one time step in the environment.

        Args:
            action_index (int): The index of the location to travel to.

        Returns:
            A tuple containing (next_state, reward, done).
        """
        # Penalize heavily for invalid actions (e.g., visiting an already visited location)
        if action_index not in self.remaining_locations:
            return self._get_state(), -10000, True

        distance = self.distance_matrix[self.current_pos_index][action_index]
        # Reward is the negative of the distance traveled
        reward = -distance
        
        self.current_pos_index = action_index
        self.remaining_locations.remove(action_index)
        
        done = len(self.remaining_locations) == 0
        if done:
            # If the tour is complete, add the cost of returning to the depot
            reward -= self.distance_matrix[self.current_pos_index][self.start_pos_index]
            
        return self._get_state(), reward, done

    def get_possible_actions(self) -> list:
        """Returns a list of valid actions from the current state."""
        return list(self.remaining_locations)

    def get_environment_summary(self) -> dict:
        """Returns a dictionary summarizing the environment's configuration."""
        return {
            "city": self.city_name,
            "num_locations": self.num_locations,
            "distance_metric": self.distance_metric,
            "depot_location": self.addresses[0] if self.addresses else "N/A"
        }
