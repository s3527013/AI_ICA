import numpy as np
from osm_client import OSMClient, OSMNXClient

class DeliveryEnvironment:
    def __init__(self, addresses=None, locations=None, city_name=None, distance_metric='network'):
        self.distance_metric = distance_metric
        self.city_name = city_name
        self.osmnx_client = None
        self.nodes = None

        if locations is not None:
            self.locations = np.array(locations)
            self.addresses = [f"Location {i}" for i in range(len(locations))]
        elif addresses is not None:
            self.addresses = addresses
            geocoding_client = OSMClient()
            self.locations = self._get_locations_from_addresses(geocoding_client)
        else:
            raise ValueError("You must provide either 'addresses' or 'locations'.")

        self.num_locations = len(self.locations)

        if self.num_locations > 0:
            self.distance_matrix = self._get_distance_matrix()
            self._sanitize_distance_matrix()
        else:
            self.distance_matrix = np.array([])

        self.start_pos_index = 0
        self.current_pos_index = self.start_pos_index
        self.remaining_locations = set(range(1, self.num_locations))

    def _sanitize_distance_matrix(self):
        """Replaces infinite distances with a large finite penalty."""
        if self.distance_matrix is None:
            return

        finite_vals = self.distance_matrix[np.isfinite(self.distance_matrix)]

        if len(finite_vals) > 0:
            max_dist = np.max(finite_vals)
            penalty = max_dist * 10.0
        else:
            penalty = 100000.0

        self.distance_matrix[np.isinf(self.distance_matrix)] = penalty

    def get_state_size(self):
        return self.num_locations * 2

    def _get_locations_from_addresses(self, client):
        locations = []
        for address in self.addresses:
            lat, lng = client.get_coordinates(address)
            if lat is not None and lng is not None:
                locations.append((lat, lng))
        return np.array(locations)

    def _get_distance_matrix(self):
        if self.distance_metric == 'network':
            if not self.city_name: raise ValueError("A 'city_name' must be provided for 'network' metric.")
            self.osmnx_client = OSMNXClient(self.city_name)
            matrix, self.nodes = self.osmnx_client.get_distance_matrix(self.locations)
            return matrix
        elif self.distance_metric == 'haversine':
            return self._haversine_distance_matrix()
        else: # Fallback to haversine
            return self._haversine_distance_matrix()

    def _haversine_distance_matrix(self):
        num_locs = self.num_locations
        matrix = np.zeros((num_locs, num_locs))
        for i in range(num_locs):
            for j in range(i + 1, num_locs):
                lat1, lon1 = self.locations[i]
                lat2, lon2 = self.locations[j]
                R = 6371000  # Earth radius in meters
                phi1 = np.radians(lat1)
                phi2 = np.radians(lat2)
                delta_phi = np.radians(lat2 - lat1)
                delta_lambda = np.radians(lon2 - lon1)
                a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                dist = R * c
                matrix[i][j] = dist
                matrix[j][i] = dist
        return matrix

    def reset(self, vectorized=False):
        self.current_pos_index = self.start_pos_index
        self.remaining_locations = set(range(1, self.num_locations))
        return self._get_state(vectorized)

    def _get_state(self, vectorized=False):
        if vectorized:
            current_loc_one_hot = np.zeros(self.num_locations)
            current_loc_one_hot[self.current_pos_index] = 1
            remaining_mask = np.zeros(self.num_locations)
            for loc_idx in self.remaining_locations:
                remaining_mask[loc_idx] = 1
            return np.concatenate([current_loc_one_hot, remaining_mask])
        else:
            return self.current_pos_index, tuple(sorted(list(self.remaining_locations)))

    def step(self, action_index):
        if action_index not in self.remaining_locations:
            return self._get_state(), -10000, True

        distance = self.distance_matrix[self.current_pos_index][action_index]
        reward = -distance
        self.current_pos_index = action_index
        self.remaining_locations.remove(action_index)
        done = len(self.remaining_locations) == 0
        if done:
            reward -= self.distance_matrix[self.current_pos_index][self.start_pos_index]
        return self._get_state(), reward, done

    def get_possible_actions(self):
        return list(self.remaining_locations)

    def get_state_description(self, state_tuple=None):
        if state_tuple is None:
            current_idx, remaining = self._get_state()
        else:
            current_idx, remaining = state_tuple

        current_address = self.addresses[current_idx]
        remaining_count = len(remaining)

        return (f"**Current Location**: {current_address} (Index: {current_idx})\\n"
                f"**Deliveries Remaining**: {remaining_count} out of {self.num_locations - 1}")

    def get_action_description(self, action_index):
        if 0 <= action_index < self.num_locations:
            return f"**Travel to**: {self.addresses[action_index]} (Index: {action_index})"
        return "Unknown Action"

    def get_environment_summary(self):
        return {
            "city": self.city_name,
            "num_locations": self.num_locations,
            "distance_metric": self.distance_metric,
            "depot_location": self.addresses[0]
        }
