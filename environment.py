import numpy as np
from google_maps import GoogleMapsClient
from osm_client import OSMClient, OSMNXClient

class DeliveryEnvironment:
    def __init__(self, addresses=None, locations=None, city_name=None, map_provider='google', api_key=None, distance_metric='driving'):
        self.distance_metric = distance_metric
        self.city_name = city_name
        self.osmnx_client = None
        self.nodes = None # To store osmnx graph nodes

        if locations is not None:
            self.locations = np.array(locations)
            self.addresses = [f"Location {i}" for i in range(len(locations))]
        elif addresses is not None:
            self.addresses = addresses
            if map_provider == 'osm' or self.distance_metric != 'driving':
                geocoding_client = OSMClient()
            else:
                geocoding_client = GoogleMapsClient(api_key)
            self.locations = self._get_locations_from_addresses(geocoding_client)
        else:
            raise ValueError("You must provide either 'addresses' or 'locations'.")

        self.num_locations = len(self.locations)
        
        if self.num_locations > 0:
            self.distance_matrix = self._get_distance_matrix(map_provider, api_key)
        else:
            self.distance_matrix = np.array([])

        self.start_pos_index = 0
        self.current_pos_index = self.start_pos_index
        self.remaining_locations = set(range(1, self.num_locations))
        self.time_step = 0

    def get_state_size(self):
        return self.num_locations + self.num_locations

    def _get_locations_from_addresses(self, client):
        locations = []
        print("Geocoding addresses...")
        for address in self.addresses:
            lat, lng = client.get_coordinates(address)
            if lat is not None and lng is not None:
                locations.append((lat, lng))
        return np.array(locations)

    def _get_distance_matrix(self, map_provider, api_key):
        if self.distance_metric == 'network':
            if not self.city_name:
                raise ValueError("A 'city_name' must be provided for the 'network' distance metric.")
            self.osmnx_client = OSMNXClient(self.city_name)
            matrix, self.nodes = self.osmnx_client.get_distance_matrix(self.locations)
            return matrix

        if self.distance_metric == 'manhattan':
            return self._manhattan_distance_matrix()
        
        if self.distance_metric == 'haversine':
            return self._haversine_distance_matrix()

        if map_provider == 'osm':
            client = OSMClient()
            matrix = client.get_distance_matrix(self.locations)
        else:
            client = GoogleMapsClient(api_key)
            origins = [tuple(loc) for loc in self.locations]
            matrix_response = client.get_distance_matrix(origins, origins)
            
            num_locs = self.num_locations
            matrix = np.zeros((num_locs, num_locs))
            if matrix_response:
                for i in range(num_locs):
                    for j in range(num_locs):
                        element = matrix_response['rows'][i]['elements'][j]
                        if element['status'] == 'OK':
                            matrix[i][j] = element['distance']['value']
                        else:
                            matrix[i][j] = self._haversine_distance(self.locations[i], self.locations[j]) * 1000
            else:
                matrix = None

        if matrix is None:
            print("Warning: Driving distance API failed. Falling back to Haversine distance.")
            return self._haversine_distance_matrix()
            
        return matrix

    def _haversine_distance(self, coord1, coord2):
        R = 6371
        lat1, lon1 = np.radians(coord1)
        lat2, lon2 = np.radians(coord2)
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def _haversine_distance_matrix(self):
        num_locs = self.num_locations
        matrix = np.zeros((num_locs, num_locs))
        for i in range(num_locs):
            for j in range(i + 1, num_locs):
                dist = self._haversine_distance(self.locations[i], self.locations[j]) * 1000
                matrix[i][j] = dist
                matrix[j][i] = dist
        return matrix

    def _manhattan_distance(self, coord1, coord2):
        lat_diff = abs(coord1[0] - coord2[0])
        lon_diff = abs(coord1[1] - coord2[1])
        
        lat_rad = np.radians((coord1[0] + coord2[0]) / 2)
        m_per_deg_lat = 111132.954 - 559.822 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
        m_per_deg_lon = 111320 * np.cos(lat_rad)
        
        return (lat_diff * m_per_deg_lat) + (lon_diff * m_per_deg_lon)

    def _manhattan_distance_matrix(self):
        num_locs = self.num_locations
        matrix = np.zeros((num_locs, num_locs))
        for i in range(num_locs):
            for j in range(i + 1, num_locs):
                dist = self._manhattan_distance(self.locations[i], self.locations[j])
                matrix[i][j] = dist
                matrix[j][i] = dist
        return matrix

    def reset(self, vectorized=False):
        self.current_pos_index = self.start_pos_index
        self.remaining_locations = set(range(1, self.num_locations))
        self.time_step = 0
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
        self.time_step += 1
        
        done = len(self.remaining_locations) == 0
        if done:
            reward -= self.distance_matrix[self.current_pos_index][self.start_pos_index]

        return self._get_state(), reward, done

    def get_possible_actions(self):
        return list(self.remaining_locations)
