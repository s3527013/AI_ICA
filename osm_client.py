import time
import requests
import numpy as np
from geopy.geocoders import Nominatim

# Try importing osmnx and networkx, handle if not present
try:
    import osmnx as ox
    import networkx as nx
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False
    print("Warning: osmnx or networkx not installed. 'network' distance metric will not work.")

class OSMClient:
    """
    A client for OpenStreetMap services (Nominatim for geocoding, OSRM for routing).
    """
    def __init__(self):
        self.geolocator = Nominatim(user_agent="delivery_route_optimizer")
        self.osrm_endpoint = "http://router.project-osrm.org"

    def get_coordinates(self, address):
        try:
            time.sleep(1)
            location = self.geolocator.geocode(address)
            if location:
                return location.latitude, location.longitude
        except Exception as e:
            print(f"Error geocoding {address} with Nominatim: {e}")
        return None, None

    def get_bounding_box(self, city_name):
        try:
            time.sleep(1)
            location = self.geolocator.geocode(city_name, exactly_one=True)
            if location and 'boundingbox' in location.raw:
                return [float(x) for x in location.raw['boundingbox']]
        except Exception as e:
            print(f"Could not get bounding box for {city_name}: {e}")
        return None

    def get_distance_matrix(self, locations):
        if len(locations) < 2:
            return np.zeros((len(locations), len(locations)))

        coords_str = ";".join([f"{lon},{lat}" for lat, lon in locations])
        url = f"{self.osrm_endpoint}/table/v1/driving/{coords_str}?annotations=distance"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if data['code'] == 'Ok' and 'distances' in data:
                return np.array(data['distances'])
            else:
                print(f"OSRM API error: {data.get('message', 'Unknown error')}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error getting distance matrix from OSRM: {e}")
            return None

class OSMNXClient:
    """
    A client for using OSMnx to calculate distances on the local graph.
    """
    def __init__(self, place_name):
        if not OSMNX_AVAILABLE:
            raise ImportError("osmnx is not installed.")
        
        print(f"Downloading street network for {place_name}...")
        # Download the graph for driving
        G_full = ox.graph_from_place(place_name, network_type='drive')
        
        # Keep only the largest connected component to avoid isolated nodes
        largest_component = max(nx.weakly_connected_components(G_full), key=len)
        self.G = G_full.subgraph(largest_component).copy()
        
        print(f"Graph downloaded and processed. Using the largest connected component with {len(self.G.nodes)} nodes.")

    def get_nearest_nodes(self, locations):
        """
        Find the nearest graph nodes for a list of (lat, lon) coordinates.
        """
        lats = [loc[0] for loc in locations]
        lons = [loc[1] for loc in locations]
        return ox.nearest_nodes(self.G, lons, lats)

    def get_distance_matrix(self, locations):
        """
        Calculate the distance matrix using network shortest paths.
        """
        nodes = self.get_nearest_nodes(locations)
        n = len(nodes)
        matrix = np.zeros((n, n))
        
        print("Calculating network distance matrix (this may take a while)...")
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                try:
                    # Calculate shortest path length in meters
                    length = nx.shortest_path_length(self.G, nodes[i], nodes[j], weight='length')
                    matrix[i][j] = length
                except nx.NetworkXNoPath:
                    # If no path, set to infinity. This will be handled in the environment.
                    matrix[i][j] = float('inf')
        
        return matrix, nodes
