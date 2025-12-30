"""
This module provides clients for interacting with OpenStreetMap (OSM) data.
- OSMClient: Uses online services (Nominatim) for geocoding.
- OSMNXClient: Uses the OSMnx library to build a local road network graph for routing.
"""

import os
import time

import numpy as np
from geopy.geocoders import Nominatim

# Conditionally import osmnx and networkx to avoid hard dependencies
try:
    import osmnx as ox
    import networkx as nx
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False
    print("Warning: osmnx or networkx not installed. The 'network' distance metric will not be available.")


class OSMClient:
    """
    A client for interacting with the public OpenStreetMap Nominatim API for geocoding services.
    """

    def __init__(self):
        """Initializes the Nominatim geolocator."""
        self.geolocator = Nominatim(user_agent="delivery_route_optimizer")

    def get_coordinates(self, address: str) -> tuple:
        """
        Geocodes a given address string to latitude and longitude coordinates.

        Args:
            address (str): The address to geocode.

        Returns:
            tuple: A tuple containing (latitude, longitude), or (None, None) if not found.
        """
        try:
            # Add a delay to respect the API's usage policy (1 request per second)
            time.sleep(1)
            location = self.geolocator.geocode(address)
            if location:
                return location.latitude, location.longitude
        except Exception as e:
            print(f"Error geocoding '{address}' with Nominatim: {e}")
        return None, None

    def get_bounding_box(self, city_name: str) -> list:
        """
        Gets the bounding box coordinates for a given city name.

        Args:
            city_name (str): The name of the city.

        Returns:
            list: A list containing [min_lat, max_lat, min_lon, max_lon], or None if not found.
        """
        try:
            time.sleep(1)
            location = self.geolocator.geocode(city_name, exactly_one=True)
            if location and 'boundingbox' in location.raw:
                return [float(x) for x in location.raw['boundingbox']]
        except Exception as e:
            print(f"Could not get bounding box for '{city_name}': {e}")
        return None


class OSMNXClient:
    """
    A client that uses the OSMnx library to build and query a local road network graph.
    Includes caching to save and load the graph, speeding up subsequent runs.
    """

    def __init__(self, place_name: str):
        """
        Initializes the OSMNXClient, loading the graph from cache or downloading it.

        Args:
            place_name (str): The name of the place (e.g., "Middlesbrough, UK") to get the graph for.
        """
        if not OSMNX_AVAILABLE:
            raise ImportError("osmnx is required to use the OSMNXClient.")

        # Define the directory and file path for the cached graph
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        filepath = os.path.join(data_dir, f"{place_name.replace(' ', '_').lower()}.graphml")

        if os.path.exists(filepath):
            # Load the graph from the file if it exists
            print(f"Loading cached graph for '{place_name}' from {filepath}...")
            self.G = ox.io.load_graphml(filepath)
            print("  ✓ Graph loaded from cache.")
        else:
            # If no cached graph, download it from OpenStreetMap
            print(f"Downloading street network for '{place_name}'...")
            G_full = ox.graph_from_place(place_name, network_type='drive')
            
            # Keep only the largest connected component to ensure the graph is fully navigable
            largest_component = max(nx.weakly_connected_components(G_full), key=len)
            self.G = G_full.subgraph(largest_component).copy()
            
            # Save the processed graph to the cache for future use
            print(f"  Saving graph to {filepath} for future use...")
            ox.io.save_graphml(self.G, filepath)
            print("  ✓ Graph saved to cache.")

        print(f"Graph ready with {len(self.G.nodes)} nodes and {len(self.G.edges)} edges.")

    def get_nearest_nodes(self, locations: np.ndarray) -> list:
        """
        Finds the nearest graph nodes for a list of (latitude, longitude) coordinates.

        Args:
            locations (np.ndarray): An array of coordinates.

        Returns:
            list: A list of the nearest OSMnx node IDs.
        """
        lats = locations[:, 0]
        lons = locations[:, 1]
        return ox.nearest_nodes(self.G, lons, lats)

    def get_distance_matrix(self, locations: np.ndarray) -> tuple:
        """
        Calculates the all-pairs shortest-path distance matrix for a set of locations.

        Args:
            locations (np.ndarray): An array of coordinates.

        Returns:
            tuple: A tuple containing (distance_matrix, node_ids).
                   The matrix contains distances in meters.
        """
        nodes = self.get_nearest_nodes(locations)
        n = len(nodes)
        matrix = np.full((n, n), fill_value=np.inf)

        print("Calculating network distance matrix (this may take a while)...")
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 0
                    continue
                try:
                    # Calculate the shortest path length in meters
                    length = nx.shortest_path_length(self.G, nodes[i], nodes[j], weight='length')
                    matrix[i, j] = length
                except nx.NetworkXNoPath:
                    # Keep as infinity if no path exists
                    continue
        
        print("  ✓ Distance matrix calculation complete.")
        return matrix, nodes
