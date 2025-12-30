"""
This package contains modules related to the simulation environment and external data clients.

It includes:
- DeliveryEnvironment: The main environment class that manages state, actions, and rewards.
- OSMClient: A client for interacting with OpenStreetMap APIs (Nominatim).
- OSMNXClient: A client for building and using local road network graphs with OSMnx.
"""

from .environment import DeliveryEnvironment
from .osm_client import OSMClient, OSMNXClient

# Defines the public API for the 'world' package.
__all__ = ['DeliveryEnvironment', 'OSMClient', 'OSMNXClient']
