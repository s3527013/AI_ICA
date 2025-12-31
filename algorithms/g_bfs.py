"""
This module implements the Greedy Best-First Search agent for the Traveling Salesperson Problem.
"""

from .informed_search import InformedSearchAgent


class GreedyBestFirstSearchAgent(InformedSearchAgent):
    """
    Solves the TSP using a Greedy Best-First Search approach.

    This implementation is functionally equivalent to the Nearest Neighbor heuristic.
    At each step, it greedily chooses the closest unvisited node as the next
    destination, without considering the long-term impact of that choice.
    """

    def __init__(self):
        """Initializes the Greedy Best-First Search agent."""
        super().__init__("Greedy Best-First Search")

    def solve(self, env):
        """
        Finds a route by always moving to the nearest unvisited location.

        Args:
            env (DeliveryEnvironment): The environment containing the distance matrix.

        Returns:
            A tuple containing:
            - list: The calculated route as a sequence of location indices.
            - float: The total distance of the route in kilometers.
        """
        # Start at the depot
        current_pos = env.start_pos_index
        route = [current_pos]
        unvisited = set(range(env.num_locations))
        unvisited.remove(current_pos)
        total_distance = 0

        # Sequentially visit the nearest unvisited node
        while unvisited:
            # Find the nearest neighbor from the set of unvisited nodes
            nearest_neighbor = min(unvisited, key=lambda x: env.distance_matrix[current_pos][x])

            # Update distance and move to the new location
            total_distance += env.distance_matrix[current_pos][nearest_neighbor]
            current_pos = nearest_neighbor
            route.append(current_pos)
            unvisited.remove(current_pos)

        # Complete the tour by returning to the depot
        total_distance += env.distance_matrix[current_pos][env.start_pos_index]
        route.append(env.start_pos_index)

        # Return the final route and distance in kilometers
        return route, total_distance / 1000
