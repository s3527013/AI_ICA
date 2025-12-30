"""
This module implements an A* (A-Star) search agent to solve the Traveling Salesperson Problem (TSP).
"""

import heapq

from .informed_search import InformedSearchAgent


class AStarAgent(InformedSearchAgent):
    """
    Solves the TSP using the A* search algorithm.

    A* is an informed search algorithm that finds the least-cost path from a start node
    to a goal node. It balances the cost to reach the current node (g-cost) and a
    heuristic estimate of the cost from the current node to the goal (h-cost).

    The state in this implementation is defined by the current node and a bitmask
    representing the set of visited nodes. The heuristic used is the cost of the
    Minimum Spanning Tree (MST) of all unvisited nodes, which is an admissible
    heuristic for the TSP (it never overestimates the true cost).
    """

    def __init__(self):
        """Initializes the A* Search agent."""
        super().__init__("A* Search")

    def solve(self, env):
        """
        Executes the A* search algorithm to find the optimal route.

        Args:
            env (DeliveryEnvironment): The environment containing the distance matrix.

        Returns:
            A tuple containing:
            - list: The optimal route as a sequence of location indices.
            - float: The total distance of the optimal route in kilometers.
        """
        start_node = env.start_pos_index
        num_locations = env.num_locations

        # The priority queue stores tuples: (f_cost, g_cost, current_node, path, visited_mask)
        # f_cost = g_cost + h_cost (estimated total cost)
        # g_cost = actual cost from start to current_node
        pq = [(0, 0, start_node, [start_node], 1 << start_node)]

        # A dictionary to keep track of the minimum cost to reach a specific state (node, visited_mask)
        min_costs = {(start_node, 1 << start_node): 0}

        while pq:
            # Pop the node with the lowest f_cost
            _, g_cost, current_node, path, visited_mask = heapq.heappop(pq)

            # If the path includes all locations, the tour is complete.
            if len(path) == num_locations:
                # Add the final leg back to the depot
                final_path = path + [start_node]
                total_distance = g_cost + env.distance_matrix[current_node][start_node]
                return final_path, total_distance / 1000  # Convert to km

            # Explore neighbors of the current node
            for neighbor in range(num_locations):
                # If the neighbor has not been visited yet
                if not (visited_mask & (1 << neighbor)):
                    new_g_cost = g_cost + env.distance_matrix[current_node][neighbor]
                    new_visited_mask = visited_mask | (1 << neighbor)

                    # If we found a cheaper path to this state, update it and add to the queue
                    if min_costs.get((neighbor, new_visited_mask), float('inf')) > new_g_cost:
                        min_costs[(neighbor, new_visited_mask)] = new_g_cost

                        # Calculate the MST heuristic for the remaining unvisited nodes
                        remaining_nodes = [n for n in range(num_locations) if not (new_visited_mask & (1 << n))]
                        h_cost = self._mst_heuristic(env, remaining_nodes + [neighbor]) if remaining_nodes else 0
                        
                        f_cost = new_g_cost + h_cost
                        new_path = path + [neighbor]
                        heapq.heappush(pq, (f_cost, new_g_cost, neighbor, new_path, new_visited_mask))

        return [], float('inf')  # Return empty path if no solution is found

    def _mst_heuristic(self, env, nodes: list) -> float:
        """
        Calculates the Minimum Spanning Tree (MST) cost for a set of nodes.
        This serves as an admissible heuristic for the TSP.

        Args:
            env (DeliveryEnvironment): The environment with the distance matrix.
            nodes (list): A list of node indices for which to calculate the MST.

        Returns:
            float: The total weight of the MST.
        """
        if not nodes or len(nodes) < 2:
            return 0

        # Prim's algorithm for MST
        mst_cost = 0
        # key[i] stores the minimum weight edge to connect node i to the MST
        key = {node: float('inf') for node in nodes}
        # Set of nodes included in MST
        in_mst = {node: False for node in nodes}
        
        # Start with the first node
        key[nodes[0]] = 0
        
        for _ in range(len(nodes)):
            # Find the node with the smallest key value that is not yet in the MST
            min_key = float('inf')
            u = -1
            for v_node in nodes:
                if not in_mst[v_node] and key[v_node] < min_key:
                    min_key = key[v_node]
                    u = v_node
            
            if u == -1: break  # No more reachable nodes

            # Add the chosen node to the MST
            in_mst[u] = True
            mst_cost += min_key
            
            # Update the key values of adjacent nodes
            for v_node in nodes:
                if not in_mst[v_node] and env.distance_matrix[u][v_node] < key[v_node]:
                    key[v_node] = env.distance_matrix[u][v_node]

        return mst_cost
