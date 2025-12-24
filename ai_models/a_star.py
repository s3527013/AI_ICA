import heapq

from .informed_search import InformedSearchAgent


class AStarAgent(InformedSearchAgent):
    """
    Solves the TSP using A* Search. This agent finds the optimal route by considering both the
    cost to reach the current node (g-cost) and a heuristic estimate of the cost to the goal (h-cost).
    """

    def __init__(self):
        super().__init__("A* Search")

    def solve(self, env):
        start_node = env.start_pos_index
        num_locations = env.num_locations

        # The state is (current_cost, current_node, path_so_far, visited_mask)
        # The priority queue will store tuples of (estimated_total_cost, current_cost, current_node, path, visited_mask)
        pq = [(0, 0, start_node, [start_node], 1 << start_node)]

        # A dictionary to keep track of the minimum cost to reach a state (node, visited_mask)
        min_costs = {(start_node, 1 << start_node): 0}

        while pq:
            _, g_cost, current_node, path, visited_mask = heapq.heappop(pq)

            # If we have visited all locations, return to the depot and complete the path
            if len(path) == num_locations:
                final_path = path + [start_node]
                total_distance = g_cost + env.distance_matrix[current_node][start_node]
                return final_path, total_distance / 1000  # Convert to km

            # Explore neighbors
            for neighbor in range(num_locations):
                # Check if the neighbor has not been visited yet
                if not (visited_mask & (1 << neighbor)):
                    new_g_cost = g_cost + env.distance_matrix[current_node][neighbor]
                    new_visited_mask = visited_mask | (1 << neighbor)

                    # If we found a cheaper path to this state, update it
                    if min_costs.get((neighbor, new_visited_mask), float('inf')) > new_g_cost:
                        min_costs[(neighbor, new_visited_mask)] = new_g_cost

                        # Heuristic: Minimum Spanning Tree (MST) of remaining nodes
                        remaining_nodes = [n for n in range(num_locations) if not (new_visited_mask & (1 << n))]
                        h_cost = self._mst_heuristic(env, remaining_nodes + [neighbor]) if remaining_nodes else 0

                        f_cost = new_g_cost + h_cost
                        new_path = path + [neighbor]
                        heapq.heappush(pq, (f_cost, new_g_cost, neighbor, new_path, new_visited_mask))

        return [], float('inf')  # Should not be reached if a path exists

    def _mst_heuristic(self, env, nodes):
        """
        Heuristic for A* search using the cost of the Minimum Spanning Tree (MST)
        of the unvisited nodes. This is an admissible heuristic for the TSP.
        """
        if not nodes or len(nodes) < 2:
            return 0

        # Prim's algorithm for MST
        mst_cost = 0
        key = {node: float('inf') for node in nodes}
        parent = {node: None for node in nodes}
        in_mst = {node: False for node in nodes}

        key[nodes[0]] = 0

        for _ in range(len(nodes)):
            # Find the node with the smallest key value, from the set of nodes not yet in MST
            min_key = float('inf')
            u = -1
            for v_node in nodes:
                if not in_mst[v_node] and key[v_node] < min_key:
                    min_key = key[v_node]
                    u = v_node

            if u == -1: break  # All remaining vertices are inaccessible

            in_mst[u] = True
            mst_cost += min_key

            for v_node in nodes:
                if not in_mst[v_node] and env.distance_matrix[u][v_node] < key[v_node]:
                    key[v_node] = env.distance_matrix[u][v_node]
                    parent[v_node] = u

        return mst_cost
