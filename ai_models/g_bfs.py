from .informed_search import InformedSearchAgent


class GreedyBestFirstSearchAgent(InformedSearchAgent):
    """
    Solves the TSP using a Greedy Best-First Search approach. This is essentially
    the Nearest Neighbor heuristic, as it always chooses the closest unvisited node.
    """

    def __init__(self):
        super().__init__("Greedy Best-First Search")

    def solve(self, env):
        current_pos = env.start_pos_index
        route = [current_pos]
        unvisited = set(range(env.num_locations))
        unvisited.remove(current_pos)
        total_distance = 0

        while unvisited:
            # Greedily choose the nearest unvisited neighbor
            nearest_neighbor = min(unvisited, key=lambda x: env.distance_matrix[current_pos][x])

            total_distance += env.distance_matrix[current_pos][nearest_neighbor]
            current_pos = nearest_neighbor
            route.append(current_pos)
            unvisited.remove(current_pos)

        # Return to the depot to complete the loop
        total_distance += env.distance_matrix[current_pos][env.start_pos_index]
        route.append(env.start_pos_index)

        return route, total_distance / 1000  # Convert distance to km
