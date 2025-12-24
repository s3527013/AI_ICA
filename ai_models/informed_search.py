"""
Informed search algorithm agents for solving the Traveling Salesperson Problem (TSP).
"""


class InformedSearchAgent:
    """
    Base class for an informed search agent that finds a route in the environment.
    """

    def __init__(self, name):
        self.name = name

    def solve(self, env):
        """
        Solves the TSP for the given environment.

        Args:
            env (DeliveryEnvironment): The environment containing the distance matrix and locations.

        Returns:
            A tuple containing:
            - list: The optimized route as a sequence of location indices.
            - float: The total distance of the route in kilometers.
        """
        raise NotImplementedError("Each search agent must implement the solve method.")

    def get_name(self):
        return self.name
