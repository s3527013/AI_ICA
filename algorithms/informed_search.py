"""
This module defines the base class for all informed search algorithm agents
used to solve the Traveling Salesperson Problem (TSP).
"""


class InformedSearchAgent:
    """
    Abstract base class for an informed search agent.

    This class provides a common interface for different search-based TSP solvers.
    Each subclass must implement the `solve` method, which contains the core
    logic for finding an optimized delivery route.
    """

    def __init__(self, name: str):
        """
        Initializes the agent with a given name.

        Args:
            name (str): The name of the search algorithm (e.g., "A* Search").
        """
        self.name = name

    def solve(self, env):
        """
        Solves the TSP for the given environment. This method must be implemented by subclasses.

        Args:
            env (DeliveryEnvironment): The environment instance, which contains the
                                     distance matrix and location data required for the search.

        Returns:
            A tuple containing:
            - list: The optimized route as a sequence of location indices.
            - float: The total distance of the route in kilometers.
        
        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Each search agent must implement the `solve` method.")

    def get_name(self) -> str:
        """
        Returns the name of the agent.

        Returns:
            str: The name of the agent.
        """
        return self.name
