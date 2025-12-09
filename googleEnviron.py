import numpy as np
import pandas as pd

from environment import DeliveryEnvironment


class EnhancedDeliveryEnvironmentWithAI(DeliveryEnvironment):
    """
    Enhanced environment with methods for Google AI explanation.
    """

    def __init__(self, num_locations=10, grid_size=10):
        super().__init__(num_locations, grid_size)
        self.action_history = []
        self.state_history = []
        self.reward_history = []

    def get_state_description(self, state=None):
        """Get detailed state description for AI analysis"""
        if state is None:
            state = self.current_pos

        description = [
            f"**Spatial Information**:",
            f"- Current Position: ({state[0]:.1f}, {state[1]:.1f})",
            f"- Grid Size: {self.grid_size} x {self.grid_size}",
        ]

        if hasattr(self, 'delivery_locations'):
            remaining = len(self.delivery_locations) - len(self.delivered)
            description.extend([
                f"**Delivery Status**:",
                f"- Locations Delivered: {len(self.delivered)}",
                f"- Locations Remaining: {remaining}",
                f"- Completion: {(len(self.delivered) / len(self.delivery_locations) * 100):.1f}%"
            ])

            if remaining > 0:
                # Calculate distances to remaining locations
                distances = []
                for loc in self.delivery_locations:
                    if tuple(loc) not in self.delivered:
                        dist = np.linalg.norm(state - loc)
                        distances.append((dist, loc))

                if distances:
                    distances.sort()
                    description.append(f"**Nearest Delivery**:")
                    description.append(f"- Distance: {distances[0][0]:.2f}")
                    description.append(f"- Location: ({distances[0][1][0]:.1f}, {distances[0][1][1]:.1f})")

        return "\n".join(description)

    def get_action_description(self, action):
        """Get detailed action description"""
        action_descriptions = {
            0: "**Move North** - Decrease Y coordinate, moving upward on the grid",
            1: "**Move East** - Increase X coordinate, moving right on the grid",
            2: "**Move South** - Increase Y coordinate, moving downward on the grid",
            3: "**Move West** - Decrease X coordinate, moving left on the grid",
            4: "**Stay in Place** - No movement, possibly waiting or delivering",
            5: "**Deliver Package** - Attempt to deliver at current location"
        }

        return action_descriptions.get(action, f"Unknown Action: {action}")

    def get_environment_summary(self):
        """Get comprehensive environment summary"""
        return {
            "grid_size": self.grid_size,
            "num_locations": self.num_locations,
            "delivery_locations": [list(map(float, loc)) for loc in self.delivery_locations],
            "current_position": list(map(float, self.current_pos)),
            "delivered_count": len(self.delivered),
            "remaining_count": self.num_locations - len(self.delivered),
            "action_space_size": self.action_space.n if hasattr(self, 'action_space') else "Unknown"
        }

    def step(self, action):
        """Override step to record history"""
        self.action_history.append(action)
        self.state_history.append(self.current_pos.copy())

        next_state, reward, done = super().step(action)

        self.reward_history.append(reward)

        return next_state, reward, done

    def get_training_statistics(self):
        """Get training statistics for AI analysis"""
        if len(self.reward_history) == 0:
            return {}

        stats = {
            "total_episodes": len(self.reward_history),
            "total_reward": sum(self.reward_history),
            "average_reward": np.mean(self.reward_history),
            "std_reward": np.std(self.reward_history),
            "max_reward": max(self.reward_history),
            "min_reward": min(self.reward_history),
            "common_actions": pd.Series(self.action_history).value_counts().to_dict() if self.action_history else {}
        }

        return stats