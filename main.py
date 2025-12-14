import numpy as np
import time
from environment import DeliveryEnvironment
from visualize import plot_delivery_route, plot_learning_curves, plot_comparison, plot_osmnx_route
from q_learning import QLearningAgent
from sarsa import SarsaAgent
from dqn import DQNAgent
from osm_client import OSMClient

def generate_random_locations(city_name, num_locations):
    osm = OSMClient()
    bbox = osm.get_bounding_box(city_name)
    if not bbox:
        print(f"Could not find bounding box for {city_name}. Exiting.")
        return None
    
    min_lat, max_lat, min_lon, max_lon = bbox
    lats = np.random.uniform(min_lat, max_lat, num_locations)
    lons = np.random.uniform(min_lon, max_lon, num_locations)
    
    locations = np.vstack((lats, lons)).T
    print(f"Generated {num_locations} random locations in {city_name}.")
    return locations

def train_tabular_agent(agent, env, num_episodes):
    reward_history = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        action = agent.choose_action(state, env.get_possible_actions())

        while not done:
            if action is None: break

            next_state, reward, done = env.step(action)
            next_possible_actions = env.get_possible_actions()
            
            if isinstance(agent, SarsaAgent):
                next_action = agent.choose_action(next_state, next_possible_actions)
                agent.update_q_table(state, action, reward, next_state, next_action)
                action = next_action
            else: # Q-Learning
                agent.update_q_table(state, action, reward, next_state, next_possible_actions)
                action = agent.choose_action(next_state, next_possible_actions)

            state = next_state
            total_reward += reward
        
        agent.decay_epsilon()
        reward_history.append(total_reward)
        if (episode + 1) % 500 == 0:
            print(f"  Episode {episode + 1}/{num_episodes}, Reward: {total_reward:.2f}")
            
    return reward_history

def train_dqn_agent(agent, env, num_episodes):
    reward_history = []
    for episode in range(num_episodes):
        state = env.reset(vectorized=True)
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state, env.get_possible_actions())
            if action is None: break

            _, reward, done = env.step(action)
            next_state_vec = env._get_state(vectorized=True)
            
            agent.add_experience(state, action, reward, next_state_vec, done)
            agent.update_model()
            
            state = next_state_vec
            total_reward += reward
        
        agent.decay_epsilon()
        reward_history.append(total_reward)
        if (episode + 1) % 500 == 0:
            print(f"  Episode {episode + 1}/{num_episodes}, Reward: {total_reward:.2f}")
            
    return reward_history

def evaluate_agent(agent, env):
    state = env.reset()
    is_dqn = isinstance(agent, DQNAgent)
    if is_dqn:
        state = env._get_state(vectorized=True)
        
    route_indices = [env.start_pos_index]
    total_distance = 0
    
    while len(route_indices) <= env.num_locations:
        possible_actions = env.get_possible_actions()
        if not possible_actions: break
        
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0
        action = agent.choose_action(state, possible_actions)
        agent.epsilon = original_epsilon
        
        if action is None: break
        
        distance = env.distance_matrix[env.current_pos_index][action]
        total_distance += distance
        
        route_indices.append(action)
        
        if is_dqn:
            _, _, _ = env.step(action)
            state = env._get_state(vectorized=True)
        else:
            state, _, _ = env.step(action)

    if route_indices[-1] != env.start_pos_index:
        total_distance += env.distance_matrix[route_indices[-1]][env.start_pos_index]
        route_indices.append(env.start_pos_index)
        
    return route_indices, total_distance / 1000

def main():
    # --- CONFIGURATION ---
    city = "Middlesbrough"
    num_parcels = 20
    distance_metric = 'network' # Use 'network' for osmnx
    num_episodes = 5000
    # ---------------------

    locations = generate_random_locations(city, num_parcels + 1)
    if locations is None: return

    env = DeliveryEnvironment(locations=locations, city_name=city, distance_metric=distance_metric)
    if env.num_locations == 0: return

    agents = {
        "Q-Learning": QLearningAgent(action_space=list(range(env.num_locations))),
        "SARSA": SarsaAgent(action_space=list(range(env.num_locations))),
        "DQN": DQNAgent(state_size=env.get_state_size(), action_size=env.num_locations)
    }

    reward_histories = {}
    final_results = {}
    best_route_info = {"agent": None, "route": [], "distance": np.inf}

    for name, agent in agents.items():
        print(f"\n--- Training {name} ---")
        start_time = time.time()
        
        if isinstance(agent, DQNAgent):
            reward_histories[name] = train_dqn_agent(agent, env, num_episodes)
        else:
            reward_histories[name] = train_tabular_agent(agent, env, num_episodes)
            
        print(f"Training finished in {time.time() - start_time:.2f} seconds.")
        
        route, distance = evaluate_agent(agent, env)
        final_results[name] = {"total_distance_km": distance}
        print(f"Evaluation - Route Distance: {distance:.2f} km")
        
        if distance < best_route_info["distance"]:
            best_route_info = {"agent": name, "route": route, "distance": distance}

    print("\n--- Results ---")
    print(f"Best route found by: {best_route_info['agent']} ({best_route_info['distance']:.2f} km)")
    
    plot_learning_curves(reward_histories)
    plot_comparison(final_results)
    
    # Use OSMnx for visualization if the metric was 'network'
    if env.osmnx_client:
        print("Generating OSMnx route plot...")
        # Convert route indices to graph node IDs
        best_route_nodes = [env.nodes[i] for i in best_route_info["route"]]
        plot_osmnx_route(
            env.osmnx_client.G, 
            best_route_nodes, 
            file_path=f"best_route_{city.lower()}_network.png"
        )
    else: # Fallback to Folium
        file_path = f"best_route_{city.lower()}.html"
        plot_delivery_route(
            locations=env.locations,
            route=best_route_info["route"],
            depot_index=env.start_pos_index,
            file_path=file_path
        )

if __name__ == "__main__":
    main()
