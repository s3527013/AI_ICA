# %% [markdown]
# # Delivery Route Optimization using RL and Informed Search
#
# ## Overview
# This notebook evaluates and compares a variety of algorithms for solving the delivery route optimization problem. It uses a unified pipeline to test:
# 1.  **Reinforcement Learning Agents**: Q-Learning, SARSA, and DQN.
# 2.  **Informed Search Agents**: A* Search and Greedy Best-First Search.
#
# The script uses `osmnx` to calculate real road network distances and leverages a Google AI model to provide a final analysis and explanation of the results.

# %% [markdown]
# ### 1. Imports and Setup
# This cell imports all necessary libraries and modules, and sets up the environment.
# %%
import os
import time
import numpy as np
from IPython.display import display, Markdown
from dotenv import load_dotenv

print("All libraries imported successfully!")
try:
    import osmnx as ox
    import networkx as nx
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False

from algorithms import *
from world import *
from utils import *

if not OSMNX_AVAILABLE:
    print("\nWARNING: OSMnx is not available. The 'network' distance metric will fail.")
    print("Please install with: pip install osmnx")

# %% [markdown]
# ### 2. Helper Functions
# This cell defines the helper functions for training and evaluating agents.
# %%
def generate_random_locations(city_name, num_locations):
    osm = OSMClient()
    bbox = osm.get_bounding_box(city_name)
    if not bbox: return None
    lats = np.random.uniform(bbox[0], bbox[1], num_locations)
    lons = np.random.uniform(bbox[2], bbox[3], num_locations)
    return np.vstack((lats, lons)).T

def train_agent(agent, env, num_episodes, is_dqn=False):
    reward_history = []
    for _ in range(num_episodes):
        state = env.reset(vectorized=is_dqn)
        done = False
        total_reward = 0
        while not done:
            actions = env.get_possible_actions()
            if not actions: break
            action = agent.choose_action(state, actions)
            if action is None: break
            next_state_tuple, reward, done = env.step(action)
            if is_dqn:
                next_state = env._get_state(vectorized=True)
                agent.add_experience(state, action, reward, next_state, done)
                agent.update_model()
                state = next_state
            else:
                next_actions = env.get_possible_actions()
                if isinstance(agent, SarsaAgent):
                    next_action = agent.choose_action(next_state_tuple, next_actions)
                    agent.update_q_table(state, action, reward, next_state_tuple, next_action)
                    state, action = next_state_tuple, next_action
                else:
                    agent.update_q_table(state, action, reward, next_state_tuple, next_actions)
                    state = next_state_tuple
            total_reward += reward
        agent.decay_epsilon()
        reward_history.append(total_reward)
    return reward_history

def evaluate_agent(agent, env, is_dqn=False):
    state = env.reset(vectorized=is_dqn)
    route = [env.start_pos_index]
    agent.epsilon = 0.0
    while len(route) <= env.num_locations:
        actions = env.get_possible_actions()
        if not actions: break
        action = agent.choose_action(state, actions)
        if action is None or action in route: break
        route.append(action)
        state, _, done = env.step(action)
        if is_dqn: state = env._get_state(vectorized=True)
        if done: break
    if route[-1] != env.start_pos_index:
        route.append(env.start_pos_index)
    return route, sum(env.distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1)) / 1000

# %% [markdown]
# ### 3. Main Simulation Function
# This function encapsulates the entire process for a single simulation run.
# %%
def run_simulation(scenario_name, city, num_parcels, distance_metric, tune_episodes, final_episodes, output_dir, include_astar=True):
    """
    Runs a full simulation scenario from environment setup to final analysis.
    """
    print("\n" + "#"*80)
    print(f"# Running Scenario: {scenario_name}")
    print("#"*80)
    
    # --- Environment Setup ---
    if not OSMNX_AVAILABLE:
        exit("OSMnx is required for 'network' distance metric. Please install it and try again.")

    explainer = GoogleAIModelExplainer()
    env = None

    if explainer.available:
        print("\nAttempting to generate realistic locations using Google AI...")
        ai_addresses = explainer.generate_locations_for_city(city, num_parcels + 1)
        if ai_addresses:
            print("  ✓ Successfully generated addresses from AI.")
            env = DeliveryEnvironment(addresses=ai_addresses, city_name=city, distance_metric=distance_metric)
        else:
            print("  ✗ AI failed to return valid addresses. Falling back to random locations.")

    if env is None:
        print("  Generating random locations as a fallback...")
        locations_coords = generate_random_locations(city, num_parcels + 1)
        if locations_coords is None:
            exit("Failed to generate random locations. Exiting.")
        env = DeliveryEnvironment(locations=locations_coords, city_name=city, distance_metric=distance_metric)

    print(f"\n  ✓ Environment ready. Matrix shape: {env.distance_matrix.shape}")

    # --- Agent Initialization ---
    rl_agents = {
        "Q-Learning": QLearningAgent(action_space=list(range(env.num_locations)), alpha=0.1, gamma=0.9, epsilon=1.0),
        "SARSA": SarsaAgent(action_space=list(range(env.num_locations)), alpha=0.1, gamma=0.9, epsilon=1.0),
        "DQN": DQNAgent(state_size=env.get_state_size(), action_size=env.num_locations, learning_rate=0.001, epsilon=1.0),
    }
    informed_search_agents = {"Greedy_Best-First": GreedyBestFirstSearchAgent()}
    if include_astar:
        informed_search_agents["A-Star_Search"] = AStarAgent()
    
    all_agents = {**rl_agents, **informed_search_agents}
    print("\nAgents to be tested:", ", ".join(all_agents.keys()))

    # --- RL Agent Tuning ---
    print("\n" + "=" * 50)
    print("Phase 1: RL Agent Tuning")
    initial_reward_histories = {}
    for name, agent in rl_agents.items():
        print(f"  Tuning {name} for {tune_episodes} episodes...")
        initial_reward_histories[name] = train_agent(agent, env, tune_episodes, isinstance(agent, DQNAgent))
    print("Tuning phase complete.")

    # --- Final Run ---
    print("\n" + "=" * 50)
    print("Phase 2: Optimization and Final Run")
    final_results = {}
    best_route_info = {"agent": None, "route": [], "distance": float('inf')}
    final_reward_histories = {}

    optimized_params = {"Q-Learning": {'alpha': 0.5, 'gamma': 0.95}, "SARSA": {'alpha': 0.2, 'gamma': 0.98}, "DQN": {'learning_rate': 0.0005}}

    for name, agent in all_agents.items():
        print(f"\n--- Processing Agent: {name} ---")
        start_time = time.time()
        if isinstance(agent, InformedSearchAgent):
            route, distance = agent.solve(env)
        else:
            print(f"  Applying optimized parameters and running for {final_episodes} episodes...")
            params = optimized_params.get(name, {})
            agent_class = agent.__class__
            if name == "DQN":
                final_agent = agent_class(state_size=env.get_state_size(), action_size=env.num_locations, **params)
            else:
                final_agent = agent_class(action_space=list(range(env.num_locations)), **params)
            final_reward_histories[name] = train_agent(final_agent, env, final_episodes, isinstance(final_agent, DQNAgent))
            print("  Evaluating final policy...")
            route, distance = evaluate_agent(final_agent, env, isinstance(final_agent, DQNAgent))
        
        duration = time.time() - start_time
        final_results[name] = {"total_distance_km": distance, "route_length": len(route), "duration_sec": duration, "route": route}
        print(f"  ✓ Finished in {duration:.2f}s. Route Distance: {distance:.2f} km")

        agent_map_filename = os.path.join(output_dir, f"route_{scenario_name}_{name.replace(' ', '_')}.html")
        plot_delivery_route(env, route, agent_map_filename, agent_name=f"{name} ({scenario_name})")

        if distance < best_route_info["distance"]:
            best_route_info = {"agent": name, "route": route, "distance": distance}
            print(f"  >>> New best route found by {name}! <<<")

    # --- Visualization and Analysis ---
    print("\n" + "=" * 50)
    print(f"VISUALIZING AND EXPORTING RESULTS for {scenario_name}")
    plot_optimization_impact(initial_reward_histories, final_reward_histories, output_dir, scenario_name)
    plot_performance_comparison(final_results, output_dir, scenario_name)
    
    best_map_filename = os.path.join(output_dir, f"best_route_{scenario_name}.html")
    plot_delivery_route(env, best_route_info["route"], best_map_filename, agent_name=f"Best Route: {best_route_info['agent']} ({scenario_name})")

    if explainer.available:
        print("\nRequesting AI-Powered Analysis from Google...")
        env_config = env.get_environment_summary()
        env_config.update({"scenario": scenario_name, "rl_tuning_episodes": tune_episodes, "rl_final_episodes": final_episodes, "best_agent": best_route_info["agent"], "best_distance_km": best_route_info["distance"]})
        analysis = explainer.analyze_performance(final_results, env_config)
        display(Markdown(analysis))
    else:
        print("\nGoogle AI Explainer not available. Skipping analysis.")
    # Return results for multi-scenario analysis
    return final_results

# %% [markdown]
# ### 4. Run Scenario 1
# This is the main execution block. It will run all defined simulation scenarios.
# %%
all_scenario_results = {}
load_dotenv()

# Scenario 1
scenario_1_results = run_simulation(
    scenario_name="standard_scale",
    city="Middlesbrough",
    num_parcels=20,
    distance_metric='network',
    tune_episodes=500,
    final_episodes=3000,
    output_dir="visualisations"
)
if scenario_1_results:
    all_scenario_results["standard_scale"] = scenario_1_results

print("\nAll simulations finished.")

# %% [markdown]
# ### 5. Run Scenario 2
# This will run the second scenario.
# %%
# Scenario 2
scenario_2_results = run_simulation(
    scenario_name="large_scale",
    city="Middlesbrough",
    num_parcels=50,
    distance_metric='network',
    tune_episodes=1000,
    final_episodes=5000,
    output_dir="visualisations",
    include_astar=False
)
if scenario_2_results:
    all_scenario_results["large_scale"] = scenario_2_results

print("\nAll simulations finished.")

# %% [markdown]
# ### 6. Final Multi-Scenario Analysis
# This final step provides a high-level comparison of how the algorithms performed across the different scenarios, focusing on scalability and overall performance.
# %%
explainer = GoogleAIModelExplainer()
if explainer.available and all_scenario_results:
    print("\n" + "=" * 80)
    print("Requesting Final Multi-Scenario Analysis from Google")
    print("=" * 80)

    multi_scenario_analysis = explainer.analyze_multiple_scenarios(all_scenario_results)
    display(Markdown(multi_scenario_analysis))
else:
    print("\nCould not generate multi-scenario analysis. (AI unavailable or no scenarios were run).")

# %% [markdown]
# ### 7. Cross-Scenario Performance Visualization
# This chart provides a direct visual comparison of algorithm performance across the different scenarios that were run.
# %%
if all_scenario_results:
    plot_multi_scenario_comparison(all_scenario_results, "visualisations")
