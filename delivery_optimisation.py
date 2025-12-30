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
import folium
import matplotlib.pyplot as plt
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

from ai_models import *
from world import *

if not OSMNX_AVAILABLE:
    print("\nWARNING: OSMnx is not available. The 'network' distance metric will fail.")
    print("Please install with: pip install osmnx")

# %% [markdown]
# ### 2. Visualization and Helper Functions
# This cell defines all the functions used for plotting charts and maps.
# %%
def plot_delivery_route(env, route, file_path, agent_name=""):
    """
    Creates an interactive Folium map, saves it to a file, and displays it in the notebook.
    """
    locations = env.locations
    if not isinstance(locations, np.ndarray) or locations.size == 0:
        print(f"    Cannot plot route for {file_path}: No locations provided.")
        return

    map_center = np.mean(locations, axis=0)
    m = folium.Map(location=map_center, zoom_start=12, tiles="cartodbpositron")

    title_html = f'<h3 align="center" style="font-size:16px"><b>Route for {agent_name}</b></h3>'
    m.get_root().html.add_child(folium.Element(title_html))

    folium.Marker(locations[0], popup="Depot", tooltip="DEPOT (Start/End)", icon=folium.Icon(color="red", icon="warehouse", prefix="fa")).add_to(m)

    for i, loc_index in enumerate(route):
        if i == 0 or i == len(route) - 1:
            continue
        folium.Marker(locations[loc_index], popup=f"Stop {i}: Location {loc_index}", tooltip=f"Stop #{i}", icon=folium.DivIcon(html=f'<div style="font-family: sans-serif; background-color: #3388ff; color: white; border-radius: 50%; width: 24px; height: 24px; text-align: center; line-height: 24px; font-weight: bold;">{i}</div>')).add_to(m)

    route_coords = [locations[i] for i in route]
    folium.PolyLine(route_coords, color="purple", weight=2, opacity=0.8, dash_array='5, 10', tooltip="Straight-line path").add_to(m)

    if env.distance_metric == 'network' and env.osmnx_client and hasattr(env.osmnx_client, 'G'):
        G = env.osmnx_client.G
        for i in range(len(route) - 1):
            try:
                path_nodes = nx.shortest_path(G, env.nodes[route[i]], env.nodes[route[i+1]], weight='length')
                path_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in path_nodes]
                folium.PolyLine(path_coords, color="green", weight=4, opacity=0.7, tooltip="Road network path").add_to(m)
            except (nx.NetworkXNoPath, KeyError):
                continue

    m.save(file_path)
    print(f"    ✓ Interactive map saved to {file_path}")
    display(m)

def plot_performance_comparison(results, output_dir, scenario_name):
    """Plots bar charts, saves them to a file, and displays them in the notebook."""
    labels = list(results.keys())
    distances = [res['total_distance_km'] for res in results.values()]
    durations = [res['duration_sec'] for res in results.values()]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    fig.suptitle(f'Performance Comparison: {scenario_name}', fontsize=16, fontweight='bold')

    ax1.bar(labels, distances, color=plt.cm.plasma(np.linspace(0.4, 0.8, len(labels))), edgecolor='black')
    ax1.set_ylabel('Total Distance (km)')
    ax1.set_title('Route Distance', fontsize=14)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
    for i, v in enumerate(distances):
        ax1.text(i, v + 0.5, f"{v:.2f}", ha='center', va='bottom')

    ax2.bar(labels, durations, color=plt.cm.viridis(np.linspace(0.4, 0.8, len(labels))), edgecolor='black')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_title('Execution Time', fontsize=14)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
    for i, v in enumerate(durations):
        ax2.text(i, v + 0.1, f"{v:.2f}s", ha='center', va='bottom')

    plt.setp(ax1.get_xticklabels(), rotation=20, ha="right")
    plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(output_dir, f"performance_comparison_{scenario_name}.png")
    plt.savefig(save_path, dpi=150)
    print(f"    ✓ Performance comparison chart saved to {save_path}")
    plt.show()

def plot_optimization_impact(initial_rewards, final_rewards, output_dir, scenario_name):
    """Plots optimization impact, saves it to a file, and displays it in the notebook."""
    rl_agent_names = list(initial_rewards.keys())
    if not rl_agent_names:
        return

    fig, axes = plt.subplots(len(rl_agent_names), 1, figsize=(12, 6 * len(rl_agent_names)), squeeze=False)
    fig.suptitle(f'RL Agent Optimization Impact: {scenario_name}', fontsize=16, fontweight='bold')

    for i, name in enumerate(rl_agent_names):
        ax = axes[i, 0]
        ax.plot(np.convolve(initial_rewards[name], np.ones(100)/100, mode='valid'), label='Before Optimization', color='orange', linestyle='--')
        ax.plot(np.convolve(final_rewards[name], np.ones(100)/100, mode='valid'), label='After Optimization', color='green')
        ax.set_title(f'Optimization Impact for {name}', fontsize=14)
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Moving Average Reward')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, f"rl_optimization_impact_{scenario_name}.png")
    plt.savefig(save_path, dpi=150)
    print(f"    ✓ Optimization impact chart saved to {save_path}")
    plt.show()

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
# ### 4. Run Scenarios
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
# ### 5. Final Multi-Scenario Analysis
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
