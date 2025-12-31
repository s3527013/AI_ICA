"""
This module contains all the visualization functions for the delivery optimization project,
using Matplotlib for charts and Folium for interactive maps.
"""

import os
import folium
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

try:
    import networkx as nx
except ImportError:
    # This allows the module to be imported even if networkx is not installed,
    # though functions requiring it will fail at runtime.
    pass

def plot_delivery_route(env, route: list, file_path: str, agent_name: str = ""):
    """
    Creates an interactive Folium map of a delivery route.

    This function plots the depot and numbered delivery stops on a map. It overlays two
    path types for comparison:
    1. A straight-line (as-the-crow-flies) path in dashed purple.
    2. The actual road network path in solid green, if available.

    Args:
        env (DeliveryEnvironment): The environment instance containing locations and graph data.
        route (list): The sequence of location indices representing the route.
        file_path (str): The path to save the HTML map file.
        agent_name (str, optional): The name of the agent who generated the route, for the map title.
    """
    locations = env.locations
    if not isinstance(locations, np.ndarray) or locations.size == 0:
        print(f"    Cannot plot route for {file_path}: No locations provided.")
        return

    # Center the map on the average coordinates of the locations
    map_center = np.mean(locations, axis=0)
    m = folium.Map(location=map_center, zoom_start=12, tiles="cartodbpositron")

    # Add a title to the map
    title_html = f'<h3 align="center" style="font-size:16px"><b>Route for {agent_name}</b></h3>'
    m.get_root().html.add_child(folium.Element(title_html))

    # Add a marker for the depot (start and end point)
    folium.Marker(
        locations[0],
        popup="Depot",
        tooltip="DEPOT (Start/End)",
        icon=folium.Icon(color="red", icon="warehouse", prefix="fa")
    ).add_to(m)

    # Add numbered markers for each delivery stop in the route
    for i, loc_index in enumerate(route):
        if i == 0 or i == len(route) - 1:  # Skip the depot
            continue
        folium.Marker(
            locations[loc_index],
            popup=f"Stop {i}: Location {loc_index}",
            tooltip=f"Stop #{i}",
            icon=folium.DivIcon(
                html=f'<div style="font-family: sans-serif; background-color: #3388ff; color: white; border-radius: 50%; width: 24px; height: 24px; text-align: center; line-height: 24px; font-weight: bold;">{i}</div>')
        ).add_to(m)

    # Layer 1: Draw the straight-line path
    route_coords = [locations[i] for i in route]
    folium.PolyLine(
        route_coords,
        color="purple",
        weight=2,
        opacity=0.8,
        dash_array='5, 10',
        tooltip="Straight-line path"
    ).add_to(m)

    # Layer 2: Draw the actual road network path if available
    if env.distance_metric == 'network' and env.osmnx_client and hasattr(env.osmnx_client, 'G'):
        G = env.osmnx_client.G
        for i in range(len(route) - 1):
            try:
                start_node = env.nodes[route[i]]
                end_node = env.nodes[route[i+1]]
                path_nodes = nx.shortest_path(G, start_node, end_node, weight='length')
                path_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in path_nodes]
                folium.PolyLine(path_coords, color="green", weight=4, opacity=0.7, tooltip="Road network path").add_to(m)
            except (nx.NetworkXNoPath, KeyError):
                # If a path is not found between two nodes, skip that segment
                continue

    # Save the map to an HTML file and display it in the notebook
    m.save(file_path)
    print(f"    ✓ Interactive map saved to {file_path}")
    display(m)

def plot_performance_comparison(results: dict, output_dir: str, scenario_name: str):
    """
    Generates and saves bar charts comparing algorithm performance on distance and time.

    Args:
        results (dict): A dictionary containing performance data for each agent.
        output_dir (str): The directory where the output chart image will be saved.
        scenario_name (str): The name of the simulation scenario, used in the title and filename.
    """
    labels = list(results.keys())
    distances = [res['total_distance_km'] for res in results.values()]
    durations = [res['duration_sec'] for res in results.values()]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    fig.suptitle(f'Performance Comparison: {scenario_name}', fontsize=16, fontweight='bold')

    # Subplot 1: Route Distance
    ax1.bar(labels, distances, color=plt.cm.plasma(np.linspace(0.4, 0.8, len(labels))), edgecolor='black')
    ax1.set_ylabel('Total Distance (km)')
    ax1.set_title('Route Distance', fontsize=14)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
    for i, v in enumerate(distances):
        ax1.text(i, v + 0.5, f"{v:.2f}", ha='center', va='bottom')

    # Subplot 2: Execution Time
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

def plot_optimization_impact(initial_rewards: dict, final_rewards: dict, output_dir: str, scenario_name: str):
    """
    Generates and saves a plot comparing RL agent learning curves before and after optimization.

    Args:
        initial_rewards (dict): A dictionary of reward histories from the tuning phase.
        final_rewards (dict): A dictionary of reward histories from the final, optimized run.
        output_dir (str): The directory to save the output chart.
        scenario_name (str): The name of the simulation scenario.
    """
    rl_agent_names = list(initial_rewards.keys())
    if not rl_agent_names:
        return

    fig, axes = plt.subplots(len(rl_agent_names), 1, figsize=(12, 6 * len(rl_agent_names)), squeeze=False)
    fig.suptitle(f'RL Agent Optimization Impact: {scenario_name}', fontsize=16, fontweight='bold')

    for i, name in enumerate(rl_agent_names):
        ax = axes[i, 0]
        # Use a moving average to smooth the learning curves
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

def plot_multi_scenario_comparison(all_results: dict, output_dir: str):
    """
    Plots a grouped bar chart comparing algorithm performance across multiple scenarios.

    Args:
        all_results (dict): A nested dictionary containing results from all scenarios.
        output_dir (str): The directory to save the output chart.
    """
    scenario_names = list(all_results.keys())
    if not scenario_names:
        return
    
    # Collect all unique agent names from all scenarios
    agent_names = sorted(list(set(agent for res in all_results.values() for agent in res.keys())))
    
    # Prepare data for grouped bar chart
    distance_data = {agent: [all_results[sc].get(agent, {}).get('total_distance_km', 0) for sc in scenario_names] for agent in agent_names}
    duration_data = {agent: [all_results[sc].get(agent, {}).get('duration_sec', 0) for sc in scenario_names] for agent in agent_names}
    
    x = np.arange(len(scenario_names))  # the label locations
    width = 0.15  # the width of the bars
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14))
    fig.suptitle('Multi-Scenario Performance Comparison', fontsize=16, fontweight='bold')

    # Plot for distance
    for i, agent in enumerate(agent_names):
        offset = width * (i - len(agent_names) / 2)
        rects = ax1.bar(x + offset, distance_data[agent], width, label=agent)
        ax1.bar_label(rects, padding=3, fmt='%.1f')
        
    ax1.set_ylabel('Total Distance (km)')
    ax1.set_title('Route Distance Across Scenarios')
    ax1.set_xticks(x, scenario_names)
    ax1.legend()
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Plot for duration
    for i, agent in enumerate(agent_names):
        offset = width * (i - len(agent_names) / 2)
        rects = ax2.bar(x + offset, duration_data[agent], width, label=agent)
        ax2.bar_label(rects, padding=3, fmt='%.1f')

    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_title('Execution Time Across Scenarios')
    ax2.set_xticks(x, scenario_names)
    ax2.legend()
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, "multi_scenario_comparison.png")
    plt.savefig(save_path, dpi=150)
    print(f"    ✓ Multi-scenario comparison chart saved to {save_path}")
    plt.show()
