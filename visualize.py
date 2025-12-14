import matplotlib.pyplot as plt
import numpy as np
import folium

try:
    import osmnx as ox
    import networkx as nx
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False

def plot_osmnx_route(graph, route_nodes, file_path="osmnx_route.png"):
    """
    Plots the delivery route on the OSMnx street graph.

    Args:
        graph: The OSMnx graph object.
        route_nodes (list): An ordered list of graph node IDs representing the route.
        file_path (str): Path to save the image file.
    """
    if not OSMNX_AVAILABLE:
        print("Cannot plot OSMnx route: osmnx is not installed.")
        return

    # Get the full path segments between the route nodes
    full_route = []
    for i in range(len(route_nodes) - 1):
        try:
            path_segment = nx.shortest_path(graph, route_nodes[i], route_nodes[i+1], weight='length')
            full_route.extend(path_segment[1:]) # Add all but the first node to avoid duplicates
        except nx.NetworkXNoPath:
            print(f"Warning: No path found between node {route_nodes[i]} and {route_nodes[i+1]}")
            continue
    
    # Add the first node back
    if route_nodes:
        full_route.insert(0, route_nodes[0])

    if not full_route:
        print("Cannot plot route: The calculated route is empty or invalid.")
        return

    fig, ax = ox.plot_graph_route(
        graph, full_route, show=False, close=False,
        route_color='r', route_linewidth=4, node_size=0,
        filepath=file_path, dpi=300, save=True
    )
    print(f"OSMnx route map saved to {file_path}")
    plt.close(fig)


def plot_learning_curves(reward_histories, title="RL Agent Learning Curves"):
    plt.figure(figsize=(12, 8))
    
    for agent_name, rewards in reward_histories.items():
        window_size = 100
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(moving_avg, label=f'{agent_name} (Moving Avg)')
        else:
            plt.plot(rewards, label=agent_name, alpha=0.6)

    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_curves.png")
    plt.show()

def plot_tuning_impact(initial_histories, optimized_histories):
    """
    Plots the learning curves before and after hyperparameter tuning for each agent.
    """
    agents = list(initial_histories.keys())
    num_agents = len(agents)
    
    fig, axes = plt.subplots(num_agents, 1, figsize=(12, 6 * num_agents))
    if num_agents == 1:
        axes = [axes]
        
    for i, agent_name in enumerate(agents):
        ax = axes[i]
        
        # Plot Initial
        initial_rewards = initial_histories.get(agent_name, [])
        window_size = max(1, len(initial_rewards) // 20)
        if len(initial_rewards) >= window_size:
            initial_avg = np.convolve(initial_rewards, np.ones(window_size)/window_size, mode='valid')
            ax.plot(initial_avg, label='Before Tuning (Moving Avg)', linestyle='--', alpha=0.7)
        else:
            ax.plot(initial_rewards, label='Before Tuning', linestyle='--', alpha=0.7)
            
        # Plot Optimized
        opt_rewards = optimized_histories.get(agent_name, [])
        window_size = max(1, len(opt_rewards) // 20)
        if len(opt_rewards) >= window_size:
            opt_avg = np.convolve(opt_rewards, np.ones(window_size)/window_size, mode='valid')
            ax.plot(opt_avg, label='After Tuning (Moving Avg)', linewidth=2)
        else:
            ax.plot(opt_rewards, label='After Tuning', linewidth=2)
            
        ax.set_title(f"Hyperparameter Tuning Impact: {agent_name}")
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Total Reward")
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout()
    plt.savefig("tuning_impact.png")
    plt.show()

def plot_delivery_route(locations, route, depot_index=0, file_path="delivery_route.html"):
    if locations.size == 0:
        print("Cannot plot route: No locations provided.")
        return

    map_center = np.mean(locations, axis=0)
    m = folium.Map(location=map_center, zoom_start=12)

    folium.Marker(
        location=locations[depot_index],
        popup="Depot",
        icon=folium.Icon(color="red", icon="home"),
    ).add_to(m)

    for i, loc in enumerate(locations):
        if i != depot_index:
            folium.Marker(
                location=loc,
                popup=f"Delivery {i}",
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(m)

    if route:
        route_coords = [locations[i] for i in route]
        folium.PolyLine(route_coords, color="green", weight=2.5, opacity=1).add_to(m)

    m.save(file_path)
    print(f"Route map saved to {file_path}")

def plot_comparison(results, title="Algorithm Performance Comparison"):
    labels = list(results.keys())
    if not labels:
        return
        
    metrics = list(results[labels[0]].keys())
    
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(len(metrics), 1, figsize=(10, 5 * len(metrics)))
    if len(metrics) == 1:
        ax = [ax]

    for i, metric in enumerate(metrics):
        metric_values = [res.get(metric, 0) for res in results.values()]
        ax[i].bar(x, metric_values, width, label=metric)
        ax[i].set_ylabel(metric.replace('_', ' ').title())
        ax[i].set_title(f"Comparison of {metric.replace('_', ' ').title()}")
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(labels)
        ax[i].legend()

    fig.tight_layout()
    plt.savefig("algorithm_comparison.png")
    plt.show()
