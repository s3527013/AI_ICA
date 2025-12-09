import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(results, title="Algorithm Performance Comparison"):
    """
    Plots a comparison of different RL algorithms based on their evaluation results.
    """
    labels = list(results.keys())
    metrics = list(results[labels[0]].keys())
    
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(len(metrics), 1, figsize=(10, 5 * len(metrics)))
    
    for i, metric in enumerate(metrics):
        metric_values = [res[metric] for res in results.values()]
        ax[i].bar(x, metric_values, width, label=metric)
        ax[i].set_ylabel(metric.replace('_', ' ').title())
        ax[i].set_title(f"Comparison of {metric.replace('_', ' ').title()}")
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(labels)
        ax[i].legend()

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    # This is an example of how to use the plot_comparison function.
    # In the final version, this would be called from train.py with the actual results.
    
    # Example data
    q_learning_results = {'avg_reward': -25.5, 'avg_time': 9.0, 'avg_distance': 25.5}
    sarsa_results = {'avg_reward': -28.2, 'avg_time': 9.0, 'avg_distance': 28.2}
    dqn_results = {'avg_reward': -22.1, 'avg_time': 9.0, 'avg_distance': 22.1}

    all_results = {
        "Q-Learning": q_learning_results,
        "SARSA": sarsa_results,
        "DQN": dqn_results
    }
    
    plot_comparison(all_results)
