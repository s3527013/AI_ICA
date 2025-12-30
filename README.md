# Delivery Route Optimization using AI and Reinforcement Learning

## Overview

This project provides a comprehensive framework for solving the delivery route optimization problem, a variant of the Traveling Salesperson Problem (TSP). It evaluates and compares a variety of algorithms, from classical informed search to modern reinforcement learning, to find the most efficient delivery routes.

The entire pipeline is orchestrated within a single, well-documented script (`delivery_optimisation.py`) that can be run as a standard Python script or interactively in a Jupyter-compatible environment.

---

## Key Features

- **Multiple Algorithm Comparison**: The framework tests and compares a diverse set of algorithms:
  - **Informed Search Agents**:
    - **A* Search**: A powerful algorithm that guarantees the shortest path by balancing the actual distance traveled with a heuristic estimate of the remaining distance.
    - **Greedy Best-First Search**: A faster, heuristic-based approach that always chooses the next closest location.
  - **Reinforcement Learning (RL) Agents**:
    - **Q-Learning**: A classic value-based, off-policy TD algorithm.
    - **SARSA**: An on-policy TD algorithm that considers the current policy for updates.
    - **Deep Q-Network (DQN)**: A deep learning approach that uses a neural network to approximate Q-values, suitable for more complex state spaces.

- **Realistic Environment**:
  - **Real Road Networks**: Utilizes the **OSMnx** library to calculate actual driving distances based on the real-world road network of a specified city (e.g., Middlesbrough).
  - **Graph Caching**: The downloaded road network graph is automatically cached to a `data/` directory, significantly speeding up subsequent runs.

- **AI-Powered Insights**:
  - **Realistic Location Generation**: Leverages the **Google Gemini** model to generate realistic delivery addresses within a city, providing more practical scenarios than purely random coordinates.
  - **Automated Performance Analysis**: After each simulation, the results are sent to the Gemini model to generate a comprehensive, natural-language analysis that compares algorithm performance, discusses trade-offs, and provides business-oriented recommendations.

- **Multi-Scenario Simulation**:
  - The main script is designed to run multiple simulation scenarios with different configurations (e.g., a "standard scale" with 20 parcels and a "large scale" with 50 parcels).
  - This allows for robust analysis of how each algorithm's performance and efficiency scale with the size of the problem.

- **Comprehensive Visualization**:
  - **Interactive Route Maps**: For each agent in every scenario, an interactive HTML map is generated in the `visualisations/` directory. These maps overlay the actual road network path (solid green line) on top of the direct straight-line path (dashed purple line), providing a clear comparison.
  - **Performance Charts**: Automatically generates and saves bar charts comparing all algorithms on key metrics like total route distance and execution time.
  - **Optimization Impact Plots**: For RL agents, the script visualizes the "before and after" of hyperparameter optimization, clearly showing the improvement in learning.

---

## Project Structure

```
.
├── ai_models/            # Contains all algorithm implementations
│   ├── __init__.py
│   ├── a_star.py
│   ├── dqn.py
│   ├── g_bfs.py
│   ├── google_LLM.py
│   ├── informed_search.py
│   ├── q_learning.py
│   └── sarsa.py
├── data/                 # Caches downloaded road network graphs
├── visualisations/       # All output charts and maps are saved here
├── world/                # Environment and data client modules
│   ├── __init__.py
│   ├── environment.py
│   └── osm_client.py
├── .env.example          # Example for environment variables
├── delivery_optimisation.py # Main executable script
└── README.md
```

---

## Setup and Usage

### 1. Prerequisites

- Python 3.8+
- An environment variable `GOOGLE_API_KEY` with a valid API key for Google's Generative AI services.

### 2. Installation

Clone the repository and install the required packages:

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```

Create a `.env` file in the root directory and add your Google API key:

```
GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

### 3. Running the Simulation

The entire simulation can be run from the `delivery_optimisation.py` script. You can run it as a standard Python script or in an interactive environment like a Jupyter Notebook or VS Code's notebook editor.

```bash
python delivery_optimisation.py
```

The script is divided into logical cells using `#%%` separators. You can run each cell sequentially to step through the process:
- **Configuration**: Adjust parameters like the city, number of parcels, and training episodes.
- **Run Scenarios**: Execute the pre-defined simulation scenarios.
- **Multi-Scenario Analysis**: Get a final high-level comparison from the AI model.

All outputs, including detailed performance charts and interactive route maps for each agent, will be saved in the `visualisations/` directory.
