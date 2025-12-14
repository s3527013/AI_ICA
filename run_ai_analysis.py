import numpy as np
from IPython.display import display, Markdown

from models.dqn import DQNAgent
from environment import DeliveryEnvironment
from main_with_google_ai import GoogleAIModelExplainer
from models.q_learning import QLearningAgent
from models.sarsa import SarsaAgent
from visualize import plot_learning_curves, plot_comparison, plot_osmnx_route, plot_delivery_route


def train_agent(agent, env, num_episodes, is_dqn=False):
    """A unified training function for all agent types."""
    reward_history = []
    for episode in range(num_episodes):
        state = env.reset(vectorized=is_dqn)
        done = False
        total_reward = 0
        
        action = agent.choose_action(state, env.get_possible_actions())

        while not done:
            if action is None: break
            
            next_state_tuple, reward, done = env.step(action)
            
            if is_dqn:
                next_state = env._get_state(vectorized=True)
                agent.add_experience(state, action, reward, next_state, done)
                agent.update_model()
            else:
                next_state = next_state_tuple
                next_possible_actions = env.get_possible_actions()
                if isinstance(agent, SarsaAgent):
                    next_action = agent.choose_action(next_state, next_possible_actions)
                    agent.update_q_table(state, action, reward, next_state, next_action)
                    action = next_action
                else: # Q-Learning
                    agent.update_q_table(state, action, reward, next_state, next_possible_actions)
            
            if not isinstance(agent, SarsaAgent):
                action = agent.choose_action(next_state, env.get_possible_actions())

            state = next_state
            total_reward += reward
        
        agent.decay_epsilon()
        reward_history.append(total_reward)
    return reward_history

def evaluate_agent(agent, env, is_dqn=False):
    state = env.reset(vectorized=is_dqn)
    route = [env.start_pos_index]
    total_distance = 0
    
    while len(route) <= env.num_locations:
        possible_actions = env.get_possible_actions()
        if not possible_actions: break
        
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0
        action = agent.choose_action(state, possible_actions)
        agent.epsilon = original_epsilon
        
        if action is None: break
        
        distance = env.distance_matrix[env.current_pos_index][action]
        total_distance += distance
        route.append(action)
        
        state, _, _ = env.step(action)
        if is_dqn:
            state = env._get_state(vectorized=True)

    if route[-1] != env.start_pos_index:
        total_distance += env.distance_matrix[route[-1]][env.start_pos_index]
        route.append(env.start_pos_index)
        
    return route, total_distance / 1000

def main():
    # --- CONFIGURATION ---
    CITY = "Middlesbrough"
    NUM_PARCELS = 200
    DISTANCE_METRIC = 'network'
    NUM_EPISODES = 2000
    
    # --- 1. INITIALIZE AI ---
    explainer = GoogleAIModelExplainer()
    if not explainer.available:
        print("Google AI not available. Exiting.")
        return

    # --- 2. GENERATE LOCATIONS WITH AI ---
    print(f"Using Gemini to generate {NUM_PARCELS + 1} locations in {CITY}...")
    locations_list = explainer.generate_locations_for_city(CITY, NUM_PARCELS + 1)
    print(locations_list)
    
    env = DeliveryEnvironment(addresses=locations_list, city_name=CITY, distance_metric=DISTANCE_METRIC)
    
    agents = {
        "Q-Learning": QLearningAgent(action_space=list(range(env.num_locations))),
        "SARSA": SarsaAgent(action_space=list(range(env.num_locations))),
        "DQN": DQNAgent(state_size=env.get_state_size(), action_size=env.num_locations)
    }

    # --- 3. INITIAL TRAINING & AI-TUNING ---
    tuned_params = {}
    print("\\n--- Initial Training & Hyperparameter Tuning ---")
    for name, agent in agents.items():
        print(f"Training {name} for tuning...")
        reward_history = train_agent(agent, env, 500, isinstance(agent, DQNAgent)) # Short run for tuning
        
        print(f"Asking Gemini for hyperparameter recommendations for {name}...")
        recommendations = explainer.provide_hyperparameter_recommendations(name, reward_history)
        
        if isinstance(recommendations, dict):
            tuned_params[name] = recommendations
            print(f"  > AI recommends: {recommendations}")
        else:
            print(f"  > Could not get AI recommendations for {name}.")
            tuned_params[name] = {} # Use defaults

    # --- 4. OPTIMIZED TRAINING ---
    print("\\n--- Optimized Training Run ---")
    final_results = {}
    reward_histories = {}
    best_route_info = {"agent": None, "route": [], "distance": float('inf')}

    for name, agent in agents.items():
        # Apply tuned parameters
        if name in tuned_params:
            for param, value in tuned_params[name].items():
                if hasattr(agent, param):
                    setattr(agent, param, value)
        
        print(f"Training {name} with optimized parameters...")
        reward_histories[name] = train_agent(agent, env, NUM_EPISODES, isinstance(agent, DQNAgent))
        
        route, distance = evaluate_agent(agent, env, isinstance(agent, DQNAgent))
        final_results[name] = {"total_distance_km": distance, "avg_reward": np.mean(reward_histories[name][-100:])}
        
        if distance < best_route_info["distance"]:
            best_route_info = {"agent": name, "route": route, "distance": distance}

    # --- 5. FINAL AI ANALYSIS ---
    print("\\n--- Final AI-Powered Analysis ---")
    env_config = env.get_environment_summary()
    env_config["episodes"] = NUM_EPISODES
    
    analysis = explainer.analyze_performance(final_results, env_config)
    display(Markdown(analysis))

    # --- 6. VISUALIZATION ---
    print("\\n--- Generating Visualizations ---")
    plot_learning_curves(reward_histories)
    plot_comparison(final_results)
    
    if env.osmnx_client:
        best_route_nodes = [env.nodes[i] for i in best_route_info["route"]]
        plot_osmnx_route(env.osmnx_client.G, best_route_nodes, file_path=f"best_route_{CITY.lower()}_final.png")
    else:
        plot_delivery_route(locations=env.locations, route=best_route_info["route"], file_path="best_route_final.html")

if __name__ == "__main__":
    main()
