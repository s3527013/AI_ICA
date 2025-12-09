from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


class GoogleAIDashboard:
    """
    Streamlit dashboard with Google AI integration for RL training visualization.
    """

    def __init__(self, google_ai_explainer):
        self.explainer = google_ai_explainer
        self.setup_page()

    def setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="RL Delivery Optimization Dashboard",
            page_icon="🚚",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        st.title("🚚 Reinforcement Learning Delivery Route Optimization")
        st.markdown("---")

    def display_performance_comparison(self, results: Dict[str, Dict]):
        """Display algorithm performance comparison"""
        st.header("📊 Algorithm Performance Comparison")

        # Convert results to DataFrame for visualization
        df_data = []
        for algo, metrics in results.items():
            for metric, value in metrics.items():
                df_data.append({
                    "Algorithm": algo,
                    "Metric": metric.replace("_", " ").title(),
                    "Value": float(value)
                })

        if df_data:
            df = pd.DataFrame(df_data)

            # Create metrics comparison
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Best Algorithm",
                          max(results.items(), key=lambda x: x[1].get('avg_reward', 0))[0])

            with col2:
                best_reward = max([m.get('avg_reward', 0) for m in results.values()])
                st.metric("Best Average Reward", f"{best_reward:.2f}")

            with col3:
                # Calculate efficiency (reward per time)
                efficiencies = []
                for algo, metrics in results.items():
                    if metrics.get('avg_time', 1) > 0:
                        eff = metrics.get('avg_reward', 0) / metrics.get('avg_time', 1)
                        efficiencies.append((algo, eff))

                if efficiencies:
                    best_eff_algo = max(efficiencies, key=lambda x: x[1])[0]
                    st.metric("Most Efficient", best_eff_algo)

            # Visualization
            tab1, tab2, tab3 = st.tabs(["Bar Chart", "Radar Chart", "Raw Data"])

            with tab1:
                fig = px.bar(df, x="Algorithm", y="Value", color="Metric",
                             barmode="group", title="Algorithm Performance Metrics")
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                # Create radar chart
                self._create_radar_chart(results)

            with tab3:
                st.dataframe(df.pivot_table(index="Algorithm", columns="Metric", values="Value"))

    def _create_radar_chart(self, results):
        """Create radar chart for algorithm comparison"""
        metrics = ['avg_reward', 'avg_time', 'avg_distance']
        metrics_display = ['Average Reward', 'Average Time', 'Average Distance']

        fig = go.Figure()

        for algo, algo_results in results.items():
            values = []
            for metric in metrics:
                value = algo_results.get(metric, 0)
                # Normalize for radar chart (invert time and distance)
                if metric == 'avg_time':
                    value = 1 / (value + 1)  # Invert time (lower is better)
                elif metric == 'avg_distance':
                    value = 1 / (value + 1)  # Invert distance (lower is better)
                values.append(value)

            # Close the radar
            values.append(values[0])

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics_display + [metrics_display[0]],
                name=algo,
                fill='toself'
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Algorithm Comparison Radar Chart"
        )

        st.plotly_chart(fig, use_container_width=True)

    def display_training_progress(self, training_history: Dict[str, List[float]]):
        """Display training progress visualization"""
        st.header("📈 Training Progress")

        # Create smoothed training curves
        fig = go.Figure()

        for algo, rewards in training_history.items():
            # Smooth the rewards
            window_size = min(50, len(rewards) // 10)
            if window_size > 1:
                smoothed = pd.Series(rewards).rolling(window=window_size, center=True).mean()
            else:
                smoothed = rewards

            fig.add_trace(go.Scatter(
                x=list(range(len(smoothed))),
                y=smoothed,
                mode='lines',
                name=algo,
                opacity=0.8
            ))

        fig.update_layout(
            title='Training Progress (Smoothed Rewards)',
            xaxis_title='Episode',
            yaxis_title='Reward',
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Training statistics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Training Statistics")
            stats_data = []
            for algo, rewards in training_history.items():
                if len(rewards) > 0:
                    stats_data.append({
                        "Algorithm": algo,
                        "Final Reward": rewards[-1],
                        "Average": np.mean(rewards),
                        "Max": np.max(rewards),
                        "Std": np.std(rewards)
                    })

            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df.style.format({
                    "Final Reward": "{:.2f}",
                    "Average": "{:.2f}",
                    "Max": "{:.2f}",
                    "Std": "{:.2f}"
                }))

        with col2:
            st.subheader("Convergence Analysis")
            for algo, rewards in training_history.items():
                if len(rewards) > 100:
                    # Estimate convergence
                    window = 50
                    for i in range(window, len(rewards) - window):
                        early = np.mean(rewards[i - window:i])
                        late = np.mean(rewards[i:i + window])
                        if abs(late - early) < 0.1 * abs(early):
                            st.write(f"{algo}: Converged around episode {i}")
                            break

    def display_google_ai_insights(self,
                                   results: Dict[str, Dict],
                                   env_config: Dict,
                                   training_history: Dict[str, List[float]] = None):
        """Display Google AI insights"""
        st.header("🤖 Google AI Insights")

        if not self.explainer.available:
            st.warning("Google AI not available. Please provide API key for AI insights.")

            with st.expander("Setup Instructions"):
                st.code("""
                # Set your Google API key
                import os
                os.environ["GOOGLE_API_KEY"] = "your-api-key-here"

                # Or pass to explainer:
                explainer = GoogleAIModelExplainer(api_key="your-api-key")
                """)

            return

        # Create tabs for different AI analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "Performance Analysis",
            "Tuning Recommendations",
            "Decision Explanation",
            "Full Report"
        ])

        with tab1:
            st.subheader("AI Performance Analysis")
            if st.button("Generate AI Analysis", key="analysis_btn"):
                with st.spinner("Generating AI insights..."):
                    analysis = self.explainer.analyze_performance(results, env_config)
                    st.markdown(analysis)

        with tab2:
            st.subheader("Hyperparameter Tuning Recommendations")

            algo_selection = st.selectbox(
                "Select Algorithm",
                list(results.keys())
            )

            if st.button("Get Tuning Recommendations", key="tuning_btn"):
                with st.spinner("Generating recommendations..."):
                    # Get current parameters (simulated - in real app, get from agent objects)
                    current_params = {
                        "learning_rate": 0.001,
                        "gamma": 0.95,
                        "epsilon": 0.1
                    }

                    if training_history and algo_selection in training_history:
                        perf_history = training_history[algo_selection]
                    else:
                        perf_history = []

                    recommendations = self.explainer.provide_hyperparameter_recommendations(
                        algo_selection, current_params, perf_history
                    )
                    st.markdown(recommendations)

        with tab3:
            st.subheader("Agent Decision Explanation")

            col1, col2 = st.columns(2)
            with col1:
                state_input = st.text_area("State (comma-separated)", "0, 0")
                action_input = st.number_input("Action", min_value=0, max_value=5, value=0)

            with col2:
                next_state_input = st.text_area("Next State (optional)", "")
                reward_input = st.number_input("Reward (optional)", value=0.0)

            if st.button("Explain Decision", key="decision_btn"):
                with st.spinner("Analyzing decision..."):
                    # Parse inputs
                    state = np.array([float(x.strip()) for x in state_input.split(',')])
                    action = int(action_input)
                    next_state = None
                    if next_state_input:
                        next_state = np.array([float(x.strip()) for x in next_state_input.split(',')])
                    reward = float(reward_input) if reward_input else None

                    # Simulate agent for explanation
                    class MockAgent:
                        def __init__(self):
                            self.epsilon = 0.1

                        def __class__(self):
                            return type('MockAgent', (), {})

                    mock_agent = MockAgent()

                    explanation = self.explainer.explain_agent_decision(
                        mock_agent, None, state, action, next_state, reward
                    )
                    st.markdown(explanation)

        with tab4:
            st.subheader("Comprehensive Training Report")
            if st.button("Generate Full Report", key="report_btn"):
                with st.spinner("Generating comprehensive report..."):
                    report = self.explainer.generate_training_report(
                        training_history if training_history else {},
                        results,
                        env_config
                    )
                    st.markdown(report)

                    # Provide download option
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name=f"rl_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )

    def run(self, results=None, training_history=None, env_config=None):
        """Run the dashboard"""
        # Sidebar for configuration
        with st.sidebar:
            st.header("Configuration")

            api_key = st.text_input("Google API Key", type="password")
            if api_key and not self.explainer.available:
                self.explainer.api_key = api_key
                self.explainer._configure_google_ai()
                if self.explainer.available:
                    st.success("Google AI configured successfully!")

            st.markdown("---")
            st.header("Training Parameters")

            num_episodes = st.slider("Training Episodes", 100, 5000, 1000, 100)
            num_locations = st.slider("Delivery Locations", 5, 50, 10, 5)
            grid_size = st.slider("Grid Size", 5, 50, 10, 5)

            if st.button("🔄 Retrain Models"):
                st.session_state.retrain = True

        # Main content
        if results:
            self.display_performance_comparison(results)
            st.markdown("---")

        if training_history:
            self.display_training_progress(training_history)
            st.markdown("---")

        if results and env_config:
            self.display_google_ai_insights(results, env_config, training_history)
