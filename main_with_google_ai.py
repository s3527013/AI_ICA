import json
import os
from datetime import datetime
from typing import Dict, List

import google.generativeai as genai
import numpy as np


class GoogleAIModelExplainer:
    """
    Google Gemini AI-based explanation tool for reinforcement learning models.
    """

    def __init__(self, api_key=None, model_name="gemini-pro"):
        """
        Initialize the Google AI explainer.

        Args:
            api_key: Google AI API key
            model_name: Gemini model name ("gemini-pro" or "gemini-pro-vision")
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name

        if not self.api_key:
            print("Warning: No Google API key provided. Using local analysis only.")
            print("Set GOOGLE_API_KEY environment variable or pass api_key parameter.")

        self._configure_google_ai()

    def _configure_google_ai(self):
        """Configure Google Generative AI"""
        try:
            if self.api_key:
                genai.configure(api_key=self.api_key)

                # List available models
                available_models = [m.name for m in genai.list_models()]
                print(f"Available Google AI models: {available_models}")

                # Check if requested model is available
                model_map = {
                    "gemini-pro": "models/gemini-pro",
                    "gemini-pro-vision": "models/gemini-pro-vision"
                }

                if model_map.get(self.model_name) in available_models:
                    self.model = genai.GenerativeModel(self.model_name)
                    self.available = True
                else:
                    print(f"Model {self.model_name} not available. Using fallback.")
                    self.available = False
            else:
                self.available = False

        except Exception as e:
            print(f"Error configuring Google AI: {e}")
            self.available = False

    def analyze_performance(self, results: Dict[str, Dict], env_config: Dict) -> str:
        """
        Analyze model performance using Google Gemini.

        Args:
            results: Dictionary with algorithm performance metrics
            env_config: Environment configuration

        Returns:
            Analysis text
        """
        # Format the results
        results_str = self._format_results_for_google(results)
        env_str = json.dumps(env_config, indent=2)

        prompt = f"""
        You are an expert in reinforcement learning and logistics optimization. 
        Analyze the following delivery route optimization results from three RL algorithms.

        ENVIRONMENT CONFIGURATION:
        {env_str}

        ALGORITHM PERFORMANCE RESULTS:
        {results_str}

        Please provide a comprehensive analysis covering:

        1. PERFORMANCE SUMMARY:
           - Which algorithm performed best overall
           - Statistical significance of differences
           - Key strengths of each algorithm

        2. ALGORITHM COMPARISON:
           - Q-Learning vs SARSA vs DQN trade-offs
           - Convergence speed and stability
           - Sample efficiency

        3. BUSINESS IMPLICATIONS:
           - Impact on delivery time reduction
           - Fuel/energy savings potential
           - Scalability considerations

        4. RECOMMENDATIONS:
           - Best algorithm for production deployment
           - Potential improvements for each algorithm
           - Next steps for optimization

        Format the response in clear markdown with appropriate headings and bullet points.
        Include specific numerical insights when possible.
        """

        if self.available:
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"Google AI analysis failed: {e}")
                return self._fallback_analysis(results, env_config)
        else:
            return self._fallback_analysis(results, env_config)

    def explain_agent_decision(self, agent, env, state, action, next_state=None, reward=None) -> str:
        """
        Explain why an agent made a specific decision.

        Args:
            agent: The RL agent
            env: The environment
            state: Current state
            action: Action taken
            next_state: Next state (optional)
            reward: Reward received (optional)

        Returns:
            Explanation text
        """
        # Get environment context
        state_desc = self._describe_state(state, env)
        action_desc = self._describe_action(action, env)

        # Get agent context
        agent_info = self._get_agent_info(agent)

        prompt = f"""
        You are analyzing a reinforcement learning agent's decision in a delivery route optimization task.

        AGENT INFORMATION:
        {agent_info}

        CURRENT SITUATION:
        {state_desc}

        ACTION TAKEN:
        {action_desc}

        ADDITIONAL CONTEXT:
        - Next State: {next_state if next_state is not None else 'Not provided'}
        - Reward Received: {reward if reward is not None else 'Not provided'}
        - Agent Type: {agent.__class__.__name__}

        Please explain:
        1. Why this action makes sense (or doesn't) given the current state
        2. The trade-off between exploration and exploitation
        3. How this action contributes to the overall delivery optimization goal
        4. Alternative actions that could have been considered

        Keep the explanation concise but insightful, focusing on the optimization perspective.
        """

        if self.available:
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"Google AI explanation failed: {e}")
                return self._simple_decision_explanation(agent, state, action, env)
        else:
            return self._simple_decision_explanation(agent, state, action, env)

    def provide_hyperparameter_recommendations(self,
                                               agent_type: str,
                                               current_params: Dict,
                                               performance_history: List[float]) -> str:
        """
        Provide hyperparameter tuning recommendations using Google AI.

        Args:
            agent_type: Type of RL agent
            current_params: Current hyperparameters
            performance_history: List of rewards over training episodes

        Returns:
            Recommendations text
        """
        # Analyze performance trend
        if len(performance_history) > 0:
            avg_reward = np.mean(performance_history[-100:]) if len(performance_history) >= 100 else np.mean(
                performance_history)
            reward_std = np.std(performance_history[-100:]) if len(performance_history) >= 100 else np.std(
                performance_history)
            trend = "improving" if len(performance_history) > 10 and performance_history[-1] > performance_history[
                0] else "stable or decreasing"
        else:
            avg_reward = 0
            reward_std = 0
            trend = "unknown"

        prompt = f"""
        As an expert in reinforcement learning hyperparameter tuning, analyze the following:

        AGENT TYPE: {agent_type}

        CURRENT HYPERPARAMETERS:
        {json.dumps(current_params, indent=2)}

        PERFORMANCE ANALYSIS:
        - Average Recent Reward: {avg_reward:.2f}
        - Reward Standard Deviation: {reward_std:.2f}
        - Performance Trend: {trend}

        Provide specific, actionable recommendations for:

        1. LEARNING RATE (alpha):
           - Whether to increase or decrease
           - Suggested new value range
           - Reasoning based on current performance

        2. DISCOUNT FACTOR (gamma):
           - Optimization recommendations
           - Impact on long-term vs short-term rewards

        3. EXPLORATION RATE (epsilon):
           - Exploration strategy adjustments
           - Decay schedule recommendations

        4. OTHER PARAMETERS:
           - Any other relevant hyperparameters for this agent type

        5. TRAINING ADJUSTMENTS:
           - Suggested number of episodes
           - Batch size (if applicable)
           - Network architecture suggestions (if DQN)

        Format your response as a structured markdown document with clear sections.
        Include specific numerical suggestions when possible.
        """

        if self.available:
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"Google AI recommendations failed: {e}")
                return self._basic_tuning_recommendations(agent_type, current_params, performance_history)
        else:
            return self._basic_tuning_recommendations(agent_type, current_params, performance_history)

    def generate_training_report(self,
                                 training_history: Dict[str, List[float]],
                                 final_results: Dict[str, Dict],
                                 env_config: Dict) -> str:
        """
        Generate a comprehensive training report.

        Args:
            training_history: Dictionary of reward histories for each agent
            final_results: Final evaluation results
            env_config: Environment configuration

        Returns:
            Comprehensive report in markdown format
        """
        # Prepare data for analysis
        training_summary = {}
        for agent, rewards in training_history.items():
            if len(rewards) > 0:
                training_summary[agent] = {
                    "final_reward": rewards[-1],
                    "avg_last_100": np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
                    "max_reward": max(rewards),
                    "convergence_episode": self._estimate_convergence(rewards)
                }

        prompt = f"""
        # REINFORCEMENT LEARNING TRAINING REPORT

        ## Executive Summary
        Generate a comprehensive report on the reinforcement learning training for delivery route optimization.

        ## Data Provided

        ENVIRONMENT CONFIGURATION:
        ```json
        {json.dumps(env_config, indent=2)}
        ```

        TRAINING SUMMARY:
        ```json
        {json.dumps(training_summary, indent=2)}
        ```

        FINAL EVALUATION RESULTS:
        ```json
        {json.dumps(final_results, indent=2)}
        ```

        Please generate a professional report with the following sections:

        1. EXECUTIVE SUMMARY
           - Overall training success
           - Best performing algorithm
           - Key achievements

        2. METHODOLOGY OVERVIEW
           - Algorithms tested
           - Training approach
           - Evaluation metrics

        3. RESULTS ANALYSIS
           - Detailed algorithm comparison
           - Statistical significance
           - Learning curves analysis

        4. BUSINESS IMPACT
           - Estimated time savings
           - Cost reduction potential
           - Scalability assessment

        5. RECOMMENDATIONS
           - Production deployment recommendations
           - Further optimization opportunities
           - Risk assessment

        6. APPENDICES
           - Technical details
           - Limitations
           - Future work

        Make the report professional, data-driven, and suitable for technical stakeholders.
        Include specific numbers and percentages where applicable.
        """

        if self.available:
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"Google AI report generation failed: {e}")
                return self._generate_basic_report(training_history, final_results, env_config)
        else:
            return self._generate_basic_report(training_history, final_results, env_config)

    def _format_results_for_google(self, results: Dict) -> str:
        """Format results for Google AI prompt"""
        formatted = []
        for algo, metrics in results.items():
            formatted.append(f"\n### {algo}")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    formatted.append(f"- {metric}: {value:.4f}")
                else:
                    formatted.append(f"- {metric}: {value}")
        return "\n".join(formatted)

    def _describe_state(self, state, env) -> str:
        """Describe the current state"""
        if hasattr(env, 'get_state_description'):
            return env.get_state_description(state)
        else:
            return f"State vector (shape: {state.shape}): {state}"

    def _describe_action(self, action, env) -> str:
        """Describe the action taken"""
        if hasattr(env, 'get_action_description'):
            return env.get_action_description(action)
        else:
            return f"Action index: {action}"

    def _get_agent_info(self, agent) -> str:
        """Get information about the agent"""
        info = []
        info.append(f"- Agent Type: {agent.__class__.__name__}")

        # Add hyperparameters if available
        params = ['alpha', 'gamma', 'epsilon', 'learning_rate', 'batch_size']
        for param in params:
            if hasattr(agent, param):
                info.append(f"- {param}: {getattr(agent, param)}")

        # Add exploration rate if available
        if hasattr(agent, 'epsilon'):
            info.append(f"- Exploration Rate (ε): {agent.epsilon:.3f}")
            if agent.epsilon > 0.5:
                info.append("  (High exploration phase)")
            elif agent.epsilon > 0.1:
                info.append("  (Balanced exploration-exploitation)")
            else:
                info.append("  (Mainly exploitation phase)")

        return "\n".join(info)

    def _estimate_convergence(self, rewards: List[float], window: int = 50) -> int:
        """Estimate when the agent converged"""
        if len(rewards) < window * 2:
            return len(rewards)

        # Look for when rewards stabilize
        for i in range(window, len(rewards) - window):
            early_avg = np.mean(rewards[i - window:i])
            late_avg = np.mean(rewards[i:i + window])
            if abs(late_avg - early_avg) < 0.1 * abs(early_avg):  # Less than 10% change
                return i

        return len(rewards)

    def _fallback_analysis(self, results: Dict, env_config: Dict) -> str:
        """Fallback analysis without Google AI"""
        best_algo = max(results.items(), key=lambda x: x[1].get('avg_reward', 0))

        analysis = [
            "# Performance Analysis (Local Analysis)",
            "",
            f"**Best Algorithm**: {best_algo[0]} with average reward: {best_algo[1].get('avg_reward', 0):.2f}",
            "",
            "## Detailed Results:",
        ]

        for algo, metrics in results.items():
            analysis.append(f"### {algo}")
            for metric, value in metrics.items():
                analysis.append(f"- {metric}: {value:.4f}")

        analysis.extend([
            "",
            "## Recommendations:",
            "1. Consider increasing training episodes",
            "2. Try different exploration strategies",
            "3. Monitor convergence more closely",
            "",
            "*Note: For more detailed analysis, provide a Google API key.*"
        ])

        return "\n".join(analysis)

    def _simple_decision_explanation(self, agent, state, action, env) -> str:
        """Simple decision explanation without AI"""
        return f"""
        **Decision Explanation (Local Analysis)**

        Agent: {agent.__class__.__name__}
        Action: {action}

        Explanation:
        The agent selected action {action} based on its current policy.
        This decision balances:
        - Exploration (trying new actions): {getattr(agent, 'epsilon', 'N/A')}
        - Exploitation (using learned knowledge)

        In the context of delivery route optimization, this action moves toward
        the next delivery location or explores new areas to find optimal routes.
        """

    def _basic_tuning_recommendations(self, agent_type, current_params, performance_history) -> str:
        """Basic tuning recommendations without AI"""
        recommendations = [
            f"# Hyperparameter Recommendations for {agent_type}",
            "",
            "## Current Parameters:",
        ]

        for param, value in current_params.items():
            recommendations.append(f"- {param}: {value}")

        recommendations.extend([
            "",
            "## Basic Recommendations:",
            "1. **Learning Rate**: Adjust based on convergence speed",
            "   - Too high: unstable learning",
            "   - Too low: slow convergence",
            "",
            "2. **Discount Factor**:",
            "   - Higher values favor long-term rewards",
            "   - Lower values focus on immediate rewards",
            "",
            "3. **Exploration Rate**:",
            "   - Start high, decay over time",
            "   - Balance exploration vs exploitation",
            "",
            "*Note: For AI-powered recommendations, provide Google API key.*"
        ])

        return "\n".join(recommendations)

    def _generate_basic_report(self, training_history, final_results, env_config) -> str:
        """Generate basic report without AI"""
        report = [
            "# Reinforcement Learning Training Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            f"Training completed for {len(training_history)} algorithms.",
            "",
            "## Algorithms Tested:",
        ]

        for algo in training_history.keys():
            report.append(f"- {algo}")

        report.extend([
            "",
            "## Final Results:",
        ])

        for algo, metrics in final_results.items():
            report.append(f"### {algo}")
            for metric, value in metrics.items():
                report.append(f"- {metric}: {value:.4f}")

        report.extend([
            "",
            "## Environment Configuration:",
            f"- Grid Size: {env_config.get('grid_size', 'N/A')}",
            f"- Locations: {env_config.get('num_locations', 'N/A')}",
            "",
            "---",
            "*This is a basic report. For AI-enhanced analysis, provide Google API key.*"
        ])

        return "\n".join(report)
