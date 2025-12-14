import json
import os
from typing import Dict, List, Optional

import google.generativeai as genai
import numpy as np


class GoogleAIModelExplainer:
    """
    Google Gemini AI-based explanation tool for reinforcement learning models.
    """

    def __init__(self, api_key=None, model_name="gemini-2.5-pro"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name

        if not self.api_key:
            print("Warning: No Google API key provided. Using local analysis only.")
        
        self._configure_google_ai()

    def _configure_google_ai(self):
        try:
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                self.available = True
            else:
                self.available = False
        except Exception as e:
            print(f"Error configuring Google AI: {e}")
            self.available = False

    def generate_locations_for_city(self, city: str, num_locations: int) -> Optional[List[str]]:
        """
        Uses Google Gemini to generate a list of realistic delivery addresses in a city.
        Returns None if AI is unavailable or fails, triggering the fallback.
        """
        prompt = f"""
        You are a logistics planning assistant. I need to create a simulation for a delivery vehicle in {city}.
        Please generate a list of {num_locations} realistic delivery locations within {city}.
        The first location should be a central depot. The rest should be a mix of commercial and residential addresses, landmarks, or points of interest.
        
        Return the list as a JSON array of strings. For example:
        ["Depot, Central Middlesbrough", "Teesside University, Middlesbrough", "Riverside Stadium, Middlesbrough", ...]
        """
        
        if not self.available:
            print("Google AI not available for location generation.")
            return None

        try:
            response = self.model.generate_content(prompt)
            # Clean up the response to extract the JSON part
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
            locations = json.loads(cleaned_response)
            if isinstance(locations, list) and len(locations) == num_locations:
                return locations
            else:
                print("AI returned unexpected format.")
                return None
        except (Exception, json.JSONDecodeError) as e:
            print(f"Failed to generate locations with AI. Error: {e}")
            return None


    def analyze_performance(self, results: Dict[str, Dict], env_config: Dict) -> str:
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
        1. PERFORMANCE SUMMARY: Which algorithm performed best and why.
        2. ALGORITHM COMPARISON: Trade-offs between the algorithms (e.g., sample efficiency, stability).
        3. BUSINESS IMPLICATIONS: What these results mean for a real-world delivery business.
        4. RECOMMENDATIONS: Which algorithm to deploy and potential improvements.
        """

        if self.available:
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"Google AI analysis failed: {e}"
        return "AI analysis not available."

    def provide_hyperparameter_recommendations(self, agent_type: str, performance_history: List[float]) -> str:
        if len(performance_history) < 10:
            return "Not enough data for tuning recommendations."

        trend = "improving" if np.mean(performance_history[-10:]) > np.mean(performance_history[:10]) else "stagnant or degrading"

        prompt = f"""
        As an expert in reinforcement learning, analyze the performance of a {agent_type} agent.
        The reward history shows a trend that is {trend}.
        The last 10 rewards are: {performance_history[-10:]}.

        Based on this, provide specific, actionable hyperparameter tuning recommendations for:
        - alpha (learning rate)
        - gamma (discount factor)
        - epsilon_decay (exploration decay rate)

        Suggest new values or ranges and provide your reasoning. Return the recommendations as a JSON object.
        Example: {{"alpha": 0.05, "gamma": 0.98, "epsilon_decay": 0.999, "reasoning": "..."}}
        """

        if self.available:
            try:
                response = self.model.generate_content(prompt)
                cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
                return json.loads(cleaned_response)
            except (Exception, json.JSONDecodeError) as e:
                return f"AI recommendations failed: {e}"
        return "AI recommendations not available."

    def generate_training_report(self, training_history: Dict[str, List[float]], final_results: Dict[str, Dict], env_config: Dict) -> str:
        # This method can be expanded similarly to the others
        return "Training report generation is a premium feature."

    def _format_results_for_google(self, results: Dict) -> str:
        return json.dumps(results, indent=2)
