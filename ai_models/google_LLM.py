import json
import os
import re
from typing import Dict, List, Optional

import google.generativeai as genai
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class GoogleAIModelExplainer:
    """
    Google Gemini AI-based explanation tool for reinforcement learning models.
    """

    def __init__(self, model_name="gemini-2.5-pro"):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name

        if not self.api_key:
            print("Warning: No Google API key provided. Using local analysis only.")

        self._configure_google_ai()

    def _configure_google_ai(self):
        try:
            if self.api_key:
                genai.configure(api_key=self.api_key)
                # Configure the model to expect a JSON response
                self.model = genai.GenerativeModel(
                    self.model_name,
                    generation_config={"response_mime_type": "application/json"}
                )
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
        Generate a list of {num_locations} realistic delivery locations within {city}.
        The first location should be a central depot. The rest should be a mix of commercial and residential addresses, landmarks, or points of interest.
        
        IMPORTANT: Your response MUST be a raw JSON array of strings, with no additional text, conversation, or markdown formatting.
        
        Example of valid output:
        ["Depot, Central Middlesbrough", "Teesside University, Middlesbrough", "Riverside Stadium, Middlesbrough"]
        """

        if not self.available:
            print("Google AI not available for location generation.")
            return None

        try:
            response = self.model.generate_content(prompt)
            print(response)
            # Directly parse the JSON from the response text
            locations = json.loads(response.text)
            print(locations)
            if isinstance(locations, list) and len(locations) == num_locations:
                return locations
            else:
                print("AI returned unexpected JSON format or incorrect number of locations.")
                return None
        except (Exception, json.JSONDecodeError) as e:
            print(f"Failed to generate or parse locations with AI. Error: {e}")
            return None

    def analyze_performance(self, results: Dict[str, Dict], env_config: Dict) -> str:
        # This method is designed to return a natural language string, so it doesn't need JSON enforcement.
        results_str = self._format_results_for_google(results)
        env_str = json.dumps(env_config, indent=2, cls=NumpyEncoder)

        prompt = f"""
        You are an expert in reinforcement learning and logistics optimization. 
        Analyze the following delivery route optimization results.

        ENVIRONMENT CONFIGURATION:
        {env_str}

        ALGORITHM PERFORMANCE RESULTS:
        {results_str}

        Please provide a comprehensive, well-structured analysis covering:
        1. PERFORMANCE SUMMARY: Which algorithm performed best and why (consider both distance and time).
        2. ALGORITHM COMPARISON: Discuss the trade-offs between the different algorithm types (Informed Search vs. RL, sample efficiency vs. tuning efficiency).
        3. BUSINESS IMPLICATIONS: What these results mean for a real-world delivery business.
        4. RECOMMENDATIONS: Which algorithm you would recommend for deployment and why.
        5. IMPROVEMENTS: Suggestions for overall improvement
        """

        # Re-configure model for text response for this specific call
        text_model = genai.GenerativeModel(self.model_name)
        if self.available:
            try:
                response = text_model.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"Google AI analysis failed: {e}"
        return "AI analysis not available."

    def provide_hyperparameter_recommendations(self, agent_type: str, performance_history: List[float]) -> str:
        if len(performance_history) < 10:
            return "Not enough data for tuning recommendations."

        prompt = f"""
        As an expert in reinforcement learning, analyze the performance of a {agent_type} agent.
        The last 50 rewards are: {performance_history[-50:]}.
        Provide specific, actionable hyperparameter tuning recommendations for alpha (learning rate), gamma (discount factor), and epsilon_decay.
        
        IMPORTANT: Your response MUST be a raw JSON object, with no additional text, conversation, or markdown formatting.
        Example of valid output:
        {{"alpha": 0.05, "gamma": 0.98, "epsilon_decay": 0.999, "reasoning": "The agent's learning appears to have plateaued, suggesting a smaller learning rate..."}}
        """

        if self.available:
            try:
                response = self.model.generate_content(prompt)
                # Directly parse the JSON from the response text
                return json.loads(response.text)
            except (Exception, json.JSONDecodeError) as e:
                return f"AI recommendations failed: {e}"
        return "AI recommendations not available."

    def _format_results_for_google(self, results: Dict) -> str:
        return json.dumps(results, indent=2, cls=NumpyEncoder)
