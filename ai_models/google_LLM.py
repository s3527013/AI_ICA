"""
This module provides a client for interacting with Google's Generative AI models (Gemini).
It includes functionality for generating realistic location data, analyzing algorithm performance,
and providing hyperparameter recommendations.
"""

import json
import os
import re
from typing import Dict, List, Optional

import google.generativeai as genai
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    A custom JSON encoder to handle NumPy data types.
    This is necessary because the default `json.dumps` function cannot serialize
    NumPy-specific types like `np.int64` or `np.float64`.
    """

    def default(self, obj):
        """
        Converts NumPy types to their standard Python equivalents.
        """
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
    A client that uses the Google Gemini AI model to provide insights and data
    for the delivery optimization simulation.
    """

    def __init__(self, model_name="gemini-2.5-pro"):
        """
        Initializes the GoogleAIModelExplainer.

        Args:
            model_name (str): The name of the Gemini model to use.
        """
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        self.available = False

        if not self.api_key:
            print("Warning: GOOGLE_API_KEY environment variable not found. AI features will be disabled.")
        
        self._configure_google_ai()

    def _configure_google_ai(self):
        """
        Configures the Google Generative AI client if an API key is available.
        """
        try:
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                self.available = True
                print("Google AI client configured successfully.")
            else:
                self.available = False
        except Exception as e:
            print(f"Error configuring Google AI: {e}")
            self.available = False

    def generate_locations_for_city(self, city: str, num_locations: int) -> Optional[List[str]]:
        """
        Uses the Gemini model to generate a list of realistic delivery addresses for a given city.

        Args:
            city (str): The name of the city for which to generate locations.
            num_locations (int): The number of delivery locations to generate.

        Returns:
            Optional[List[str]]: A list of address strings, or None if the generation fails.
        """
        prompt = f"""
        You are a logistics planning assistant. Your task is to generate a list of {num_locations}
        realistic delivery locations within {city}. The first location must be a central depot.
        The rest should be a mix of commercial and residential addresses.

        IMPORTANT: Your response MUST be a raw JSON array of strings, with no additional text,
        conversation, or markdown formatting.
        """

        if not self.available:
            print("Google AI not available for location generation.")
            return None

        try:
            # Configure the model to expect a JSON response for this specific call
            json_model = genai.GenerativeModel(
                self.model_name,
                generation_config={"response_mime_type": "application/json"}
            )
            response = json_model.generate_content(prompt)
            locations = json.loads(response.text)
            
            if isinstance(locations, list) and len(locations) == num_locations:
                return locations
            else:
                print("AI returned unexpected JSON format or incorrect number of locations.")
                return None
        except (Exception, json.JSONDecodeError) as e:
            print(f"Failed to generate or parse locations with AI. Error: {e}")
            return None

    def analyze_performance(self, results: Dict[str, Dict], env_config: Dict) -> str:
        """
        Generates a natural language analysis of the performance results from a single simulation.

        Args:
            results (Dict): A dictionary containing the performance metrics for each algorithm.
            env_config (Dict): A dictionary containing the configuration of the environment.

        Returns:
            str: A markdown-formatted string containing the AI's analysis.
        """
        results_str = self._format_results_for_google(results)
        env_str = json.dumps(env_config, indent=2, cls=NumpyEncoder)

        prompt = f"""
        You are an expert in reinforcement learning and logistics optimization. 
        Analyze the following delivery route optimization results.

        ENVIRONMENT CONFIGURATION:
        {env_str}

        ALGORITHM PERFORMANCE RESULTS:
        {results_str}

        Please provide a comprehensive, well-structured analysis in markdown format covering:
        1. **PERFORMANCE SUMMARY**: Which algorithm performed best and why (consider both distance and time).
        2. **ALGORITHM COMPARISON**: Discuss the trade-offs between the different algorithm types (Informed Search vs. RL).
        3. **BUSINESS IMPLICATIONS**: What these results mean for a real-world delivery business.
        4. **RECOMMENDATIONS**: Which algorithm you would recommend for deployment and why.
        """

        text_model = genai.GenerativeModel(self.model_name)
        if self.available:
            try:
                response = text_model.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"Google AI analysis failed: {e}"
        return "AI analysis not available."

    def analyze_multiple_scenarios(self, all_scenario_results: Dict) -> str:
        """
        Analyzes and compares the results from multiple simulation scenarios.

        Args:
            all_scenario_results (Dict): A dictionary where keys are scenario names and
                                          values are the results from that scenario.

        Returns:
            str: A markdown-formatted string with a high-level comparison.
        """
        results_str = self._format_results_for_google(all_scenario_results)

        prompt = f"""
        You are a world-class expert in logistics and algorithmic performance analysis.
        You have been given the results from multiple delivery route optimization scenarios.
        Analyze the results below to provide a high-level summary and comparison.

        RESULTS FROM ALL SCENARIOS:
        {results_str}

        Please provide a comprehensive analysis in markdown format focusing on:
        1.  **Scalability Analysis**: How did the performance of each algorithm change as the problem scale increased?
        2.  **Best Overall Algorithm**: Based on scalability and performance, which algorithm is best for a real-world deployment?
        3.  **Key Takeaway**: What is the single most important insight a logistics manager should learn from this?
        4.  **Final Recommendation**: Conclude with a definitive recommendation.
        """
        text_model = genai.GenerativeModel(self.model_name)
        if self.available:
            try:
                response = text_model.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"Google AI multi-scenario analysis failed: {e}"
        return "AI multi-scenario analysis not available."

    def _format_results_for_google(self, results: Dict) -> str:
        """
        A helper function to serialize a results dictionary to a JSON string,
        correctly handling any NumPy data types.
        """
        return json.dumps(results, indent=2, cls=NumpyEncoder)
