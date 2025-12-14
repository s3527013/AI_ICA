# quick_start.py
import os
import subprocess
import sys


def setup_environment():
    """Setup the environment for Google AI integration"""
    print("Setting up environment for Google AI RL explanations...")

    # Check for Google API key
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("\n⚠️  Google API key not found!")
        print("\nTo get a Google API key:")
        print("1. Visit: https://makersuite.google.com/app/apikey")
        print("2. Create a new API key")
        print("3. Set it as environment variable:")
        print("   Linux/Mac: export GOOGLE_API_KEY='your-key-here'")
        print("   Windows: set GOOGLE_API_KEY=your-key-here")
        print("\nOr create a .env file with GOOGLE_API_KEY=your-key")

        use_without_key = input("\nContinue without Google AI? (y/n): ")
        if use_without_key.lower() != 'y':
            sys.exit(1)

    # Install required packages
    print("\nInstalling required packages...")
    requirements = [
        "google-generativeai",
        "streamlit",
        "plotly",
        "pandas",
        "numpy",
        "scikit-learn",
        "python-dotenv"
    ]

    for package in requirements:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

    print("\n✅ Setup complete!")
    print("\nTo run with Google AI:")
    print("1. Set your Google API key")
    print("2. Run: python main_with_google_ai.py")
    print("\nTo launch dashboard:")
    print("streamlit run dashboard_app.py")


if __name__ == "__main__":
    setup_environment()
