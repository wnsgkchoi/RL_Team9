"""
Environment variables and API keys.

IMPORTANT: Do not commit this file with real API keys!
Add this file to .gitignore to keep your keys secure.
"""

import os

# ChatGPT API Key
# You can set this via environment variable or directly here
CHATGPT_API = os.environ.get("OPENAI_API_KEY", None)
# If you want to set it directly (NOT recommended for production):
# CHATGPT_API = "your-api-key-here"

# Alternative name for compatibility
chatgpt_api = CHATGPT_API

# Add other API keys here as needed
# ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", None)
# COHERE_API_KEY = os.environ.get("COHERE_API_KEY", None)

