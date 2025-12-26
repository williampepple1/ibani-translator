"""
Vercel serverless function entry point for Ibani Translation API.
"""

import os
import sys

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_server import app

# Export the app for Vercel
handler = app

