"""
Configuration management for News Sentinel Agent.
"""

import os
from typing import Optional

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


def get_openai_api_key() -> Optional[str]:
    """
    Get OpenAI API key from environment variable or Streamlit secrets.
    
    Returns:
        API key string or None if not found
    """
    # Prefer environment variable first
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        return api_key
    
    # Fallback to Streamlit secrets if available
    if STREAMLIT_AVAILABLE:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", None)
            if api_key:
                return api_key
        except Exception:
            pass
    
    return None


def get_openai_model_name() -> str:
    """
    Get OpenAI model name from environment variable or use default.
    
    Returns:
        Model name (default: "gpt-4o-mini")
    """
    return os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")


def get_tavily_api_key() -> Optional[str]:
    """
    Get Tavily API key from environment variable or Streamlit secrets.
    
    Returns:
        API key string or None if not found
    """
    # Prefer environment variable first
    api_key = os.getenv("TAVILY_API_KEY")
    
    if api_key:
        return api_key
    
    # Fallback to Streamlit secrets if available
    if STREAMLIT_AVAILABLE:
        try:
            api_key = st.secrets.get("TAVILY_API_KEY", None)
            if api_key:
                return api_key
        except Exception:
            pass
    
    return None

