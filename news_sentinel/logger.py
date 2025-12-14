"""
Logging utility for News Sentinel Agent.
Provides both Python logging and Streamlit-compatible logging.
"""

import logging
import sys
from collections import deque
from typing import Optional

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


# Global log buffer for capturing logs
_log_buffer: Optional[deque] = None


def get_log_buffer() -> deque:
    """Get or create the global log buffer."""
    global _log_buffer
    if _log_buffer is None:
        _log_buffer = deque(maxlen=1000)  # Keep last 1000 log entries
    return _log_buffer


def clear_log_buffer():
    """Clear the log buffer."""
    global _log_buffer
    if _log_buffer is not None:
        _log_buffer.clear()


def get_logs() -> list:
    """Get all logs from the buffer."""
    if _log_buffer is None:
        return []
    return list(_log_buffer)


class StreamlitLogHandler(logging.Handler):
    """Custom logging handler that outputs to Streamlit and captures logs."""
    
    def __init__(self, container=None, buffer=None):
        super().__init__()
        self.container = container
        self.buffer = buffer or get_log_buffer()
    
    def emit(self, record):
        """Emit a log record to Streamlit and buffer."""
        try:
            msg = self.format(record)
            
            # Add to buffer
            log_entry = {
                'level': record.levelno,
                'levelname': record.levelname,
                'message': msg,
                'name': record.name
            }
            self.buffer.append(log_entry)
            
            # Try to output to Streamlit if available and container is provided
            if STREAMLIT_AVAILABLE and self.container is not None:
                try:
                    # Map log levels to Streamlit functions
                    if record.levelno >= logging.ERROR:
                        self.container.error(f"🔴 {msg}")
                    elif record.levelno >= logging.WARNING:
                        self.container.warning(f"⚠️ {msg}")
                    elif record.levelno >= logging.INFO:
                        self.container.info(f"ℹ️ {msg}")
                    else:  # DEBUG
                        self.container.text(f"🔍 {msg}")
                except Exception:
                    pass  # Fail silently if container is not available
        except Exception:
            pass  # Fail silently


def setup_logger(
    name: str = "news_sentinel",
    level: int = logging.INFO,
    streamlit_container: Optional[object] = None,
    enable_console: bool = True,
) -> logging.Logger:
    """
    Set up a logger for News Sentinel.
    
    Args:
        name: Logger name
        level: Logging level (logging.DEBUG, INFO, WARNING, ERROR)
        streamlit_container: Streamlit container to write logs to (e.g., st.sidebar, or a specific container)
        enable_console: Whether to also log to console
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Streamlit handler - always add buffer handler, optionally add container handler
    buffer_handler = StreamlitLogHandler(container=None, buffer=get_log_buffer())
    buffer_handler.setLevel(level)
    buffer_formatter = logging.Formatter('%(levelname)s: %(message)s')
    buffer_handler.setFormatter(buffer_formatter)
    logger.addHandler(buffer_handler)
    
    # Also add container handler if provided
    if STREAMLIT_AVAILABLE and streamlit_container is not None:
        container_handler = StreamlitLogHandler(container=streamlit_container, buffer=get_log_buffer())
        container_handler.setLevel(level)
        container_handler.setFormatter(buffer_formatter)
        logger.addHandler(container_handler)
    
    return logger


def get_logger(name: str = "news_sentinel") -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set up default
    if not logger.handlers:
        logger = setup_logger(name, level=logging.INFO, enable_console=True)
    
    return logger

