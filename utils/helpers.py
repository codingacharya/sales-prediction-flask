"""
helpers.py
-----------
Miscellaneous helper functions for logging, formatting, and utilities.
"""

import logging
import os
from datetime import datetime

# Create logs directory if not exists
LOG_DIR = os.path.join(os.getcwd(), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, 'app.log')

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def log_event(message, level="info"):
    """Write logs to file with timestamp."""
    if level == "error":
        logging.error(message)
    else:
        logging.info(message)


def format_prediction(value):
    """Return formatted prediction text."""
    return f"Predicted Sales: {value}"


def current_timestamp():
    """Return current timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
