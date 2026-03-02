import logging
import os
from datetime import datetime

from src.core.config import config

# Parse log level - extract just the first word to handle comments
log_level = config.log_level.split()[0].upper()

# Validate and set default if invalid
valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
if log_level not in valid_levels:
    log_level = "INFO"

# Create logs directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
os.makedirs(log_dir, exist_ok=True)

# Log file path with date
log_file = os.path.join(log_dir, f'proxy_{datetime.now().strftime("%Y%m%d")}.log')

# Logging format - more detailed for debugging
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"

# Create root logger
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, log_level))

# Remove existing handlers to avoid duplicates
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(getattr(logging, log_level))
console_handler.setFormatter(logging.Formatter(log_format))
root_logger.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
file_handler.setFormatter(logging.Formatter(log_format))
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)
logger.info(f"Logging to file: {log_file}")

# Configure uvicorn to be quieter
for uvicorn_logger in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
    logging.getLogger(uvicorn_logger).setLevel(logging.WARNING)
