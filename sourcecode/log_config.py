import logging
import os

# Configure root logger to output into radiotherapy.log in project root
def configure_logging():
    # Always use the project root for the log file
    # Get the directory of this script (sourcecode/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root
    project_root = os.path.dirname(script_dir)
    log_file = os.path.join(project_root, 'radiotherapy.log')
    
    # Clear any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure logging with file handler
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
        level=logging.INFO,
        force=True  # Force reconfiguration
    )
    
    print(f"Logging configured to write to: {log_file}")

# Ensure logging is configured on import
configure_logging()

# Provide module-level logger
logger = logging.getLogger('radiotherapy')
