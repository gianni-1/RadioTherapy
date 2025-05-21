import logging

# Configure root logger to output into radiotherapy.log in project root
def configure_logging():
    logging.basicConfig(
        filename='radiotherapy.log',
        filemode='a',
        format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
        level=logging.INFO
    )

# Ensure logging is configured on import
configure_logging()

# Provide module-level logger
logger = logging.getLogger('radiotherapy')
