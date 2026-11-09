import logging

# one logger for the whole backend, full error details go here and never to the client
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("aidetect")