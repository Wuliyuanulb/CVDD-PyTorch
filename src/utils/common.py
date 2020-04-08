import time
import logging

class TimeProfile:
    def __init__(self, tag):
        self.tag = tag
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        self.logger.info(f"{self.tag} - Start:")
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        elapsed_time = end_time - self.start_time
        self.logger.info(f"{self.tag} - End with {elapsed_time:.4f}s elapsed.")