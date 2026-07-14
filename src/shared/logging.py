import logging
import time
import sys

# Configure root logger to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)

class FeatureLogger:
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        self.start_time = None
    
    def start(self):
        self.start_time = time.time()
        self.logger.info(f"[{self.name}] START")
    
    def done(self):
        elapsed = time.time() - self.start_time
        self.logger.info(f"[{self.name}] DONE ({elapsed:.1f}s)")
    
    def error(self, msg: str):
        self.logger.error(f"[{self.name}] ERROR: {msg}")
