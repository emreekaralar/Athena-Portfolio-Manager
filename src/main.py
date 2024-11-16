# src/main.py

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from src.utils.config import PROJECT_ROOT
from src.utils.scheduler import start_scheduler
from src.robo_advisor import run_robo_advisor  # Ensure this import is correct
from src.models import optimize_portfolio  # Import optimize_portfolio if needed elsewhere

def main():
    """
    Entry point for the Robo Advisor Portfolio Manager.
    Parses command-line arguments to determine whether to run once or start the scheduler.
    """
    parser = argparse.ArgumentParser(description="Robo Advisor Portfolio Manager")
    parser.add_argument(
        '--schedule',
        action='store_true',
        help="Run the robo advisor with automated rebalancing based on the schedule defined in config.yaml."
    )
    args = parser.parse_args()

    if args.schedule:
        # Start the scheduler for automated rebalancing
        start_scheduler()
    else:
        # Run the robo advisor once
        run_robo_advisor()

if __name__ == "__main__":
    main()
