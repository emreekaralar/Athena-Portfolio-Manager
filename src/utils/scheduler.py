# src/utils/scheduler.py

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from src.robo_advisor import run_robo_advisor
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger(__name__)

def start_scheduler():
    """
    Starts the scheduler based on the configuration.
    """
    config = load_config()
    rebalance_config = config['rebalance']
    frequency = rebalance_config['frequency']

    scheduler = BackgroundScheduler()

    if frequency == "daily":
        trigger = CronTrigger(hour=10, minute=0)
        scheduler.add_job(run_robo_advisor, trigger, id='daily_rebalance')
        logger.info("Scheduled daily rebalancing at 10:00 AM.")
    elif frequency == "weekly":
        day = rebalance_config.get('day_of_week', 'monday').lower()
        trigger = CronTrigger(day_of_week=day, hour=10, minute=0)
        scheduler.add_job(run_robo_advisor, trigger, id='weekly_rebalance')
        logger.info(f"Scheduled weekly rebalancing on {day.capitalize()} at 10:00 AM.")
    elif frequency == "monthly":
        day = rebalance_config.get('day_of_month', 1)
        trigger = CronTrigger(day=day, hour=10, minute=0)
        scheduler.add_job(run_robo_advisor, trigger, id='monthly_rebalance')
        logger.info(f"Scheduled monthly rebalancing on day {day} at 10:00 AM.")
    else:
        logger.error(f"Unsupported rebalance frequency: {frequency}")
        return

    scheduler.start()
    logger.info("Scheduler started. Press Ctrl+C to exit.")

    try:
        # Keep the main thread alive to let the scheduler run in the background
        while True:
            pass
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler shut down successfully.")
