# src/main.py

import os
import pandas as pd
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from src.data_ingestion.fetch_data import fetch_historical_data, save_raw_data, load_raw_data, get_sp500_tickers
from src.utils.config import PROJECT_ROOT
from src.data_ingestion.preprocess import clean_stock_data, save_processed_data, load_processed_data
from src.feature_engineering.risk_indicators import calculate_returns, calculate_var, calculate_max_drawdown, \
    calculate_sharpe_ratio
from src.models.optimization import optimize_portfolio
from src.evaluation.backtesting import display_portfolio_allocation, save_portfolio_allocation, calculate_risk_metrics, \
    save_risk_metrics

def run_robo_advisor():
    logger = get_logger(__name__)
    config = load_config()

    # Remove the dependency on config['data']['tickers'] and use get_sp500_tickers instead
    tickers = get_sp500_tickers()
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    excel_path = os.path.join(PROJECT_ROOT, config['output']['excel_path'])
    risk_free_rate = config['model']['optimization']['risk_free_rate']
    confidence_level = config['model']['optimization']['confidence_level']

    # Step 1: Data Ingestion
    logger.info("Step 1: Data Ingestion")
    price_data = fetch_historical_data(tickers=tickers, start_date=start_date, end_date=end_date)
    save_raw_data(price_data)
    price_data = load_raw_data()

    # Step 2: Data Preprocessing
    logger.info("Step 2: Data Preprocessing")
    clean_data = clean_stock_data(price_data, coverage_threshold=0.5)  # Adjust threshold as needed
    save_processed_data(clean_data)
    # Optionally, reload processed data
    # clean_data = load_processed_data()

    # Step 3: Feature Engineering
    logger.info("Step 3: Feature Engineering")
    prices = clean_data.set_index('Date')
    returns = calculate_returns(prices)

    # Step 4: Portfolio Optimization
    logger.info("Step 4: Portfolio Optimization")
    optimal_weights = optimize_portfolio(
        returns,
        risk_free_rate=config['model']['optimization']['risk_free_rate'],
        objective=config['model']['optimization']['objective']
    )

    # Step 5: Display and Save Portfolio Allocation
    logger.info("Step 5: Display and Save Portfolio Allocation")
    allocation = display_portfolio_allocation(returns.columns.tolist(), optimal_weights)
    save_portfolio_allocation(allocation, excel_path)

    # Step 6: Calculate and Save Risk Metrics
    logger.info("Step 6: Calculate and Save Risk Metrics")
    risk_metrics = calculate_risk_metrics(returns, optimal_weights, risk_free_rate)
    save_risk_metrics(risk_metrics, excel_path)

    logger.info("Robo Advisor run completed successfully.")

if __name__ == "__main__":
    run_robo_advisor()
