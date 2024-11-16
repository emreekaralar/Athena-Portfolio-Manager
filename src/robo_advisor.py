# src/robo_advisor.py

import os
import pandas as pd
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from src.data_ingestion.fetch_data import (
    fetch_historical_data,
    save_raw_data,
    load_raw_data,
    get_sp500_tickers_with_sector
)
from src.utils.config import PROJECT_ROOT
from src.data_ingestion.preprocess import (
    clean_stock_data,
    save_processed_data,
    load_processed_data
)
from src.feature_engineering.risk_indicators import (
    calculate_returns,
    calculate_var,
    calculate_max_drawdown,
    calculate_sharpe_ratio
)
from src.models import optimize_portfolio
from src.evaluation.backtesting import (
    display_portfolio_allocation,
    save_portfolio_allocation,
    calculate_risk_metrics,
    save_risk_metrics
)
from src.feature_engineering.screening import (
    sector_specific_screening
)


def run_robo_advisor():
    """
    Executes the robo-advisor workflow:
    1. Data Ingestion
    2. Data Preprocessing
    3. Feature Engineering - Screening
    4. Portfolio Optimization
    5. Display and Save Portfolio Allocation
    6. Calculate and Save Risk Metrics
    """
    logger = get_logger(__name__)
    config = load_config()

    # Load configuration settings
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    excel_path = os.path.join(PROJECT_ROOT, config['output']['excel_path'])
    risk_free_rate = config['model']['optimization']['risk_free_rate']
    confidence_level = config['model']['optimization']['confidence_level']
    optimization_method = config['model']['optimization']['method']
    objective = config['model']['optimization'].get('objective', "maximize_sharpe")
    long_short_ratio = config['model']['optimization'].get('long_short_ratio', 1.0)

    # Step 1: Data Ingestion
    logger.info("Step 1: Data Ingestion")
    # Fetch historical data along with sector information
    price_data, tickers_df = fetch_historical_data(tickers_df=None, start_date=start_date, end_date=end_date)
    save_raw_data(price_data)
    price_data = load_raw_data()

    # Step 2: Data Preprocessing
    logger.info("Step 2: Data Preprocessing")
    clean_data = clean_stock_data(price_data, coverage_threshold=0.5)  # Adjust threshold as needed
    save_processed_data(clean_data)
    # Optionally, reload processed data
    # clean_data = load_processed_data()

    # Step 3: Feature Engineering - Screening
    logger.info("Step 3: Feature Engineering - Screening")

    # Reload tickers_df in case it was modified during data ingestion
    tickers_df = get_sp500_tickers_with_sector()

    criteria_config = config['screening']['sectors']

    # Apply sector-specific screening
    filtered_tickers = sector_specific_screening(clean_data.set_index('Date'), tickers_df, criteria_config)

    # Proceed with filtered tickers
    filtered_price_data = clean_data.set_index('Date')[filtered_tickers['Ticker'].tolist()]
    returns = calculate_returns(filtered_price_data)

    # Step 4: Portfolio Optimization
    logger.info("Step 4: Portfolio Optimization")

    # Select optimization method based on configuration
    if optimization_method == "long_short":
        optimal_weights = optimize_portfolio(
            returns,
            risk_free_rate=risk_free_rate,
            method="long_short",
            objective=objective,
            long_short_ratio=long_short_ratio
        )
    else:
        optimal_weights = optimize_portfolio(
            returns,
            risk_free_rate=risk_free_rate,
            method=optimization_method
        )

    # Step 5: Display and Save Portfolio Allocation
    logger.info("Step 5: Display and Save Portfolio Allocation")
    allocation = display_portfolio_allocation(filtered_price_data.columns.tolist(), optimal_weights)
    save_portfolio_allocation(allocation, excel_path)

    # Step 6: Calculate and Save Risk Metrics
    logger.info("Step 6: Calculate and Save Risk Metrics")
    risk_metrics = calculate_risk_metrics(returns, optimal_weights, risk_free_rate)
    save_risk_metrics(risk_metrics, excel_path)

    logger.info("Robo Advisor run completed successfully.")
