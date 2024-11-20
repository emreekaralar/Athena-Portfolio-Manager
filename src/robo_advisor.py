# src/robo_advisor.py

import os
import pandas as pd
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from src.utils.config import PROJECT_ROOT
from src.data_ingestion.fetch_data import (
    fetch_historical_data,
    save_raw_data,
    load_raw_data,
    get_sp500_tickers_with_sector,
    collect_stocks_data,
    save_financial_data_to_excel
)
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
from src.models.optimization import optimize_portfolio
from src.evaluation.backtesting import (
    display_portfolio_allocation,
    save_portfolio_allocation,
    calculate_risk_metrics,
    save_risk_metrics
)
from src.feature_engineering.screening import (
    sector_specific_screening
)
from src.feature_engineering.scoring import calculate_composite_scores

def run_robo_advisor():
    """
    Executes the robo-advisor workflow.
    """
    logger = get_logger(__name__)
    config = load_config()

    # Load configuration settings
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    excel_path = os.path.join(PROJECT_ROOT, config['output']['excel_path'])
    risk_free_rate = config['model']['optimization']['risk_free_rate']
    optimization_method = config['model']['optimization']['method']
    objective = config['model']['optimization'].get('objective', "maximize_sharpe")
    long_short_ratio = config['model']['optimization'].get('long_short_ratio', 1.0)

    # Step 1: Data Ingestion
    logger.info("Step 1: Data Ingestion")
    price_data, tickers_df = fetch_historical_data(start_date=start_date, end_date=end_date)
    save_raw_data(price_data)
    price_data = load_raw_data()


    # Collect additional financial data and save to Excel
    logger.info("Collecting additional financial data for tickers.")
    tickers_list = tickers_df['Ticker'].tolist()
    financial_data = collect_stocks_data(tickers_list)
    save_financial_data_to_excel(financial_data, filename='financial_data.xlsx')

    # Step 2: Data Preprocessing
    logger.info("Step 2: Data Preprocessing")
    clean_data = clean_stock_data(price_data, coverage_threshold=0.3)
    save_processed_data(clean_data)

    # Step 3: Feature Engineering - Scoring and Ranking
    logger.info("Step 3: Feature Engineering - Scoring and Ranking")

    # Load scoring configuration
    scoring_config = config['scoring']

    # Calculate composite scores
    scores_df = calculate_composite_scores(financial_data, scoring_config)

    # Number of stocks to long and short in each sector
    num_long = scoring_config.get('num_long', 5)
    num_short = scoring_config.get('num_short', 5)

    # Get available tickers from clean_data
    available_tickers = set(clean_data.columns) - set(['Date'])  # Exclude 'Date' if present

    # Perform sector-specific ranking and select stocks
    long_tickers, short_tickers = sector_specific_screening(
        financial_data,
        scores_df,
        tickers_df,
        num_long,
        num_short,
        available_tickers
    )

    # Combine long and short tickers
    selected_tickers = pd.concat([long_tickers, short_tickers]).reset_index(drop=True)

    # Get the price data for the selected tickers
    filtered_price_data = clean_data.set_index('Date')[selected_tickers['Ticker'].tolist()]
    returns = calculate_returns(filtered_price_data)

    # Step 4: Portfolio Optimization
    logger.info("Step 4: Portfolio Optimization")

    # Optimize the portfolio
    optimized_weights = optimize_portfolio(
        returns,
        risk_free_rate=risk_free_rate,
        method=optimization_method,
        objective=objective,
        long_short_ratio=long_short_ratio
    )

    # Step 5: Display and Save Portfolio Allocation
    logger.info("Step 5: Display and Save Portfolio Allocation")
    allocation = display_portfolio_allocation(returns.columns.tolist(), optimized_weights)
    save_portfolio_allocation(allocation, excel_path)

    # Step 6: Calculate and Save Risk Metrics
    logger.info("Step 6: Calculate and Save Risk Metrics")
    risk_metrics = calculate_risk_metrics(returns, optimized_weights, risk_free_rate)
    save_risk_metrics(risk_metrics, excel_path)

    logger.info("Robo Advisor run completed successfully.")
