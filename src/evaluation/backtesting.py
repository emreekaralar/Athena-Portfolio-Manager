# src/evaluation/backtesting.py

import pandas as pd
from src.utils.logger import get_logger
import os
from src.feature_engineering.risk_indicators import calculate_var, calculate_max_drawdown, calculate_sharpe_ratio

logger = get_logger(__name__)


def display_portfolio_allocation(tickers, weights):
    """
    Displays the optimal portfolio allocation.
    """
    allocation = pd.DataFrame({
        'Ticker': tickers,
        'Weight': weights
    })
    logger.info("\nOptimal Portfolio Allocation:")
    logger.info(allocation)
    return allocation


def save_portfolio_allocation(allocation, excel_path):
    """
    Saves the portfolio allocation to an Excel file.
    """
    allocation.to_excel(excel_path, index=False)
    logger.info(f"Portfolio allocation saved to {excel_path}")


def calculate_risk_metrics(returns, weights, risk_free_rate=0.04):
    """
    Calculates VaR, Max Drawdown, and Sharpe Ratio for the portfolio.
    """
    logger.info("Calculating risk metrics for the portfolio.")
    portfolio_returns = returns.dot(weights)
    var = calculate_var(portfolio_returns, confidence_level=0.95)

    cumulative_returns = (1 + portfolio_returns).cumprod()
    max_drawdown = calculate_max_drawdown(cumulative_returns)

    sharpe_ratio = calculate_sharpe_ratio(portfolio_returns, risk_free_rate)

    logger.info(f"Value at Risk (VaR): {var:.4f}")
    logger.info(f"Max Drawdown: {max_drawdown:.4f}")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")

    risk_metrics = {
        'Value at Risk (VaR)': var,
        'Max Drawdown': max_drawdown,
        'Sharpe Ratio': sharpe_ratio
    }

    return risk_metrics


# src/evaluation/backtesting.py

def save_risk_metrics(risk_metrics, excel_path):
    """
    Saves the risk metrics to the same Excel file.
    """
    metrics_df = pd.DataFrame(list(risk_metrics.items()), columns=['Metric', 'Value'])

    # Check if the Excel file exists
    if not os.path.exists(excel_path):
        metrics_df.to_excel(excel_path, sheet_name='Risk Metrics', index=False)
    else:
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl') as writer:
            metrics_df.to_excel(writer, sheet_name='Risk Metrics', index=False)

    logger.info(f"Risk metrics saved to {excel_path}")
