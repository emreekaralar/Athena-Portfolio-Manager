# src/feature_engineering/risk_indicators.py

import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def calculate_returns(prices):
    """
    Calculates daily returns from adjusted close prices.
    """
    logger.info("Calculating daily returns.")
    returns = prices.pct_change().dropna()
    logger.info("Daily returns calculated.")
    return returns

def calculate_var(returns, confidence_level=0.95):
    """
    Calculates Value at Risk (VaR) at the specified confidence level.
    """
    logger.info(f"Calculating Value at Risk (VaR) at {confidence_level*100}% confidence level.")
    var = np.percentile(returns, (1 - confidence_level) * 100)
    logger.info(f"VaR calculated: {var:.4f}")
    return var

def calculate_max_drawdown(cumulative_returns):
    """
    Calculates the maximum drawdown from cumulative returns.
    """
    logger.info("Calculating Max Drawdown.")
    drawdown = cumulative_returns / cumulative_returns.cummax() - 1
    max_drawdown = drawdown.min()
    logger.info(f"Max Drawdown calculated: {max_drawdown:.4f}")
    return max_drawdown

def calculate_sharpe_ratio(returns, risk_free_rate=0.04):
    """
    Calculates the Sharpe Ratio of the portfolio.
    """
    logger.info("Calculating Sharpe Ratio.")
    excess_return = returns.mean() - risk_free_rate / 252  # Assuming daily returns
    portfolio_volatility = returns.std()
    sharpe_ratio = excess_return / portfolio_volatility
    mean_sharpe = sharpe_ratio.mean() * np.sqrt(252)  # Annualized Sharpe Ratio
    logger.info(f"Sharpe Ratio calculated: {mean_sharpe:.4f}")
    return mean_sharpe
