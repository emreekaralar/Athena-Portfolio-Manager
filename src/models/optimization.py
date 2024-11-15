# src/models/optimization.py

import numpy as np
from scipy.optimize import minimize
from src.utils.logger import get_logger

logger = get_logger(__name__)


def optimize_portfolio(returns, risk_free_rate=0.04, objective="minimize_risk"):
    """
    Optimizes the portfolio using mean-variance optimization.

    Parameters:
    - returns (DataFrame): DataFrame of asset returns.
    - risk_free_rate (float): Risk-free rate for Sharpe ratio calculation.
    - objective (str): Optimization objective, options include:
        - "minimize_risk": Minimize portfolio risk (standard deviation).
        - "maximize_sharpe": Maximize the Sharpe ratio.

    Returns:
    - optimized_weights (ndarray): Array of optimized asset weights that sum to 1.
    """
    logger.info("Starting portfolio optimization.")

    num_assets = len(returns.columns)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Initial guess (equal weights)
    weights = np.ones(num_assets) / num_assets
    logger.debug(f"Initial weights: {weights}")

    # Constraints: Sum of weights = 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    # Bounds: No short selling (weights between 0 and 1)
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Define objective functions for different optimization goals
    def portfolio_risk(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def sharpe_ratio(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = portfolio_risk(weights)
        return -(portfolio_return - risk_free_rate) / portfolio_volatility  # Negative for minimization

    # Select objective function based on chosen objective
    if objective == "maximize_sharpe":
        objective_function = sharpe_ratio
        logger.info("Objective: Maximize Sharpe ratio.")
    else:
        objective_function = portfolio_risk
        logger.info("Objective: Minimize risk (standard deviation).")

    # Perform optimization
    result = minimize(objective_function, weights, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        optimized_weights = result.x
        # Round weights to avoid small numerical errors
        optimized_weights = np.round(optimized_weights, decimals=10)
        # Ensure weights sum to 1 by normalizing if necessary
        optimized_weights /= np.sum(optimized_weights)

        logger.info("Portfolio optimization successful.")
        logger.debug(f"Optimized weights: {optimized_weights}")
        logger.debug(f"Sum of weights: {np.sum(optimized_weights)}")
        return optimized_weights
    else:
        logger.error(f"Portfolio optimization failed: {result.message}")
        raise ValueError("Optimization failed:", result.message)
