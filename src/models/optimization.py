# src/models/optimization.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.utils.logger import get_logger

logger = get_logger(__name__)

def optimize_portfolio_long_short(returns, risk_free_rate=0.04, objective="maximize_sharpe", long_short_ratio=1.0):
    """
    Optimizes a long-short portfolio.

    Parameters:
    - returns (DataFrame): Asset returns.
    - risk_free_rate (float): Risk-free rate.
    - objective (str): Optimization objective; options: "maximize_sharpe", "minimize_risk".
    - long_short_ratio (float): Ratio of total long positions to total short positions.

    Returns:
    - optimized_weights (Series): Optimized weights with possible negative values for short positions.
    """
    logger.info("Starting long-short portfolio optimization.")

    num_assets = len(returns.columns)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Initial guess: Equal long and short weights
    num_long_positions = int(num_assets * (long_short_ratio / (2 * long_short_ratio)))
    num_short_positions = num_assets - num_long_positions

    weights = np.zeros(num_assets)
    weights[:num_long_positions] = long_short_ratio / num_long_positions
    weights[num_long_positions:] = -long_short_ratio / num_short_positions

    # Constraints:
    # 1. Sum of long weights = long_short_ratio
    # 2. Sum of short weights = -1.0
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w)},
        {'type': 'eq', 'fun': lambda w: np.sum(w[w > 0]) - long_short_ratio},
        {'type': 'eq', 'fun': lambda w: -np.sum(w[w < 0]) - long_short_ratio}
    ]

    # Bounds: Allow shorting up to 100% of each asset
    bounds = tuple((-1.0, 1.0) for _ in range(num_assets))

    # Define objective functions
    def portfolio_risk(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

    def negative_sharpe_ratio(w):
        portfolio_return = np.dot(w, mean_returns)
        portfolio_volatility = portfolio_risk(w)
        return -(portfolio_return - risk_free_rate) / portfolio_volatility  # Negative for minimization

    if objective == "maximize_sharpe":
        objective_function = negative_sharpe_ratio
        logger.info("Objective: Maximize Sharpe ratio for long-short portfolio.")
    elif objective == "minimize_risk":
        objective_function = portfolio_risk
        logger.info("Objective: Minimize risk for long-short portfolio.")
    else:
        logger.error(f"Unsupported optimization objective: {objective}")
        raise ValueError(f"Unsupported optimization objective: {objective}")

    # Perform optimization
    result = minimize(
        objective_function,
        weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )

    if result.success:
        optimized_weights = pd.Series(result.x, index=returns.columns)
        logger.info("Long-short portfolio optimization successful.")
        logger.debug(f"Optimized weights: {optimized_weights}")
        return optimized_weights
    else:
        logger.error(f"Long-short portfolio optimization failed: {result.message}")
        raise ValueError("Long-short portfolio optimization failed:", result.message)

def optimize_portfolio_mean_variance(returns, risk_free_rate=0.04, risk_aversion=1.0):
    """
    Optimizes the portfolio using Mean-Variance Optimization (MVO).

    Parameters:
    - returns (DataFrame): Asset returns.
    - risk_free_rate (float): Risk-free rate for expected return calculation.
    - risk_aversion (float): Risk aversion coefficient.

    Returns:
    - optimized_weights (Series): Optimized asset weights that sum to 1.
    """
    logger.info("Starting Mean-Variance Optimization.")

    num_assets = len(returns.columns)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Define the objective function
    def objective(w):
        portfolio_return = np.dot(w, mean_returns)
        portfolio_variance = np.dot(w.T, np.dot(cov_matrix, w))
        return risk_aversion * portfolio_variance - portfolio_return

    # Constraints: Sum of weights = 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    # Bounds: Weights between 0 and 1 (long-only portfolio)
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Initial guess: Equal weights
    initial_guess = np.ones(num_assets) / num_assets

    # Perform optimization
    result = minimize(
        objective,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if result.success:
        optimized_weights = pd.Series(result.x, index=returns.columns)
        logger.info("Mean-Variance Optimization successful.")
        logger.debug(f"Optimized weights: {optimized_weights}")
        return optimized_weights
    else:
        logger.error(f"Mean-Variance Optimization failed: {result.message}")
        raise ValueError("Mean-Variance Optimization failed:", result.message)

def optimize_portfolio_min_variance(returns):
    """
    Optimizes the portfolio to minimize portfolio variance.

    Parameters:
    - returns (DataFrame): Asset returns.

    Returns:
    - optimized_weights (Series): Optimized asset weights that sum to 1.
    """
    logger.info("Starting Minimum Variance Optimization.")

    num_assets = len(returns.columns)
    cov_matrix = returns.cov()

    # Objective function: Portfolio variance
    def portfolio_variance(w):
        return np.dot(w.T, np.dot(cov_matrix, w))

    # Constraints: Sum of weights = 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    # Bounds: Weights between 0 and 1
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Initial guess: Equal weights
    initial_guess = np.ones(num_assets) / num_assets

    # Perform optimization
    result = minimize(
        portfolio_variance,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if result.success:
        optimized_weights = pd.Series(result.x, index=returns.columns)
        logger.info("Minimum Variance Optimization successful.")
        logger.debug(f"Optimized weights: {optimized_weights}")
        return optimized_weights
    else:
        logger.error(f"Minimum Variance Optimization failed: {result.message}")
        raise ValueError("Minimum Variance Optimization failed:", result.message)

def optimize_portfolio_max_sharpe(returns, risk_free_rate=0.04):
    """
    Optimizes the portfolio to maximize the Sharpe ratio.

    Parameters:
    - returns (DataFrame): Asset returns.
    - risk_free_rate (float): Risk-free rate for Sharpe ratio calculation.

    Returns:
    - optimized_weights (Series): Optimized asset weights that sum to 1.
    """
    logger.info("Starting Maximum Sharpe Ratio Optimization.")

    num_assets = len(returns.columns)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    def negative_sharpe_ratio(w):
        portfolio_return = np.dot(w, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        return -(portfolio_return - risk_free_rate) / portfolio_volatility

    # Constraints: Sum of weights = 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    # Bounds: Weights between 0 and 1
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Initial guess
    initial_guess = np.ones(num_assets) / num_assets

    # Perform optimization
    result = minimize(
        negative_sharpe_ratio,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if result.success:
        optimized_weights = pd.Series(result.x, index=returns.columns)
        logger.info("Maximum Sharpe Ratio Optimization successful.")
        logger.debug(f"Optimized weights: {optimized_weights}")
        return optimized_weights
    else:
        logger.error(f"Maximum Sharpe Ratio Optimization failed: {result.message}")
        raise ValueError("Maximum Sharpe Ratio Optimization failed:", result.message)

def optimize_portfolio_risk_parity(returns):
    """
    Optimizes the portfolio to achieve risk parity.

    Parameters:
    - returns (DataFrame): Asset returns.

    Returns:
    - optimized_weights (Series): Optimized asset weights that sum to 1.
    """
    logger.info("Starting Risk Parity Optimization.")

    num_assets = len(returns.columns)
    cov_matrix = returns.cov()

    # Initial guess: Equal weights
    initial_guess = np.ones(num_assets) / num_assets

    # Constraints: Sum of weights = 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    # Bounds: Weights between 0 and 1
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Define the risk parity objective function
    def risk_parity_objective(w):
        portfolio_variance = np.dot(w.T, np.dot(cov_matrix, w))
        portfolio_std = np.sqrt(portfolio_variance)
        marginal_risk_contrib = np.dot(cov_matrix, w)
        risk_contrib = w * marginal_risk_contrib
        risk_contrib_fraction = risk_contrib / portfolio_std
        # Target is equal risk contribution
        target_contrib = portfolio_std / num_assets
        return np.sum((risk_contrib_fraction - target_contrib) ** 2)

    # Perform optimization
    result = minimize(
        risk_parity_objective,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if result.success:
        optimized_weights = pd.Series(result.x, index=returns.columns)
        logger.info("Risk Parity Optimization successful.")
        logger.debug(f"Optimized weights: {optimized_weights}")
        return optimized_weights
    else:
        logger.error(f"Risk Parity Optimization failed: {result.message}")
        raise ValueError("Risk Parity Optimization failed:", result.message)

def optimize_portfolio_multiple_methods(returns, risk_free_rate=0.04, method="mean_variance"):
    """
    Optimizes the portfolio using different methods.

    Parameters:
    - returns (DataFrame): Asset returns.
    - risk_free_rate (float): Risk-free rate.
    - method (str): Optimization method.

    Returns:
    - optimized_weights (Series): Optimized weights.
    """
    logger.info(f"Starting portfolio optimization using method: {method}")

    if method == "mean_variance":
        return optimize_portfolio_mean_variance(returns, risk_free_rate)
    elif method == "minimum_variance":
        return optimize_portfolio_min_variance(returns)
    elif method == "maximum_sharpe":
        return optimize_portfolio_max_sharpe(returns, risk_free_rate)
    elif method == "risk_parity":
        return optimize_portfolio_risk_parity(returns)
    else:
        logger.error(f"Unsupported optimization method: {method}")
        raise ValueError(f"Unsupported optimization method: {method}")

def optimize_portfolio(returns, risk_free_rate=0.04, method="mean_variance", objective="maximize_sharpe", long_short_ratio=1.0):
    """
    General optimization function.

    Parameters:
    - returns (DataFrame): Asset returns.
    - risk_free_rate (float): Risk-free rate.
    - method (str): Optimization method.
    - objective (str): Objective for long-short optimization.
    - long_short_ratio (float): Ratio for long-short optimization.

    Returns:
    - optimized_weights (Series): Optimized weights.
    """
    if method == "long_short":
        return optimize_portfolio_long_short(
            returns,
            risk_free_rate=risk_free_rate,
            objective=objective,
            long_short_ratio=long_short_ratio
        )
    else:
        return optimize_portfolio_multiple_methods(
            returns,
            risk_free_rate=risk_free_rate,
            method=method
        )
