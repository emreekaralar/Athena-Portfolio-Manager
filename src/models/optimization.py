# src/models/optimization.py

import numpy as np
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
    - optimized_weights (ndarray): Optimized weights with possible negative values for short positions.
    """
    logger.info("Starting long-short portfolio optimization.")

    num_assets = len(returns.columns)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Initial guess: Equal long and short weights
    weights = np.zeros(num_assets)
    weights[:num_assets // 2] = 1.0 / (num_assets // 2)
    weights[num_assets // 2:] = -1.0 / (num_assets // 2)

    # Constraints:
    # 1. Sum of long weights = long_short_ratio
    # 2. Sum of short weights = -1
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w[w > 0]) - long_short_ratio},
        {'type': 'eq', 'fun': lambda w: np.sum(w[w < 0]) + 1.0}  # Ensures sum of shorts is -1
    ]

    # Bounds: Allow shorting up to 100% of each asset
    bounds = tuple((-1.0, 1.0) for _ in range(num_assets))

    # Define objective functions
    def portfolio_risk(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

    def sharpe_ratio(w):
        portfolio_return = np.dot(w, mean_returns)
        portfolio_volatility = portfolio_risk(w)
        return -(portfolio_return - risk_free_rate) / portfolio_volatility  # Negative for minimization

    if objective == "maximize_sharpe":
        objective_function = sharpe_ratio
        logger.info("Objective: Maximize Sharpe ratio for long-short portfolio.")
    elif objective == "minimize_risk":
        objective_function = portfolio_risk
        logger.info("Objective: Minimize risk for long-short portfolio.")
    else:
        logger.error(f"Unsupported optimization objective: {objective}")
        raise ValueError(f"Unsupported optimization objective: {objective}")

    # Perform optimization
    result = minimize(objective_function, weights, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        optimized_weights = result.x
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
    - risk_aversion (float): Risk aversion coefficient. Higher values imply higher aversion to risk.

    Returns:
    - optimized_weights (ndarray): Optimized asset weights that sum to 1.
    """
    logger.info("Starting Mean-Variance Optimization.")

    num_assets = len(returns.columns)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Define the objective function: risk_aversion * portfolio_variance - portfolio_return
    def objective(w):
        portfolio_return = np.dot(w, mean_returns)
        portfolio_variance = np.dot(w.T, np.dot(cov_matrix, w))
        return risk_aversion * portfolio_variance - portfolio_return

    # Constraints: Sum of weights = 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    # Bounds: Weights between 0 and 1 (no short selling)
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Initial guess: Equal weights
    initial_guess = np.ones(num_assets) / num_assets

    # Perform optimization
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        optimized_weights = result.x
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
    - optimized_weights (ndarray): Optimized asset weights that sum to 1.
    """
    logger.info("Starting Minimum Variance Optimization.")

    num_assets = len(returns.columns)
    cov_matrix = returns.cov()

    # Objective function: Portfolio variance
    def portfolio_variance(w):
        return np.dot(w.T, np.dot(cov_matrix, w))

    # Constraints: Sum of weights = 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    # Bounds: Weights between 0 and 1 (no short selling)
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Initial guess: Equal weights
    initial_guess = np.ones(num_assets) / num_assets

    # Perform optimization
    result = minimize(portfolio_variance, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        optimized_weights = result.x
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
    - optimized_weights (ndarray): Optimized asset weights that sum to 1.
    """
    logger.info("Starting Maximum Sharpe Ratio Optimization.")

    num_assets = len(returns.columns)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    def negative_sharpe_ratio(w):
        portfolio_return = np.dot(w, mean_returns)
        portfolio_std = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        return -(portfolio_return - risk_free_rate) / portfolio_std

    # Constraints: Sum of weights = 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    # Bounds: Weights between 0 and 1 (no short selling)
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Initial guess: Equal weights
    initial_guess = np.ones(num_assets) / num_assets

    # Perform optimization
    result = minimize(negative_sharpe_ratio, initial_guess,
                      method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        optimized_weights = result.x
        logger.info("Maximum Sharpe Ratio Optimization successful.")
        logger.debug(f"Optimized weights: {optimized_weights}")
        return optimized_weights
    else:
        logger.error(f"Maximum Sharpe Ratio Optimization failed: {result.message}")
        raise ValueError("Maximum Sharpe Ratio Optimization failed:", result.message)


def optimize_portfolio_risk_parity(returns):
    """
    Optimizes the portfolio to achieve Risk Parity, where each asset contributes equally to the total portfolio risk.

    Parameters:
    - returns (DataFrame): Asset returns.

    Returns:
    - optimized_weights (ndarray): Optimized asset weights that sum to 1.
    """
    logger.info("Starting Risk Parity Optimization.")

    num_assets = len(returns.columns)
    cov_matrix = returns.cov()
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    # Initial guess: Equal weights
    initial_guess = np.ones(num_assets) / num_assets

    # Constraints: Sum of weights = 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    # Bounds: Weights between 0 and 1 (no short selling)
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Define the objective function for Risk Parity
    def risk_parity_objective(w):
        # Portfolio variance
        portfolio_variance = np.dot(w.T, np.dot(cov_matrix, w))
        portfolio_std = np.sqrt(portfolio_variance)

        # Marginal contribution to risk
        mcr = np.dot(cov_matrix, w) / portfolio_std

        # Risk contributions
        rc = w * mcr

        # Target risk contribution: Equal for all assets
        target_rc = portfolio_std / num_assets

        # Objective: Sum of squared differences between actual and target risk contributions
        return np.sum((rc - target_rc) ** 2)

    # Perform optimization
    result = minimize(risk_parity_objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        optimized_weights = result.x
        logger.info("Risk Parity Optimization successful.")
        logger.debug(f"Optimized weights: {optimized_weights}")
        return optimized_weights
    else:
        logger.error(f"Risk Parity Optimization failed: {result.message}")
        raise ValueError("Risk Parity Optimization failed:", result.message)


def optimize_portfolio_multiple_methods(returns, risk_free_rate=0.04, method="mean_variance"):
    """
    Optimizes the portfolio using different optimization methods.

    Parameters:
    - returns (DataFrame): Asset returns.
    - risk_free_rate (float): Risk-free rate for Sharpe ratio calculations.
    - method (str): Optimization method to use. Options: "mean_variance", "minimum_variance", "maximum_sharpe", "risk_parity".

    Returns:
    - optimized_weights (ndarray): Optimized asset weights that sum to 1.
    """
    logger.info(f"Starting portfolio optimization using method: {method}")

    if method == "mean_variance":
        # You can adjust the risk_aversion parameter as needed
        risk_aversion = 1.0
        return optimize_portfolio_mean_variance(returns, risk_free_rate, risk_aversion)
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
    General optimization function that selects the optimization method based on the 'method' parameter.

    Parameters:
    - returns (DataFrame): Asset returns.
    - risk_free_rate (float): Risk-free rate.
    - method (str): Optimization method; options: "mean_variance", "minimum_variance", "maximum_sharpe", "risk_parity", "long_short".
    - objective (str): Optimization objective; used only for "long_short" method; options: "maximize_sharpe", "minimize_risk".
    - long_short_ratio (float): Ratio of total long positions to total short positions; used only for "long_short" method.

    Returns:
    - optimized_weights (ndarray): Optimized asset weights.
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
