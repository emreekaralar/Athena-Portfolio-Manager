# src/feature_engineering/scoring.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils.logger import get_logger

logger = get_logger(__name__)

def calculate_composite_scores(financial_data, scoring_config):
    """
    Calculates composite scores for each stock based on selected metrics and weights.

    Parameters:
    - financial_data (DataFrame): DataFrame containing financial metrics for each stock.
    - scoring_config (dict): Configuration for scoring, including selected metrics and weights.

    Returns:
    - scores_df (DataFrame): DataFrame containing Ticker and Composite Score.
    """
    logger.info("Calculating composite scores for stocks.")

    # Extract the selected metrics
    metrics = scoring_config['metrics']
    weights = scoring_config['weights']

    # Ensure metrics and weights have the same length
    if len(metrics) != len(weights):
        logger.error("Number of metrics and weights must be the same.")
        return None

    # Filter the financial data to include only the selected metrics
    data = financial_data[['Ticker'] + metrics].dropna()
    tickers = data['Ticker']
    data_metrics = data[metrics]

    # Standardize the metrics
    scaler = StandardScaler()
    standardized_metrics = scaler.fit_transform(data_metrics)

    # Create a DataFrame of standardized metrics
    standardized_df = pd.DataFrame(standardized_metrics, columns=metrics)

    # Calculate the composite score
    standardized_df['Composite Score'] = standardized_df.mul(weights).sum(axis=1)

    # Add the Ticker back to the DataFrame
    standardized_df['Ticker'] = tickers.values

    # Return the DataFrame with Ticker and Composite Score
    scores_df = standardized_df[['Ticker', 'Composite Score']]

    logger.info("Composite scores calculated.")
    return scores_df
