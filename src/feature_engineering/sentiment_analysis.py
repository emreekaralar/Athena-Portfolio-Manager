from src.sentiment.sentiment_pipeline import get_sentiment_score

def add_sentiment_features(df, news_data):
    df['sentiment_score'] = news_data.apply(get_sentiment_score)
    return df
