from src.sentiment.nlp_models import analyze_sentiment
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    tokens = [t for t in tokens if t not in string.punctuation]
    return ' '.join(tokens)

def get_sentiment_score(text):
    clean_text = preprocess_text(text)
    return analyze_sentiment(clean_text)
