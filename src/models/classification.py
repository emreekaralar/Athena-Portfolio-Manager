from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_random_forest(X,y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X,y)
    return model

def train_xgboost(X,y):
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    return model

def evaulate_classification_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    print(report)
