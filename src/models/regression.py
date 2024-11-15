from sklearn_linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

def train_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def train_gradient_boosting(X, y):
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return model

def evaluate_regression_model(model, X_test, y_test):
    predictons = model.predict(X_test)
    mse = mean_squared_error(y_test, predictons)
    print(f"Mean Squared Error: {mse}")

