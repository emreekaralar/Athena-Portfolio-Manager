from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

def apply_pca(df, n_components=10):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df)
    return principal_components

def select_best_features(X, y, k=20):
    selector = SelectKBest(score_func=f_regression, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new, selector
