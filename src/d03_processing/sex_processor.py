import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SexProcessor:
    def __init__(self, encode_sex=True):
        self.encode_sex = encode_sex

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if (self.encode_sex):
            sex_dum_df = pd.get_dummies(X[['Sex']])
            # X = X.drop('Sex', axis=1)
            return pd.concat([X, sex_dum_df], axis=1)
        return X