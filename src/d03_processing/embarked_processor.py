import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class EmbarkedProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, encode_embarked=True):
        self.encode_embarked = encode_embarked

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if (self.encode_embarked):
            X['Embarked'] = X['Embarked'].fillna('S')
            embarked_dum_df = pd.get_dummies(X[['Embarked']])
            X = X.drop('Embarked', axis=1)
            return pd.concat([X, embarked_dum_df], axis=1)
        return X