from sklearn.base import BaseEstimator, TransformerMixin


class CabinPurger(BaseEstimator, TransformerMixin):
    def __init__(self, encode_cabin=True):
        self.encode_cabin = encode_cabin

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if (self.encode_cabin):
            X.loc[X.Cabin.isna(), 'Cabin'] = 0
            X.loc[X.Cabin != 0, 'Cabin'] = 1
            return X