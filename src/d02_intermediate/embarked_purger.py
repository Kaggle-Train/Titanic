from sklearn.base import BaseEstimator, TransformerMixin


class EmbarkedPurger(BaseEstimator, TransformerMixin):
    def __init__(self, fill_embarked=True):
        self.fill_embarked = fill_embarked

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if (self.fill_embarked):
            X['Embarked'] = X['Embarked'].fillna('S')
        return X
