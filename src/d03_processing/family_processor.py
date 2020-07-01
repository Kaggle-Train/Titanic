from sklearn.base import BaseEstimator, TransformerMixin


class FamilyProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, add_family_size=True):
        self.add_family_size = add_family_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if (self.add_family_size):
            X['FamilySize'] = X.SibSp + X.Parch
        return X