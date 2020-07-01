from sklearn.base import BaseEstimator, TransformerMixin
from src.d03_processing.title_processor import TitleProcessor


class AgePurger(BaseEstimator, TransformerMixin):
    def __init__(self, fill_age=True):
        self.fill_age = fill_age

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if (self.fill_age):
            title_processor = TitleProcessor(add_title=True, add_is_rev=False)
            X = title_processor.transform(X)
            X['Age'] = X['Age'].fillna(X.groupby('Title')['Age'].transform('mean'))
            X = X.drop(['Title'], axis=1)  # remove title since it should be treated as hyperparameter from processor
        return X
