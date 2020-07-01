from sklearn.base import BaseEstimator, TransformerMixin

# Processor is required in AgePurger
class TitleProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, add_title=True, add_is_rev=True):
        self.add_title = add_title
        self.add_is_rev = add_is_rev

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if (self.add_title):

            X['Title'] = [i[i.find(',') + 1: i.find('.')].strip() for i in X.Name]
            if (self.add_is_rev):
                X['IsRev'] = 0
                X.loc[X.Title == 'Rev', 'IsRev'] = 1
        return X
