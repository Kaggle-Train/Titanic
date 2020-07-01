from sklearn.base import BaseEstimator, TransformerMixin


class MotherChildProcessor:
    def __init__(self, add_last_name=False, add_mother_child_relationship=True):
        self.add_last_name = add_last_name
        self.add_mother_child_relationshop = add_mother_child_relationship

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if (self.add_mother_child_relationshop):
            X['LastName'] = [i[0:i.find(',')] for i in X.Name]
            age_margin = 11
            X['MotherChildRelation'] = 0
            for index, row in X.iterrows():
                if (row.Age < age_margin):
                    for index2, row2 in X.iterrows():
                        if (row.LastName == row2.LastName):
                            if (self.is_row_mother(row2)):
                                if (row2.Survived == 1):
                                    X.loc[index, 'MotherChildRelation'] = 1
                                else:
                                    X.loc[index, 'MotherChildRelation'] = -1
                                if (row.Survived == 1):
                                    X.loc[index2, 'MotherChildRelation'] = 1
                                else:
                                    X.loc[index2, 'MotherChildRelation'] = -1
            if (not self.add_last_name):
                X = X.drop('LastName', axis=1)
        return X

    def is_row_mother(self, row):
        if (row.Sex == 'female' and row.Age > 20):
            return True
        return False