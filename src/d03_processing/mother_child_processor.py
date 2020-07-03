from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class MotherChildProcessor:
    def __init__(self, add_last_name=False, add_mother_child_relationship=True, data_type='train', data_eval=np.NAN):
        self.age_margin = 11
        self.add_last_name = add_last_name
        self.add_mother_child_relationship = add_mother_child_relationship
        self.data_type = data_type
        self.X_eval = data_eval

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if (self.add_mother_child_relationship):
            if(self.data_type == 'train'):
                X = self.transform_train(X)
            if(self.data_type == 'eval'):
                X = self.transform_eval(X)
            if(self.data_type == 'test'):
                X = self.transform_test(X)
        return X
    
    def transform_train(self, X):
        X['LastName'] = [i[0:i.find(',')] for i in X.Name]
        X['MotherChildRelation'] = 0
        for index, row in X.iterrows():
            if (row.Age < self.age_margin):
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
    
    def transform_eval(self, X):
        return self.transform_train(X)
    
    def transform_test(self, X):
        X['LastName'] = [i[0:i.find(',')] for i in X.Name]
        self.X_eval['LastName'] = [i[0:i.find(',')] for i in self.X_eval.Name]
        X['MotherChildRelation'] = 0
        
        for index, row in X.iterrows():
            for index2, row2 in self.X_eval.iterrows():
                if (row.LastName == row2.LastName):
                    if (self.is_row_mother(row) or (row.Age < self.age_margin and self.is_row_mother(row2))):
                        if (row2.Survived == 1):
                            X.loc[index, 'MotherChildRelation'] = 1
                        else:
                            X.loc[index, 'MotherChildRelation'] = -1
        if (not self.add_last_name):
            X = X.drop('LastName', axis=1)
        return X
    
    def is_row_mother(self, row):
        if (row.Sex == 'female' and row.Age > 20):
            return True
        return False