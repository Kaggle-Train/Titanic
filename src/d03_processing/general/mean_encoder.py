from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


# class for mean encoding
class MeanEncoder:
    def __init__(self, add_feature=True, new_feature_name='NewFeatureName', mean_feature_name='MeanFeatureName',
                 target_name='TargetLabel', data_type='train', data_eval=np.NAN):
        self.add_feature = add_feature
        self.new_feature_name = new_feature_name
        self.mean_feature_name = mean_feature_name
        self.target_name = target_name
        self.data_type = data_type
        self.X_eval = data_eval

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if (self.add_feature):
            if (self.data_type == 'train'):
                X = self.transform_train(X)
            if (self.data_type == 'eval'):
                X = self.transform_eval(X)
            if (self.data_type == 'test'):
                X = self.transform_test(X)
        return X

    def transform_train(self, X):
        X[self.new_feature_name] = 0
        for index, row in X.iterrows():
            mean_feature = row[self.mean_feature_name]
            target_array = []
            for index2, row2 in X.iterrows():
                if (index != index2 and mean_feature == row2[self.mean_feature_name]):
                    target_array.append(row2[self.target_name])
            if (len(target_array) > 0):
                X.loc[index, self.new_feature_name] = sum(target_array) / len(target_array)
            else:
                X.loc[index, self.new_feature_name] = X[self.target_name].mean()

        return X

    def transform_eval(self, X):
        X[self.new_feature_name] = np.NAN
        # data point also includes it's own label for calculating ticket probability
        X[self.new_feature_name] = X[self.new_feature_name].fillna(
            X.groupby(self.mean_feature_name)[self.target_name].transform('mean'))

        return X

    def transform_test(self, X):
        X[self.new_feature_name] = self.X_eval[self.target_name].mean()
        for index, row in X.iterrows():
            for index2, row2 in self.X_eval.iterrows():
                if (row[self.mean_feature_name] == row2[self.mean_feature_name]):
                    X.loc[index, self.new_feature_name] = row2[self.new_feature_name]
                    break
        return X
