from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class TicketProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, add_ticket_propability=True, data_type='train', data_eval=np.NAN):
        self.add_ticket_propability = add_ticket_propability
        self.data_type = data_type
        self.X_eval = data_eval

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if (self.add_ticket_propability):
            if (self.data_type == 'train'):
                X = self.transform_train(X)
            if (self.data_type == 'eval'):
                X = self.transform_eval(X)
            if (self.data_type == 'test'):
                X = self.transform_test(X)
        return X

    def transform_train(self, X):
        X['TicketProbability'] = 0
        for index, row in X.iterrows():
            ticket = row.Ticket
            survived_array = []
            for index2, row2 in X.iterrows():
                if (index != index2 and row.Ticket == row2.Ticket):
                    survived_array.append(row2.Survived)
            if (len(survived_array) > 0):
                X.loc[index, 'TicketProbability'] = sum(survived_array) / len(survived_array)
            else:
                X.loc[index, 'TicketProbability'] = X.Survived.mean()

        return X

    def transform_eval(self, X):
        X['TicketProbability'] = np.NAN
        # data point also includes it's own label for calculating ticket probability
        X['TicketProbability'] = X['TicketProbability'].fillna(X.groupby('Ticket')['Survived'].transform('mean'))

        return X

    def transform_test(self, X):
        X['TicketProbability'] = self.X_eval.Survived.mean()
        for index, row in X.iterrows():
            for index2, row2 in self.X_eval.iterrows():
                if (row.Ticket == row2.Ticket):
                    X.loc[index, 'TicketProbability'] = row2.TicketProbability
                    break
        return X
