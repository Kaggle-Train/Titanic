from sklearn.base import BaseEstimator, TransformerMixin


class TicketProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, add_ticket_propability=True):
        self.add_ticket_propability = add_ticket_propability

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if (self.add_ticket_propability):
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