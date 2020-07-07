from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from src.d05_evaluation.confusion_matrix import ConfusionMatrix
from src.d05_evaluation.roc import ROC


class EvaluationSummary:
    def __init__(self, classifier, X, y):
        self.K = 5  # cross validation size
        self.classifier = classifier
        self.X = X
        self.y = y
        model_fit = self.classifier.fit(self.X, self.y)
        self.y_pred = model_fit.predict(X)
        self.y_pred_proba = model_fit.predict_proba(X)[::, 1]

        self.train_accuracy = model_fit.score(self.X, self.y)
        self.scores = cross_val_score(self.classifier, self.X, self.y, cv=self.K)

    def show_summary(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(type(self.classifier).__name__)

        # ROC curve
        roc = ROC()
        roc.show(self.y, self.y_pred_proba, ax1)

        # Confusion matrix
        cfm = ConfusionMatrix()
        cfm.show(self.classifier, self.X, self.y, ax2)
        plt.show()

        print('------------- BEGIN EVALUATION SUMMARY ({}) -------------'.format(type(self.classifier).__name__))

        print('Train accuracy: ' + str(self.train_accuracy))
        print('---------------------------------------------')
        print(classification_report(self.y, self.y_pred))
        print('---------------------------------------------')
        print('Confusion matrix')
        print(confusion_matrix(self.y, self.y_pred))
        print('---------------------------------------------')

        # Cross value scores
        print(self.K, "Cross accuracy validation: %0.2f (+/- %0.2f)" % (self.scores.mean(), self.scores.std() * 2))

        print('------------- END EVALUATION SUMMARY ({}) -------------'.format(type(self.classifier).__name__))
