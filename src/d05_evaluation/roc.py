from sklearn.metrics import roc_curve, roc_auc_score


class ROC:
    def show(self, y, y_pred_proba, axes):
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        auc = roc_auc_score(y, y_pred_proba)
        axes.set_title('ROC')
        axes.plot(fpr, tpr, label="auc=%0.2f" % auc)
        axes.legend(loc=4)
