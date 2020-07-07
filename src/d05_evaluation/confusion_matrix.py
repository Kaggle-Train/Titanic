from sklearn.metrics import plot_confusion_matrix


class ConfusionMatrix:
    def show(self, model, X, y, axes):
        disp = plot_confusion_matrix(model, X, y, normalize='true', cmap='Blues', ax=axes)
        disp.ax_.set_title('Confusion Matrix')
