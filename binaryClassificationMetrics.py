from sklearn.metrics import confusion_matrix, recall_score, f1_score, roc_curve, auc, precision_score


class BinaryClassificationMetrics:
    def __init__(self):
        self.specificity = None
        self._confusion_matrix = None
        self._accuracy = None
        self._sensitivity = None
        self._specificity = None
        self._false_positive_rate = None
        self._precision = None
        self._recall = None
        self._f1_score = None
        self._roc_curve = None

    def calculate_metrics(self, y_true, y_pred):
        self._confusion_matrix = confusion_matrix(y_true, y_pred)
      #  tn, fp, fn, tp = self._confusion_matrix.ravel()
        self._accuracy = precision_score(y_true, y_pred, average='macro')
        # self._sensitivity = tp / (tp + fn)
        # self._specificity = tn / (tn + fp)
        # self._false_positive_rate = fp / (fp + tn)
        self._precision = precision_score(y_true, y_pred, average='macro')
        self._recall = recall_score(y_true, y_pred, average='macro')
        self._f1_score = f1_score(y_true, y_pred, average='macro')
        # fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
        # self._roc_curve = (fpr, tpr, auc(fpr, tpr))

    def get_confusion_matrix(self):
        return self._confusion_matrix

    def get_accuracy(self):
        return self._accuracy

    def get_sensitivity(self):
        return self._sensitivity

    def get_specificity(self):
        return self._specificity

    def get_false_positive_rate(self):
        return self._false_positive_rate

    def get_precision(self):
        return self._precision

    def get_recall(self):
        return self._recall

    def get_f1_score(self):
        return self._f1_score

    def get_roc_curve(self):
        return self._roc_curve

    def print_metrics_with_model_name(self, model_name):
        print(model_name, 'model')
        print(model_name, 'model confusion matrix:')
        print(self.get_confusion_matrix())
        print(model_name, 'model accuracy score -->', self.get_accuracy())
        print(model_name, 'model recall score -->', self.get_recall())
        print(model_name, 'model f1 score -->', self.get_f1_score())


        print('|------------------------------------------------------------------------------------|')
