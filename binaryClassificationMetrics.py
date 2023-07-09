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
        if sum(y_true) == 0:
            # Handle the case when there are no positive samples
            self.specificity = None
            self._false_positive_rate = None
            self._roc_curve = None
            self._confusion_matrix = [[0, 0], [0, 0]]
            self._precision = 0
            self._recall = 0
            self._f1_score = 0
            self._accuracy = 0
            self._sensitivity = 0
        else:
            self._confusion_matrix = confusion_matrix(y_true, y_pred)
            self._precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            self._recall = recall_score(y_true, y_pred, average='macro')
            self._f1_score = f1_score(y_true, y_pred, average='macro')
            self._accuracy = (self._confusion_matrix[0][0] + self._confusion_matrix[1][1]) / (self._confusion_matrix[0][0] + self._confusion_matrix[0][1] + self._confusion_matrix[1][0] + self._confusion_matrix[1][1])
            self._sensitivity = self._confusion_matrix[1][1] / (self._confusion_matrix[1][1] + self._confusion_matrix[1][0])
            self.specificity = self._confusion_matrix[0][0] / (self._confusion_matrix[0][0] + self._confusion_matrix[0][1])
            self._false_positive_rate = self._confusion_matrix[0][1] / (self._confusion_matrix[0][1] + self._confusion_matrix[0][0])
            self._roc_curve = roc_curve(y_true, y_pred, pos_label=1)


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
        print(model_name, 'model recall score -->', self.get_recall())
        print(model_name, 'model f1 score -->', self.get_f1_score())
        print(model_name, 'model precision score -->', self.get_precision())

        print(model_name, 'model accuracy score -->', self.get_accuracy())
        print(model_name, 'model sensitivity score -->', self.get_sensitivity())
        print(model_name, 'model specificity score -->', self.get_specificity())
        print(model_name, 'model false positive rate score -->', self.get_false_positive_rate())
        print(model_name, 'model roc curve -->', self.get_roc_curve())


        print('|------------------------------------------------------------------------------------|')
