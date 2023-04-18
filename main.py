import pandas as pd  # module in python that works on tables- datasets

from sklearn import preprocessing

# file_path = '.\\csv\\student_data.csv'
# data = pd.read_csv(file_path)# print(train.head())  # show us the first 5 lines
# train=train.set_index('Id') #set Id index
# train.info() #gives ius information about the dataset

            #EXPLORATORY DATA ANALYSIS#

# msno.bar(train, figsize = (16,5),color = "#FFE4E1")  #show us the datasets features - gets the dataset, size and color
# plt.show()

# print(train) #shows us first 5 lines and last 5 lines

# print(train.describe()) #shows us the statistics of the dataset

# columns=train.columns
# sns.set()  #sns- Seaborn function
# sns.pairplot(train[columns],height = 5 ,kind ='scatter',diag_kind='kde') #make pairplot as defined
# plt.show()  #now we can see the information - scatter

# fig = go.Figure(data=[go.Pie(labels=train['quality'].value_counts().index, values=train['quality'].value_counts(), hole=.3)]) #figure() Function of value counts- quality
# fig.update_layout(legend_title_text='Quality')
# fig.show()

# import warnings #provided to warn the developer of situations that aren't necessarily exceptions
#
# warnings.filterwarnings('ignore') #ignore warnings
# fig, ax = plt.subplots(12, 3, figsize=(30, 90)) #three arguments that describes the layout of the figure.
# for index, i in enumerate(train.columns):
#     sns.distplot(train[i], ax=ax[index, 0], color='green')
#     sns.boxplot(train[i], ax=ax[index, 1], color='yellow')
#     stats.probplot(train[i], plot=ax[index, 2])
#
# fig.tight_layout()  #used to automatically adjust subplot parameters (basic arguments) to give specified padding.
# fig.subplots_adjust(top=0.95) # used to adjust or refine the subplot structure or design
# plt.suptitle("Visualizing continuous columns", fontsize=50)
# plt.show()

            #STATISTICAL TESTS#

# for column_name in train.columns: #showing how many unique values belong to feature
#     unique_values = len(train[column_name].unique())
#     print("Feature '{column_name}' has '{unique_values}' unique values".format(column_name = column_name,   #.format()- formats the specified value(s) and insert them inside the string's placeholder.
#                                                                                          unique_values=unique_values))

    # import scipy.stats
    #
    # # u : Mann-Whitney test statistic
    # # p : p-value
    # for feature in train.columns:
    #     u, p = scipy.stats.mannwhitneyu(train['quality'], train[feature]) #mannwhiteyu test -used to analyze the difference between two independent samples of ordinal data.
    #     print('With', feature)
    #     print('Mann-Whitney test statistic:', u)
    #     print('p-value:', p)
    #     print('--------------------')

# correlation = train.corr()  #corr() is used to find the pairwise correlation of all columns in the Pandas Dataframe in Python. Any NaN values are automatically excluded.
# print(correlation['quality'].sort_values(ascending = False),'\n')
#
# k= 10
# cols = correlation.nlargest(k,'quality')['quality'].index  #nlargest - returns a specified number of rows after sorting by the highest value for a specified column.
# print(cols)
# cm = np.corrcoef(train[cols].values.T)  #np.corrceof-  allows to compute correlation coefficients of >2 data sets
# f , ax = plt.subplots(figsize = (14,12))  #subplots method provides a way to plot multiple plots on a single figure. Given the number of rows and columns , it returns a tuple ( fig , ax ), giving a single figure fig with an array of axes ax .
#
# sns.heatmap(cm, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',  #create the colorful map
#             linecolor="white",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)


            #FEATURE ENGENEERING AND MACHINE LEARNING#

# from imblearn.over_sampling import SMOTE  #SMOTE is one of the most commonly used oversampling methods to solve the imbalance problem. It aims to balance class distribution by randomly increasing minority class examples by replicating them
# oversample = SMOTE() #. The simplest approach involves duplicating examples in the minority class, although these examples don’t add any new information to the model. Instead, new examples can be synthesized from the existing examples
 #features, labels=  oversample.fit_resample(train.drop(["quality"],axis=1),train["quality"])
#
scaler = preprocessing.MinMaxScaler()  #Transform features by scaling each feature to a given range.
# names = features.columns
# d = scaler.fit_transform(features)
#
# scaled_df = pd.DataFrame(d, columns=names)
# scaled_df.head()
#
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn import svm
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split, GridSearchCV
# import xgboost as xgb
#
# from sklearn.metrics import precision_score,recall_score
# from sklearn.metrics import f1_score
# X_train, X_test, y_train, y_test=train_test_split(scaled_df, labels,test_size=0.33,random_state=42)
#
# models = [RandomForestClassifier(), KNeighborsClassifier(), SVC(), LogisticRegression(),xgb.XGBClassifier()]
# scores = dict()
#
# for m in models:
#     m.fit(X_train, y_train)
#     y_pred = m.predict(X_test)
#
#     print(f'model: {str(m)}')
#     print(classification_report(y_test,y_pred, zero_division=1))
#     print('-'*30, '\n')

# import optuna
# import xgboost as xgb
# from optuna.samplers import TPESampler
# from sklearn.preprocessing import LabelEncoder, RobustScaler
# from sklearn.model_selection import train_test_split, cross_val_score
# def objective(trial):
#     data, target = scaled_df,labels
#     train_x, valid_x, train_y, valid_y = train_test_split(scaled_df,labels, test_size=0.3)
#
#     param = {
#         "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
#         "criterion": trial.suggest_categorical("criterion", ['gini', 'entropy']),
#         "max_depth": trial.suggest_int("max_depth", 2, 30),
#         "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
#         "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10)
#     }
#
#     rf = RandomForestClassifier(**param)
#
#     rf.fit(train_x, train_y)
#
#     preds = rf.predict(valid_x)
#     pred_labels = np.rint(preds)
#     accuracy = accuracy_score(valid_y, pred_labels)
#     return accuracy
#
#
# if __name__ == "__main__":
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=100, timeout=600)
#
#     print("Number of finished trials: {}".format(len(study.trials)))
#
#     print("Best trial:")
#     trial = study.best_trial
#
#     print("  Value: {}".format(trial.value))
#
#     print("  Params: ")
#     for key, value in trial.params.items():
#         print("    {}: {}".format(key, value))


#------------------------------------------------------------------------------------------
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
#
# # Splitting the dataset into training and testing data
# X = train.iloc[:, :-1].values
# y = train.iloc[:, -1].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#
# # Feature scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
#
# # Creating the SVM model
# svm = SVC(kernel='rbf', random_state=0)
#
# # Training the SVM model
# svm.fit(X_train, y_train)
#
# # Predicting the test set results
# y_pred = svm.predict(X_test)
#
# # Calculating the accuracy of the SVM model
# svm_acc = accuracy_score(y_test, y_pred)
# print('SVM accuracy:', svm_acc)
#
# # Creating the Linear Regression model
# lr = LinearRegression()
#
# # Training the Linear Regression model
# lr.fit(X_train, y_train)
#
# # Predicting the test set results
# y_pred = lr.predict(X_test)
#
# # Calculating the accuracy of the Linear Regression model
# lr_acc = lr.score(X_test, y_test)
# print('Linear Regression accuracy:', lr_acc)
#
# # Creating the Random Forest model
# rfc = RandomForestClassifier(n_estimators=100, random_state=0)
#
# # Training the Random Forest model
# rfc.fit(X_train, y_train)
#
# # Predicting the test set results
# y_pred = rfc.predict(X_test)
#
# # Calculating the accuracy of the Random Forest model
# rfc_acc = accuracy_score(y_test, y_pred)
# print('Random Forest accuracy:', rfc_acc)
#
# class ClassificationModel:
#     def init(self, name, model):
#      self.name = name
#      self.model = model
#      self.train_time = None
#      self.prediction_time = None
#      self.train_score = None
#      self.test_score = None
#      self.confusion_matrix = None
#      self.accuracy = None
#      self.sensitivity = None
#      self.specificity = None
#      self.false_positive_rate = None
#      self.precision = None
#      self.fpr, self.tpr, self.thresholds = None, None, None
#      self.auc_score = None
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_regression
import random
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score



file_path = '.\\csv\\WineQT.csv'
data = pd.read_csv(file_path)

# Randomly select a column to use as the target variable
target_variable = random.choice(data.columns)

# Separate the features and target variable
X = data.drop(columns=[target_variable])
y = data[target_variable]

# Randomly select the number of features to select
k = random.randint(1, len(X.columns)/2)

# Select the most important features
selector = SelectKBest(score_func=f_regression, k=k)
X_new = selector.fit_transform(X, y)

# Store the most important feature in an array
important_feature = X.columns[selector.get_support()][0]
important_feature_array = list(X.columns[selector.get_support()])

print("k is: ", k)
print("The target variable is:", target_variable)
print("The most important feature is:", important_feature)
print("Array with the most important feature:", important_feature_array)

# Drop columns that aren't in the important features array
columns_to_drop = set(X.columns) - set(important_feature_array)

X = X.drop(columns=columns_to_drop)


class BinaryClassificationMetrics:
    def __init__(self):
        self._confusion_matrix = None
        self._accuracy = None
        self._sensitivity = None
        self._specificity = None
        self._false_positive_rate = None
        self._precision = None
        self._recall = None
        self._f1_score = None
        self._roc_curve = None

    def set_metrics(self, y_true, y_pred):
        self._confusion_matrix = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = self._confusion_matrix.ravel()
        self._accuracy = (tp + tn) / (tp + tn + fp + fn)
        self._sensitivity = tp / (tp + fn)
        self._specificity = tn / (tn + fp)
        self._false_positive_rate = fp / (fp + tn)
        self._precision = tp / (tp + fp)
        self._recall = recall_score(y_true, y_pred)
        self._f1_score = f1_score(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        self._roc_curve = (fpr, tpr, auc(fpr, tpr))

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


# Generate a binary classification dataset
# X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a list to store the evaluation metrics for each model
metrics_list = []

# # SVM model
# svm = SVC(kernel='linear', probability=True, random_state=42)
# svm.fit(X_train, y_train)
# svm_y_pred = svm.predict(X_test)
# svm_metrics = BinaryClassificationMetrics()
# svm_metrics.set_metrics(y_test, svm_y_pred)
# metrics_list.append(('SVM', svm_metrics))

svm_reg = SVR(kernel='linear')
svm_reg.fit(X_train, y_train)
svm_y_pred = svm_reg.predict(X_test)

mse = mean_squared_error(y_test, svm_y_pred)
r2 = r2_score(y_test, svm_y_pred)

reg_metrics = BinaryClassificationMetrics()
reg_metrics.set_metrics(mse, r2)
metrics_list.append(('SVM Regression', reg_metrics))

# Logistic regression model
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
lr_y_pred = lr.predict(X_test)
lr_metrics = BinaryClassificationMetrics()
lr_metrics.set_metrics(y_test, lr_y_pred)
metrics_list.append(('Logistic Regression', lr_metrics))

# Random forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_y_pred = rf.predict(X_test)
rf_metrics = BinaryClassificationMetrics()
rf_metrics.set_metrics(y_test, rf_y_pred)
metrics_list.append(('Random Forest', rf_metrics))
# Print the evaluation metrics for each model
# Print the evaluation metrics for each model
for model_name, metrics in metrics_list:
    print(f"{model_name} Metrics:")
    print(f"Accuracy: {metrics.get_accuracy()}")
    print(f"Sensitivity: {metrics.get_sensitivity()}")
    print(f"Specificity: {metrics.get_specificity()}")
    print(f"False Positive Rate: {metrics.get_false_positive_rate()}")
    print(f"Precision: {metrics.get_precision()}")
    print(f"Recall: {metrics.get_sensitivity()}")
    print(f"F1 Score: {2 * metrics.get_precision() * metrics.get_sensitivity() / (metrics.get_precision() + metrics.get_sensitivity())}")
    print(f"ROC Curve: {metrics.get_roc_curve()}\n")



#--------------------------------------------------------------------------------------
