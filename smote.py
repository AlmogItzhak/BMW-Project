import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_regression
import random
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

from binaryClassificationMetrics import BinaryClassificationMetrics

# configurate variables
file_path = '.\\csv\\WineQT.csv'
random_state_number = 12
target_variable = 'quality'

# read the dataset from csv
data = pd.read_csv(file_path)

# drop the id column
del data['Id']

# show number of records and number of columns
print('number of records and number of columns before select k best features -->',data.shape)
# show quality values sort by quality values
print('quality values sort by quality values')
print(data['quality'].value_counts())
print('|------------------------------------------------------------------------------------|')

# Randomly select a column to use as the target variable
# random.choice(data.columns)

# Separate the features and target variable
x = data.drop(columns=[target_variable])
y = data[target_variable]

# Randomly select the number of features to select
k = random.randint(2, int(len(x.columns) / 2))
# print the number of features to select

# Select the most important features
selector = SelectKBest(score_func=f_regression, k=k)
x_new = selector.fit_transform(x, y)

# Store the most important feature in an array
important_feature = x.columns[selector.get_support()][0]
important_feature_array = list(x.columns[selector.get_support()])

print('Select K Best Features')
print('number of features to select -->',k)
print("The target variable is:", target_variable)
print("The most important feature is:", important_feature)
print("Array with the most important feature:", important_feature_array)

# Drop columns that aren't in the important features array
columns_to_drop = set(x.columns) - set(important_feature_array)

x = x.drop(columns=columns_to_drop)
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=random_state_number, test_size=0.30)
print('number of records and number of columns after select k best features -->',x.shape)
print('|------------------------------------------------------------------------------------|')
print('SMOTE - Synthetic Minority Oversampling Technique')
# start to apply SMOTE
print('Start to apply SMOTE')
# show the shape of x_train and y_train
print('X_train.shape  -> ', x_train.shape, '  |  Y_train.shape  -> ', y_train.shape)

# Apply SMOTE to balance the dataset
overSample = SMOTE(random_state=12, k_neighbors=3)
x_train_smote, y_train_smote = overSample.fit_resample(x_train, y_train)

# show the shape of x_train and y_train
print('X_train_smote.shape  -> ', x_train.shape, '  |  Y_train.shape_smote  -> ', y_train.shape)
print(y_train_smote.value_counts())

# show number of records and number of columns
print('number of records and number of columns after apply SMOTE -->',x_train_smote.shape)

# show quality values sort by quality values
print('quality values sort by quality values')
print(y_train_smote.value_counts())
print('|------------------------------------------------------------------------------------|')

# apply SVM model on the dataset and print the accuracy score
svm_model = SVC(kernel='linear', C=1, gamma=1)
svm_model.fit(x_train_smote, y_train_smote)
y_pred = svm_model.predict(x_test)

# create an object from BinaryClassificationMetrics class and print the score's model
svm_metrics = BinaryClassificationMetrics()
svm_metrics.calculate_metrics(y_test, y_pred)
svm_metrics.print_metrics_with_model_name('SVM')

# apply Random Forest model on the dataset and print the accuracy score
rf_model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=random_state_number)
rf_model.fit(x_train_smote, y_train_smote)
y_pred = rf_model.predict(x_test)

rf_metrics = BinaryClassificationMetrics()
rf_metrics.calculate_metrics(y_test, y_pred)
rf_metrics.print_metrics_with_model_name('Random Forest')

# apply Logistic Regression model on the dataset and print the accuracy score
lr_model = LogisticRegression(random_state=random_state_number, solver='lbfgs', multi_class='multinomial',max_iter=1000)
lr_model.fit(x_train_smote, y_train_smote)
y_pred = lr_model.predict(x_test)

lg_metrics = BinaryClassificationMetrics()
lg_metrics.calculate_metrics(y_test, y_pred)
lg_metrics.print_metrics_with_model_name('Logistic Regression')
#print specificity
