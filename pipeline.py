from enum import Enum

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, f_regression
import random

from sklearn.neural_network import MLPClassifier

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
from sklearn.metrics import recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

from binaryClassificationMetrics import BinaryClassificationMetrics


def read_dataset(file_path):
    data = pd.read_csv(file_path)

    return data

def encode_categorical_features(x):
    categorical_columns = x.select_dtypes(include=['object']).columns

    if len(categorical_columns) == 0:
        return x

    transformer = ColumnTransformer([('encoder', OneHotEncoder(), categorical_columns)], remainder='passthrough')
    encoded_data = transformer.fit_transform(x)

    return pd.DataFrame(encoded_data)

def select_k_best_features(x, y, k):
    selector = SelectKBest(score_func=f_regression, k=k)
    x_new = selector.fit_transform(x, y)
    important_feature = x.columns[selector.get_support()][0]
    important_feature_array = list(x.columns[selector.get_support()])
    columns_to_drop = set(x.columns) - set(important_feature_array)
    x = x.drop(columns=columns_to_drop)
    return x, important_feature, important_feature_array

def apply_ann(x_train, y_train, x_test):
    ann = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=random_state_number)
    ann.fit(x_train, y_train)
    return ann.predict(x_test)

def apply_method(chosen_method,x_train, y_train, x_test,y_test):
    if chosen_method == Method.SMOTE_WITH_UNDERSAMPLING.value:
        print('Start RUS ALGORITHM')
        rus = RandomUnderSampler(random_state=random_state_number)
        x_train_rus, y_train_rus = rus.fit_resample(x_train, y_train)
        print('Finish RUS ALGORITHM')
        print('Start SMOTE with undersampling ALGORITHM')
        smote = SMOTE(random_state=random_state_number)
        x_train_resampled, y_train_resampled = smote.fit_resample(x_train_rus, y_train_rus)
        print('Finish SMOTE with undersampling ALGORITHM')
        return apply_model(x_train_resampled, y_train_resampled, x_test)

    elif chosen_method == Method.SMOTE_WITH_OVERSAMPLING.value:
        print('Start SMOTE with oversampling ALGORITHM')
        smote = SMOTE(random_state=random_state_number)
        x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
        print('Finish SMOTE with oversampling ALGORITHM')
        return apply_model(x_train_resampled, y_train_resampled, x_test)

    elif chosen_method == Method.ADABOOST.value:
        print('Start Adaboost ALGORITHM')
        adaboost = AdaBoostClassifier(n_estimators=100, random_state=random_state_number)
        adaboost.fit(x_train, y_train)
        print('Finish Adaboost ALGORITHM')
        return adaboost.predict(x_test)

    elif chosen_method == Method.COST_SENSITIVE_LEARNING.value:
        print('Start cost sensetive learning ALGORITHM')
        class_weights = np.divide(1, np.multiply(np.bincount(y_train), len(y_train) / np.sum(len(y_train))))
        class_weight_dict = dict(enumerate(class_weights))
        svm_model = SVC(kernel='linear', C=1, gamma=1, class_weight=class_weight_dict)
        svm_model.fit(x_train, y_train)
        print('Finish Adaboost ALGORITHM')
        return svm_model.predict(x_test)

    elif chosen_method == Method.ACTIVE_LEARNING.value:
        print('Start active learning ALGORITHM')
        svm_model = SVC(kernel='linear', C=1, gamma=1)
        svm_model.fit(x_train, y.train)
        print('Finish active learning ALGORITHM')
        return svm_model.predict(x_test)

    elif chosen_method == Method.ANN.value:
        print('Start artificial neural network ALGORITHM')
        return apply_ann(x_train, y_train, x_test)

    elif chosen_method == Method.RANDOM_FOREST.value:
        print('Start random forest ALGORITHM')
        return apply_random_forest(x_train, y_train, x_test)

    elif chosen_method == Method.LOGISTIC_REGRESSION.value:
        print('Start logistic regression ALGORITHM')
        return apply_logistic_regression(x_train, y_train, x_test)

    elif chosen_method == Method.SVM.value:
        print('Start SVM ALGORITHM')
        return apply_svm(x_train, y_train, x_test)

    else:
        print("Method not implemented")

#convert enum to title
def convert_enum_to_title(enum):
    if enum == Method.SMOTE_WITH_UNDERSAMPLING.value:
        return Method.SMOTE_WITH_UNDERSAMPLING.name
    elif enum == Method.SMOTE_WITH_OVERSAMPLING.value:
        return Method.SMOTE_WITH_OVERSAMPLING.name
    elif enum == Method.ADABOOST.value:
        return Method.ADABOOST.name
    elif enum == Method.COST_SENSITIVE_LEARNING.value:
        return Method.COST_SENSITIVE_LEARNING.name
    elif enum == Method.ACTIVE_LEARNING.value:
        return Method.ACTIVE_LEARNING.name
    elif enum == Method.ANN.value:
        return Method.ANN.name
    elif enum == Method.RANDOM_FOREST.value:
        return Method.RANDOM_FOREST.name
    elif enum == Method.LOGISTIC_REGRESSION.value:
        return Method.LOGISTIC_REGRESSION.name
    elif enum == Method.SVM.value:
        return Method.SVM.name

def apply_model(x_train_balanced, y_train_balanced, x_test):
    svm_model = SVC(kernel='linear', C=1, gamma=1)
    svm_model.fit(x_train_balanced, y_train_balanced)
    y_pred = svm_model.predict(x_test)
    return y_pred

def apply_svm(x_train_rus, y_train_rus, x_test):
    svm_model = SVC(kernel='linear', C=1, gamma=1)
    svm_model.fit(x_train_rus, y_train_rus)
    y_pred = svm_model.predict(x_test)
    return y_pred

def apply_random_forest(x_train_rus, y_train_rus, x_test):
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=random_state_number)
    rf_model.fit(x_train_rus, y_train_rus)
    y_pred = rf_model.predict(x_test)
    return y_pred

def apply_logistic_regression(x_train_rus, y_train_rus, x_test):
    lr_model = LogisticRegression(random_state=random_state_number, solver='lbfgs', multi_class='multinomial', max_iter=1000)
    lr_model.fit(x_train_rus, y_train_rus)
    y_pred = lr_model.predict(x_test)
    return y_pred

def calculate_metrics(y_true, y_pred):
    metrics = BinaryClassificationMetrics()
    if np.sum(y_true) > 0:
        metrics.calculate_metrics(y_true, y_pred)
    else:
        metrics.set_empty_metrics()
    return metrics

def print_metrics_with_model_name(metrics, model_name):
    metrics.print_metrics_with_model_name(model_name)

# define enum for method names
class Method(Enum):
    ANN = 1
    ACTIVE_LEARNING = 2
    COST_SENSITIVE_LEARNING = 3
    ADABOOST = 4
    SMOTE_WITH_OVERSAMPLING = 5
    SMOTE_WITH_UNDERSAMPLING = 6
    RANDOM_FOREST = 7
    LOGISTIC_REGRESSION = 8
    SVM = 9

# Step 1: Request the CSV file name from the user
csv_file = input("Please enter the CSV file name: ")
file_path = '.\\csv\\'+ csv_file + '.csv'

# Step 2: Read the dataset from csv
data = read_dataset(file_path)

# Step 3: Show the headers
headers = data.columns.tolist()
print("Headers in the CSV file:")
for header in headers:
    print(header)

# Step 4: Request the target variable from the user
target_variable = input("Please enter the target variable: ")

# Step 5: Request the chosen method from the user
print("Available methods:")
print("1. Artificial Neural Network")
print("2. Active Learning")
print("3. Cost-Sensitive Learning")
print("4. AdaBoost")
print("5. SMOTE with OverSampling")
print("6. SMOTE with UnderSampling")
print("7. Random Forest")
print("8. Logistic Regression")
print("9. SVM")

chosen_method = int(input("Please enter the method number: "))
chosen_method_name = convert_enum_to_title(chosen_method)

# Configuration variables
random_state_number = 12


# Show number of records and number of columns
print('Number of records and number of columns before selecting K best features:', data.shape)

# Show quality values sorted by quality values
print(target_variable + ' values sorted by ' + target_variable + ' values:')
print(data[target_variable].value_counts())
print('|-------------------------------------------------------------------------|')

# Separate the features and target variable

oversample = SMOTE()
x = data.drop([target_variable],axis=1)
y = data[target_variable]

x, y =  oversample.fit_resample(x, y)
x = encode_categorical_features(x)

# Encode categorical variables

# Randomly select the number of features to select
# Modify the range to ensure it's within valid bounds
k_max = max(2, int(len(x.columns) / 2))
k = random.randint(2, k_max)

# Select the most important features
x, important_feature, important_feature_array = select_k_best_features(x, y, k)

print('Select K Best Features')
# Print the number of features to select
print('Number of features to select:', k)
print("The target variable is:", target_variable)
print("The most important feature is:", important_feature)
print("Array with the most important features:", important_feature_array)
# class_counts = y.value_counts()
# classes_to_keep = class_counts[class_counts >= 2].index
# data_filtered = data[data[target_variable].isin(classes_to_keep)]

# Split the dataset into train and test sets

# Show number of records and number of columns after selecting K best features
print('Number of records and number of columns after selecting K best features:', x.shape)
print('|-------------------------------------------------------------------------|')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state_number, test_size=0.30)

y_pred = apply_method(chosen_method, x_train, y_train, x_test, y_test)


y_test = y_test.to_numpy()
# Calculate and print the metrics
metrics = calculate_metrics(y_test, y_pred)
print_metrics_with_model_name(metrics, chosen_method_name)

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()