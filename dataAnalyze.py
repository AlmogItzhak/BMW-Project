import numpy as np
import pandas as pd  # module in python that works on tables- datasets
# import sns as sns
# import train as train
# from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
#
# from sklearn import preprocessing
#
# file_path = '.\\csv\\student_data.csv'
# data = pd.read_csv(file_path)# print(train.head())  # show us the first 5 lines
# train=train.set_index('Id') #set Id index
# train.info() #gives ius information about the dataset
#
#             #EXPLORATORY DATA ANALYSIS#
#
# # msno.bar(train, figsize = (16,5),color = "#FFE4E1")  #show us the datasets features - gets the dataset, size and color
# # plt.show()
#
# print(train) #shows us first 5 lines and last 5 lines
#
# print(train.describe()) #shows us the statistics of the dataset
#
# columns=train.columns
# sns.set()  #sns- Seaborn function
# sns.pairplot(train[columns],height = 5 ,kind ='scatter',diag_kind='kde') #make pairplot as defined
# plt.show()  #now we can see the information - scatter
#
# fig = go.Figure(data=[go.Pie(labels=train['quality'].value_counts().index, values=train['quality'].value_counts(), hole=.3)]) #figure() Function of value counts- quality
# fig.update_layout(legend_title_text='Quality')
# fig.show()
#
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
#
#             #STATISTICAL TESTS#
#
# for column_name in train.columns: #showing how many unique values belong to feature
#     unique_values = len(train[column_name].unique())
#     print("Feature '{column_name}' has '{unique_values}' unique values".format(column_name = column_name,   #.format()- formats the specified value(s) and insert them inside the string's placeholder.
#                                                                                          unique_values=unique_values))
#
#     import scipy.stats
#
#     # u : Mann-Whitney test statistic
#     # p : p-value
#     for feature in train.columns:
#         u, p = scipy.stats.mannwhitneyu(train['quality'], train[feature]) #mannwhiteyu test -used to analyze the difference between two independent samples of ordinal data.
#         print('With', feature)
#         print('Mann-Whitney test statistic:', u)
#         print('p-value:', p)
#         print('--------------------')
#
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
#
#
#             #FEATURE ENGENEERING AND MACHINE LEARNING#
#
# from imblearn.over_sampling import SMOTE  #SMOTE is one of the most commonly used oversampling methods to solve the imbalance problem. It aims to balance class distribution by randomly increasing minority class examples by replicating them
# oversample = SMOTE() #. The simplest approach involves duplicating examples in the minority class, although these examples don’t add any new information to the model. Instead, new examples can be synthesized from the existing examples
# features, labels=  oversample.fit_resample(train.drop(["quality"],axis=1),train["quality"])
#
# scaler = preprocessing.MinMaxScaler()  #Transform features by scaling each feature to a given range.
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
# from sklearn.metrics import accuracy_score, classification_report
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
#
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
#

