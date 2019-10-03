import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin
from sklearn import preprocessing
from sklearn import metrics

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets
from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn import preprocessing
from sklearn import utils
from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
le = LabelEncoder()
ohe = OneHotEncoder()


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)





dataset = pd.read_csv('tcd-ml-2019-20-income-prediction-training-with-labels.csv')
dataset.info()
print dataset.isnull().sum()



# dataset_imputed = DataFrameImputer().fit_transform(dataset)
#
#
#
#
#
#
# dataset_imputed['Country'] = le.fit_transform(dataset_imputed['Country'])
# dataset_imputed['Profession'] = le.fit_transform(dataset_imputed['Profession'])
# dataset_imputed['University Degree'] = le.fit_transform(dataset_imputed['University Degree'])
# dataset_imputed['Hair Color'] = le.fit_transform(dataset_imputed['Hair Color'])
# dataset_imputed['Gender'] = le.fit_transform(dataset_imputed['Gender'])
#
# y_train = dataset_imputed['Income in EUR'].values
# x_train = dataset_imputed[['Body Height [cm]', 'Country', 'Profession', 'University Degree', 'Wears Glasses', 'Hair Color', 'Gender', 'Age', 'Size of City', 'Year of Record']].values




# dataset_test_1 = pd.read_csv('tcd-ml-2019-20-income-prediction-test-without-labels.csv')
# dataset_test = DataFrameImputer().fit_transform(dataset_test_1)
#
# dataset_test['Country'] = le.fit_transform(dataset_test['Country'])
# dataset_test['Profession'] = le.fit_transform(dataset_test['Profession'])
# dataset_test['University Degree'] = le.fit_transform(dataset_test['University Degree'])
# dataset_test['Hair Color'] = le.fit_transform(dataset_test['Hair Color'])
# dataset_test['Gender'] = le.fit_transform(dataset_test['Gender'])
# x_test_data = dataset_test[['Body Height [cm]', 'Country', 'Profession', 'University Degree', 'Wears Glasses', 'Hair Color', 'Gender', 'Age', 'Size of City', 'Year of Record']].values



# correlation = dataset_imputed.corr(method='pearson')
# columns = correlation.nlargest(6, 'Income in EUR').index



# y_train = dataset_imputed['Income in EUR'].values
# dataset_imputed.drop(['Income in EUR'],axis=1, inplace=True)
#
#
# dataset_imputed = pd.concat([dataset_imputed,pd.get_dummies(dataset_imputed[['Country', 'Gender']], prefix=['Country', 'Gender'])],axis=1)
# dataset_imputed.drop(['Country'],axis=1, inplace=True)
# dataset_imputed.drop(['Profession'],axis=1, inplace=True)
# dataset_imputed.drop(['University Degree'],axis=1, inplace=True)
# dataset_imputed.drop(['Gender'],axis=1, inplace=True)
# dataset_imputed.drop(['Hair Color'],axis=1, inplace=True)

# # dataset_imputed = pd.get_dummies(dataset_imputed, columns=['Body Height [cm]', 'Country', 'Profession', 'University Degree', 'Wears Glasses', 'Hair Color', 'Gender', 'Age', 'Size of City', 'Year of Record'])
# dataset_imputed = pd.get_dummies(dataset_imputed, prefix=['Country', 'Profession', 'Gender'],
#                    columns=[ 'Country', 'Profession', 'Gender'], drop_first=True)

# x_train = dataset_imputed.values
#
# print x_train


# print("Original features:\n", list(y_train.columns), "\n")
# data_dummies = pd.DataFrame(pd.get_dummies(dataset_imputed))
# print("Features after get_dummies:\n", list(data_dummies.columns))




#
# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
#
#
#
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import ElasticNet
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.ensemble import GradientBoostingRegressor


# reg = LinearRegression().fit(x_train, y_train)
#
# y_pred = reg.predict(x_test)
#
#
# # print y_pred
#
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#
#
# reg =  Lasso().fit(x_train, y_train)
#
# y_pred = reg.predict(x_test)
#
#
# # print y_pred
#
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#
#
# reg = ElasticNet().fit(x_train, y_train)
#
# y_pred = reg.predict(x_test)
#
#
# # print y_pred
#
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#
#
# reg = KNeighborsRegressor().fit(x_train, y_train)
#
# y_pred = reg.predict(x_test)
#
#
# # print y_pred
#
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#
#
#
# reg = LinearRegression().fit(x_train, y_train)
#
# y_pred = reg.predict(x_test)
#
#
# # print y_pred
# #
# # print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# # print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#
#
#
# # reg = GradientBoostingRegressor().fit(x_train, y_train)
# #
# # y_pred = reg.predict(x_test)
# #
# #
# # # print y_pred
# #
# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# pd.DataFrame(y_pred, columns=['Income']).to_csv('tcd-ml-2019-20-income-prediction-test-without-labels.csv')



# pipelines = []
# pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
# pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
# pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
# pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
# pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
# # pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))
#
# results = []
# names = []
# for name, model in pipelines:
#     kfold = KFold(n_splits = 10, random_state = None, shuffle = False)
#     cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)
