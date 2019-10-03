import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# %matplotlib inline
from sklearn.preprocessing import LabelEncoder
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



import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin

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

dataset_imputed = DataFrameImputer().fit_transform(dataset)

dataset_imputed['Country'] = le.fit_transform(dataset_imputed['Country'])
dataset_imputed['Profession'] = le.fit_transform(dataset_imputed['Profession'])
dataset_imputed['University Degree'] = le.fit_transform(dataset_imputed['University Degree'])
dataset_imputed['Hair Color'] = le.fit_transform(dataset_imputed['Hair Color'])
dataset_imputed['Gender'] = le.fit_transform(dataset_imputed['Gender'])

# dataset_test_1 = pd.read_csv('tcd-ml-2019-20-income-prediction-test-without-labels.csv')
# dataset_test = DataFrameImputer().fit_transform(dataset_test_1)
#
# dataset_test['Country'] = le.fit_transform(dataset_test['Country'])
# dataset_test['Profession'] = le.fit_transform(dataset_test['Profession'])
# dataset_test['University Degree'] = le.fit_transform(dataset_test['University Degree'])
# dataset_test['Hair Color'] = le.fit_transform(dataset_test['Hair Color'])
# dataset_test['Gender'] = le.fit_transform(dataset_test['Gender'])

y_train = dataset_imputed['Income in EUR'].values
x_train = dataset_imputed[['Body Height [cm]', 'Country', 'Profession', 'University Degree', 'Wears Glasses', 'Hair Color', 'Gender', 'Age', 'Size of City', 'Year of Record']].values



thresholder = VarianceThreshold(threshold=0.2)
x_train = pd.DataFrame(thresholder.fit_transform(x_train))



# x_test_data = dataset_test[['Body Height [cm]', 'Country', 'Profession', 'University Degree', 'Wears Glasses', 'Hair Color', 'Gender', 'Age', 'Size of City', 'Year of Record']].values


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

# params = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 100, 1000, 100000, 1000000]}
#
# lasso_reg = GridSearchCV(lasso,params,scoring='neg_mean_squared_error', cv=5, max_iter=100000)
lasso_reg = Lasso(alpha=0.0001, max_iter=10e5)
lasso_reg.fit(x_train, y_train)

# regressor = LinearRegression()
# regressor.fit(x_train, y_train)
#
#
y_pred = lasso_reg.predict(x_test)


# print y_pred

# print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))




pd.DataFrame(y_pred, columns=['Income']).to_csv('tcd-ml-2019-20-income-prediction-test-without-labels.csv')
