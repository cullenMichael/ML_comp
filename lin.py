import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.svm import SVR
from scipy import stats
le = LabelEncoder()

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing

from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor


def simplify_ages(df):

    return df


def imputer(dataset):
    dataset = dataset.drop(columns=['Instance', 'Hair Color', 'Wears Glasses'])

    dataset['Gender'] = dataset['Gender'].replace(['0'], 'male')
    dataset['Gender'] = dataset['Gender'].replace(['unknown'], np.NaN)
    dataset['University Degree'] = dataset['University Degree'].replace(['0'], 'no')
    dataset['Country'] = dataset['Country'].replace(['0'], np.NaN)
    features_numeric = ['Year of Record', 'Age', 'Size of City', 'Body Height [cm]']
    features_categoric = ['University Degree', 'Gender', 'Country']

    imputer_numeric = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
    ])

    imputer_categoric = Pipeline(
        steps=[('imputer',
                SimpleImputer(strategy='most_frequent'))])

    preprocessor = ColumnTransformer(transformers=[('imputer_numeric',
                                                    imputer_numeric,
                                                    features_numeric),
                                                   ('imputer_categoric',
                                                    imputer_categoric,
                                                    features_categoric)])

    preprocessor.fit(dataset)
    return pd.DataFrame(preprocessor.transform(dataset),
                        columns=['Year of Record', 'Age', 'Size of City', 'Body Height [cm]', 'University Degree', 'Gender', 'Country'])


def encode(dataset):
    from sklearn.preprocessing import OneHotEncoder
    dataset['Country'] = le.fit_transform(dataset['Country'])
    dataset['University Degree'] = le.fit_transform(dataset['University Degree'])
    dataset['Gender'] = le.fit_transform(dataset['Gender'])

    onehotencoder = OneHotEncoder()

    return onehotencoder.fit_transform(dataset).toarray()



def impute_ohe(dataset):
        dataset = dataset.drop(columns=['Instance', 'Hair Color', 'Wears Glasses'])

        dataset['Gender'] = dataset['Gender'].replace(['0'], np.NaN)
        dataset['Gender'] = dataset['Gender'].replace(['unknown'], np.NaN)
        dataset['University Degree'] = dataset['University Degree'].replace(['0'], np.NaN)
        dataset['Country'] = dataset['Country'].replace(['0'], np.NaN)



        dataset['Country'] = le.fit_transform(dataset['Country'])
        dataset['University Degree'] = le.fit_transform(dataset['University Degree'])
        dataset['Gender'] = le.fit_transform(dataset['Gender'])

        from sklearn.preprocessing import OneHotEncoder


        onehotencoder = OneHotEncoder()

        return onehotencoder.fit_transform(dataset).toarray()














def scatter(feature, target, dataset):
    plt.figure(figsize=(16,8))
    plt.scatter(dataset[feature], dataset[target], c='black')
    plt.xlabel('{}'.format(feature))
    plt.ylabel('{}'.format(target))
    plt.show()





def main():
    df = pd.read_csv('tcd-ml-2019-20-income-prediction-training-with-labels.csv')
    y_train = df['Income in EUR'].values
    df = df.drop(columns=['Income in EUR'])
    df = imputer(df)

    x_train = pd.get_dummies(data=df, columns=['Gender', 'University Degree', 'Country'])




    dataset_test_1 = pd.read_csv('tcd-ml-2019-20-income-prediction-test-without-labels.csv')
    dataset_test_1 = imputer(dataset_test_1)
    x_test = pd.get_dummies(data=dataset_test_1, columns=['Gender', 'University Degree', 'Country'])
    x_train, x_test = x_train.align(x_test, join='outer', axis=1, fill_value=0)


    from sklearn.preprocessing import FunctionTransformer
    transformer = FunctionTransformer(np.log1p, validate=True)
    x_train = transformer.transform(x_train)

    scaler = MinMaxScaler(feature_range = (0,1))

    scaler.fit(x_train)
    x_train = scaler.transform(x_train)



    from sklearn.preprocessing import FunctionTransformer
    transformer = FunctionTransformer(np.log1p, validate=True)
    x_test = transformer.transform(x_test)


    scaler = MinMaxScaler(feature_range = (0,1))

    scaler.fit(x_test)
    x_test = scaler.transform(x_test)



    # x_train1, x_test, y_train1, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)



    from sklearn.model_selection import cross_val_score
    from sklearn.neural_network import MLPRegressor



    regressor = MLPRegressor(hidden_layer_sizes=(5,5,5),max_iter=500, shuffle=True, random_state=0)
    # cross_val_score(regressor, x_train, y_train, cv=5)
    regressor.fit(x_train, y_train)

    y_pred = regressor.predict(x_test)
    y_pred = np.around(y_pred,2)

    # print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    # print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))




    pd.DataFrame(y_pred, columns=['Income']).to_csv('tcd-ml-2019-20-income-prediction-test-without-labels.csv')


if __name__== "__main__":
    main()
