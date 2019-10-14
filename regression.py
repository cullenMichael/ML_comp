import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn import metrics, model_selection, preprocessing, tree
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   StandardScaler)


def imputer(dataset):
    dataset = dataset.drop(columns=['Instance', 'Hair Color'])
    dataset['Gender'] = dataset['Gender'].replace(['0'], np.NaN)
    dataset['Gender'] = dataset['Gender'].replace(['unknown'], np.NaN)
    dataset['University Degree'] = dataset['University Degree'].replace(['0'], np.NaN)
    dataset['Country'] = dataset['Country'].replace(['0'], np.NaN)

    features_numeric = ['Year of Record', 'Age', 'Size of City', 'Body Height [cm]', 'Wears Glasses']
    features_categoric = ['University Degree', 'Gender', 'Country', 'Profession']

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
                        columns=['Year of Record', 'Age', 'Size of City', 'Body Height [cm]', 'Wears Glasses', 'University Degree', 'Gender', 'Country', 'Profession'])


def remove_outliers(data, income):
    income = pd.DataFrame({'Income in EUR': income})
    z = np.abs(stats.zscore(income))
    income_z = income[(z < 3).all(axis=1)]
    Q1 = income_z.quantile(0.25)
    Q3 = income_z.quantile(0.75)
    IQR = Q3 - Q1
    return income_z[~((income_z < (Q1 - 1.5 * IQR)) |(income_z > (Q3 + 1.5 * IQR))).any(axis=1)]
    # data = data.drop(["Income in EUR"],axis=1)
    # income_y = income_y.join(data, how='outer')
    # return income_y


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

    x_train = pd.get_dummies(data=df, columns=['Gender', 'University Degree', 'Country', 'Profession'])

    # x_train = remove_outliers(x_train, y_train)




    dataset_test_1 = pd.read_csv('tcd-ml-2019-20-income-prediction-test-without-labels.csv')
    dataset_test_1 = imputer(dataset_test_1)
    x_test = pd.get_dummies(data=dataset_test_1, columns=['Gender', 'University Degree', 'Country', 'Profession'])
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



    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)



    from sklearn.model_selection import cross_val_score
    from sklearn.neural_network import MLPRegressor



    regressor = MLPRegressor(hidden_layer_sizes=(6,5,6),max_iter=1500, random_state=0)
    print 'Fitting!'


    regressor.fit(x_train, y_train)
    print 'Predicting'
    y_pred = regressor.predict(x_test)

    # print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    # print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


    pd.DataFrame(y_pred, columns=['Income']).to_csv('tcd-ml-2019-20-income-prediction-test-without-labels.csv')


if __name__== "__main__":
    main()
