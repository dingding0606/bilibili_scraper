import numpy as np
import pandas as pd
import random
import math
from data_loader import load_bilibili_data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow_addons.metrics import RSquare
import keras_tuner

def fit_linear_regression(Xmat_train, Y_train, Xmat_val, Y_val):
    # ==================================
    # BASELINE MODEL: LINEAR REGRESSION
    # ==================================
    baseline_model = LinearRegression()
    baseline_model.fit(Xmat_train, Y_train)

    # Obtain the coefficient of determination by calling the model with the score() function, then print the coefficient:
    r_squared_train = baseline_model.score(Xmat_train, Y_train)
    print('R-sqaured on training set:', r_squared_train)

    r_squared_val = baseline_model.score(Xmat_val, Y_val)
    print('R-sqaured on validation set:', r_squared_val)


    # squares

    for feature in Xmat_train.columns:
        Xmat_train[feature + "^2"] =  Xmat_train[feature].pow(2)
        Xmat_val[feature + "^2"] =  Xmat_val[feature].pow(2)

    baseline_model.fit(Xmat_train, Y_train)

    # Obtain the coefficient of determination by calling the model with the score() function, then print the coefficient:
    r_squared_train = baseline_model.score(Xmat_train, Y_train)
    print('R-sqaured on training set:', r_squared_train)

    r_squared_val = baseline_model.score(Xmat_val, Y_val)
    print('R-sqaured on validation set:', r_squared_val)


    print(Xmat_train)


def fit_polynomial_regression(Xmat_train_and_val, Y_train_and_val, split_index):
    # =====================
    # POLYNOMIAL REGRESSION
    # =====================

    '''
    Look at this: https://stackoverflow.com/questions/51459406/how-to-apply-standardscaler-in-pipeline-in-scikit-learn-sklearn
    '''
    steps = [
        ('poly', PolynomialFeatures()),
        ('scalar', StandardScaler()),
        ('model', Ridge()) #Lasso(alpha=0.9, max_iter=10000, fit_intercept=True))
    ]

    pipeline = Pipeline(steps)

    # =================== 1st Grid Search ===================
    degrees = [2, 3]
    alphas = [0.01, 0.1, 1, 10, 100, 1000, 10000, 30000, 40000, 50000, 60000]

    param_grid = {
        "poly__degree" : degrees,
        "model__alpha" : alphas,
    }

    # Use the list split_index to create PredefinedSplit
    pds = PredefinedSplit(test_fold = split_index)

    search = GridSearchCV(estimator=pipeline,
                          param_grid=param_grid,
                          scoring="r2",
                          cv=pds,
                          verbose=3)

    search.fit(Xmat_train_and_val, Y_train_and_val)

    df_gridsearch = pd.DataFrame(search.cv_results_)

    scores = search.cv_results_['split0_test_score']

    # replace negatives scores with zero, then transform into the desired shape
    scores[scores < 0] = 0
    scores = np.array(scores)
    scores = np.transpose(np.vstack(np.split(scores, len(alphas))))

    # Plot using Matplotlib
    for i, degree in enumerate(degrees):
        plt.plot(np.log(alphas), scores[i], label='degree: ' + str(degree))

    plt.legend()
    plt.xlabel('log (Alpha)')
    plt.ylabel('R Sqaured')
    plt.show()


    # =================== 2nd Grid Search ===================
    # degrees = [2]
    # alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.8, 0.99]
    #
    # param_grid = {
    #     "poly__degree" : degrees,
    #     "model__alpha" : alphas,
    #     "model__max_iter" : [100000],
    # }
    #
    # # Use the list split_index to create PredefinedSplit
    # pds = PredefinedSplit(test_fold = split_index)
    #
    # search = GridSearchCV(estimator=pipeline,
    #                       param_grid=param_grid,
    #                       scoring="r2",
    #                       cv=pds,
    #                       verbose=3)
    #
    # search.fit(Xmat_train_and_val, Y_train_and_val)
    #
    # df_gridsearch = pd.DataFrame(search.cv_results_)
    #
    # scores = search.cv_results_['split0_test_score']
    #
    # # replace negatives scores with zero, then transform into the desired shape
    # scores[scores < 0] = 0
    # scores = np.array(scores)
    # scores = np.transpose(np.vstack(np.split(scores, len(alphas))))
    #
    # # Plot using Matplotlib
    # for i, degree in enumerate(degrees):
    #     plt.plot(alphas, scores[i], label='degree: ' + str(degree))
    #
    # plt.legend()
    # plt.xlabel('Alpha')
    # plt.ylabel('R Sqaured')
    # plt.show()


def fit_random_forest(Xmat_train_and_val, Y_train_and_val, split_index):
    # =========================
    # 🌲🌲🌲 RANDOM FOREST 🌲🌲🌲
    # =========================
    # Presentation of grid search: don't have to output a graph. I can just say that
    # here are the values that a grid searched on, and here is the model with the best validation score.

    # Number of trees in random forest
    n_estimators = [1, 10, 100, 1000, 10000] # [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

    # Number of features to consider at every split
    max_features = [1.0, 'sqrt'] # [1.0, 'sqrt']

    # Maximum number of levels in tree
    max_depth = [10, 30, 60, 100] # [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)

    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # Method of selecting samples for training each tree
    bootstrap = [True] #[True, False]

    # Create the random grid
    param_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestRegressor()

    # Use the list split_index to create PredefinedSplit (predefined Validation set)
    pds = PredefinedSplit(test_fold = split_index)


    search = GridSearchCV(estimator=rf,
                          param_grid=param_grid,
                          scoring="r2",
                          cv=pds,
                          verbose=3,
                          n_jobs=-1)

    search.fit(Xmat_train_and_val, Y_train_and_val)

    print(search.cv_results_)

    return search.cv_results_

#     df_gridsearch = pd.DataFrame(search.cv_results_)

#     scores = search.cv_results_['split0_test_score']

    # Random search of parameters, using predefined validation set,
    # search across 100 different combinations, and use all available cores.
    # P.S. n_jobs = -1 means use all processors to run them in parallel
    # rf_random = RandomizedSearchCV(estimator=rf,
    #                                param_distributions=param_grid,
    #                                scoring="r2",
    #                                n_iter=100,
    #                                cv=pds,
    #                                verbose=3,
    #                                random_state=2022,
    #                                n_jobs=-1)
    #
    # rf_random.fit(Xmat_train_and_val, Y_train_and_val)


def fit_neural_network_sklearn(Xmat_train, Y_train, Xmat_val, Y_val):
    model = MLPRegressor(solver='adam', activation='relu', alpha=0.01, hidden_layer_sizes=(128, 64, 16), random_state=42, max_iter=3000)
    model.fit(Xmat_train, Y_train)

    r_squared_train = model.score(Xmat_train, Y_train)
    print('R-sqaured on training set:', r_squared_train)

    r_squared_val = model.score(Xmat_val, Y_val)
    print('R-sqaured on validation set:', r_squared_val)


def fit_neural_network_keras(Xmat_train, Y_train, Xmat_val, Y_val):
    '''
    again, i can present a list of architecture and parameters that I tried, and present graph for the best one.

    Also, at the end, make sure to talk about the "accuracy" of the model in context.

    Two ways this can be done:

    1. the average difference between Y and Y_hat
    2. whether the ordering of the videos are preserved: e.g. does the actual top 10 popular videos got predicted to be top 10 with the same ordering?
    '''
    model = Sequential()
    model.add(Dense(32, input_dim=len(Xmat_train.columns), use_bias=True, bias_initializer="zeros", kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=[RSquare()])

    model.summary()

    history = model.fit(Xmat_train,
                        Y_train,
                        batch_size=32,
                        epochs=2000,
                        validation_data=(Xmat_val, Y_val))

    r_square = history.history['r_square']
    val_r_square = history.history['val_r_square']
    epochs = range(1, len(r_square) + 1)

    plt.plot(epochs, r_square, 'y', label='Training R Squared')
    plt.plot(epochs, val_r_square, 'r', label='Validation R Squared')

    plt.title('Training and validation R Squared')
    plt.xlabel('Epochs')
    plt.ylabel('R squared')
    plt.legend()
    plt.show()

    # build_neural_network(keras_tuner.HyperParameters())


    # print(model.evaluate(Xmat_val, Y_val))


def build_neural_network(hp, input_dim):
    model = keras.Sequential()

    '''
    Some Default Values of Dense:
        use_bias=True,
        bias_initializer='zeros',
        kernel_regularizer=None
    '''

    model.add(Dense(units=hp.Int("units", min_value=16, max_value=256, step=16),
                    input_dim=input_dim,
                    kernel_initializer='random_normal',
                    activation='relu'))

    model.add(Dense(units=hp.Int("units", min_value=8, max_value=32, step=8),
                    kernel_initializer='random_normal',
                    activation='relu'))

    # Output Layer
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam', metrics=[RSquare()])

    return model


def main():

    # All of those are pandas objects
    Xmat_train_and_val, Y_train_and_val, Xmat_train, Xmat_val, Xmat_test, Y_train, Y_val, Y_test = load_bilibili_data()

    # Create a list where train data indices are -1 and validation data indices are 0
    split_index = [-1 if x in Xmat_train.index else 0 for x in Xmat_train_and_val.index]

    fit_linear_regression(Xmat_train, Y_train, Xmat_val, Y_val)

    fit_polynomial_regression(Xmat_train_and_val, Y_train_and_val, split_index)
    #
    # fit_random_forest(Xmat_train_and_val, Y_train_and_val, split_index)
    #
    # fit_neural_network_keras(Xmat_train, Y_train, Xmat_val, Y_val)

    #
    # for i in range(29, 40):
    #     print("max_features = ", i)
    #     forest = RandomForestRegressor(n_estimators=100, max_features=i)
    #     forest.fit(Xmat_train, Y_train)
    #     print('Training score: {}'.format(forest.score(Xmat_train, Y_train)))
    #     print('Validation score: {}'.format(forest.score(Xmat_val, Y_val)))
    #     print("\n")


main()
