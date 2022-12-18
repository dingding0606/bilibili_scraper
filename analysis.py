import numpy as np
import pandas as pd
import random
import os
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
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow_addons.metrics import RSquare
import keras_tuner
from sklearn.model_selection import cross_val_score
from tensorflow.random import set_seed

from keras import backend as K
import tensorflow as tf

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

    coefficients = pd.DataFrame({"Feature":Xmat_train.columns, "Coefficients":np.transpose(baseline_model.coef_)})

    print(coefficients)

    # squares

    # for feature in Xmat_train.columns:
    #     Xmat_train[feature + "^2"] =  Xmat_train[feature].pow(2)
    #     Xmat_val[feature + "^2"] =  Xmat_val[feature].pow(2)
    #
    # baseline_model.fit(Xmat_train, Y_train)
    #
    # # Obtain the coefficient of determination by calling the model with the score() function, then print the coefficient:
    # r_squared_train = baseline_model.score(Xmat_train, Y_train)
    # print('R-sqaured on training set:', r_squared_train)
    #
    # r_squared_val = baseline_model.score(Xmat_val, Y_val)
    # print('R-sqaured on validation set:', r_squared_val)
    #
    #
    # print(Xmat_train)


def fit_polynomial_regression(ns_Xmat_train_and_val, ns_Y_train_and_val, split_index):
    # =====================
    # POLYNOMIAL REGRESSION
    # =====================

    '''
    Look at this: https://stackoverflow.com/questions/51459406/how-to-apply-standardscaler-in-pipeline-in-scikit-learn-sklearn
    '''
    steps = [
        ('poly', PolynomialFeatures()),
        ('scalar', StandardScaler()),
        ('model', Ridge(max_iter=10000)) #Lasso(alpha=0.9, max_iter=10000, fit_intercept=True))
    ]

    pipeline = Pipeline(steps)

    # =================== 1st Grid Search ===================
    degrees = [4]
    alphas = [160000, 200000, 300000, 400000, 500000] # [0.01, 0.1, 1, 10, 100, 1000, 10000, 30000, 40000, 50000, 60000]

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

    search.fit(ns_Xmat_train_and_val, ns_Y_train_and_val)

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
    n_estimators = [2000, 4000, 6000, 8000, 10000] # [int(x) for x in np.linspace(start = 100, stop = 2000, num = 6)]

    # Number of features to consider at every split
    max_features = ['sqrt'] # [1.0, 'sqrt']

    # Maximum number of levels in tree
    max_depth = [] #[int(x) for x in np.linspace(10, 100, num = 10)]
    max_depth.append(None)

    # Minimum number of samples required to split a node
    # min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 4]

    # Method of selecting samples for training each tree
    bootstrap = [True] #[True, False]

    # Create the random grid
    param_grid = { 'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                   # 'min_samples_split': min_samples_split,
                   # 'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'random_state': [42]}

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

    # print(search.cv_results_)

    scores = search.cv_results_['split0_test_score']

    # Transform into the desired shape
    scores = np.array(scores)
    scores = np.transpose(np.vstack(np.split(scores, len(max_depth))))

    # change None to 200 so that it is easier to plot
    max_depth[-1] = 110
    print("MAX DEPTH: ", max_depth)

    # Plot using Matplotlib
    for i, num_of_trees in enumerate(n_estimators):
        plt.plot(max_depth, scores[i], label='m = ' + str(num_of_trees))

    plt.legend()
    plt.xlabel('Maximum Depth of Each Tree')
    plt.ylabel('R Sqaured')
    plt.show()

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


def fit_best_random_forest(Xmat_train, Y_train, Xmat_val, Y_val, Xmat_train_and_val, Y_train_and_val, split_index):
    model = RandomForestRegressor(bootstrap=True, max_depth=30, max_features=1.0, n_estimators=100, random_state=42)
    model.fit(Xmat_train, Y_train)

    # Obtain the coefficient of determination by calling the model with the score() function, then print the coefficient:
    r_squared_train = model.score(Xmat_train, Y_train)
    print('[Best RF] R-sqaured on training set:', r_squared_train)

    r_squared_val = model.score(Xmat_val, Y_val)
    print('[Best RF] R-sqaured on validation set:', r_squared_val)

    pds = PredefinedSplit(test_fold = split_index)

    cv_score = cross_val_score(model, Xmat_train_and_val, Y_train_and_val, cv=pds)
    print(cv_score)

    return pds


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

    [32, 16, 1], no dropout, relu, adam -> 0.81 on validation.
    '''
    model = Sequential()
    model.add(Dense(352, input_dim=len(Xmat_train.columns), kernel_initializer='random_normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(224, kernel_initializer='random_normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    # set up an SGD optimizer with custom learning rate
    sgd = SGD(learning_rate=0.005579936781430615, clipvalue=0.5)

    model.compile(loss='mse', optimizer=sgd, metrics=[RSquare()])

    model.summary()

    num_of_rows = len(Xmat_train)

    history = model.fit(Xmat_train,
                        Y_train,
                        batch_size=64,
                        epochs=1000,
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

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(r_square) + 1)

    plt.plot(epochs, loss, 'y', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')

    plt.title('Training and validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.show()



def grid_search_neural_network(Xmat_train, Y_train, Xmat_val, Y_val):

    print("[STATUS] Starting Grid Search On Neural Nets")

    tuner = keras_tuner.RandomSearch(
        hypermodel=build_neural_network,
        objective="val_loss",
        max_trials=100,
        executions_per_trial=2,
        seed=42,
        overwrite=True,
        directory="tuner",
        project_name="keras_nn_tuner",
    )

    # early_stopping_detector = EarlyStopping(monitor='r_square',
    #                                         min_delta=0.01,
    #                                         patience=50,
    #                                         start_from_epoch=100)

    tuner.search(Xmat_train,
                 Y_train,
                 batch_size=64,
                 epochs=1000,
                 validation_data=(Xmat_val, Y_val),
                 #callbacks=[early_stopping_detector]
                 )

    print(tuner.results_summary())

    # Get the top 3 models.
    # models = tuner.get_best_models(num_models=3)
    # best_model = models[0]
    # # Build the model.
    # # Needed for `Sequential` without specified `input_shape`.
    # best_model.build(input_shape=(None, 28, 28))
    # best_model.summary()


def build_neural_network_old(hp):
    INPUT_DIM = 29
    model = Sequential()

    # Tune the number of layers
    # for i in range(hp.Int("num_layers", 1, 3)):
    #     model.add(Dense(units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
    #                     input_dim=input_dim,
    #                     kernel_initializer='random_normal',
    #                     activation='relu'))
    #
    #     if hp.Boolean("dropout"):
    #         model.add(Dropout(rate=0.25))

    dropout_prob = hp.Float("dropout_prob", min_value=0.1, max_value=0.5, step=0.1)

    # Hidden Layer 1
    model.add(Dense(units=hp.Int("First Hidden Layer", min_value=200, max_value=400, step=32),
                    input_dim=INPUT_DIM,
                    kernel_initializer='random_normal',
                    activation='relu'))

    if apply_dropout:
        model.add(Dropout(rate=dropout_prob))

    # Hidden Layer 2
    model.add(Dense(units=hp.Int("Second Hidden Layer", min_value=100, max_value=300, step=32),
                    kernel_initializer='random_normal',
                    activation='relu'))

    if apply_dropout:
        model.add(Dropout(rate=dropout_prob))

    # Output Layer
    model.add(Dense(1, activation='linear'))

    # Learning rate choices: 0.0001, 0.001, 0.01
    learning_rate = hp.Float("lr", min_value=1e-3, max_value=1e-2, sampling="log")
    sgd = SGD(learning_rate=learning_rate, clipvalue=0.5)

    model.compile(loss='mse', optimizer=sgd, metrics=[RSquare()])

    return model


def test_final_model(Xmat_train, Y_train, Xmat_val, Y_val, Xmat_test, Y_test):

    model = Sequential()
    model.add(Dense(352, input_dim=len(Xmat_train.columns), kernel_initializer='random_normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(224, kernel_initializer='random_normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    # set up an SGD optimizer with custom learning rate
    sgd = SGD(learning_rate=0.005579936781430615, clipvalue=0.5)

    model.compile(loss='mse', optimizer=sgd, metrics=[RSquare()])

    model.summary()

    num_of_rows = len(Xmat_train)

    model.fit(Xmat_train,
                Y_train,
                batch_size=64,
                epochs=441,
                validation_data=(Xmat_val, Y_val))

    # evaluate the model using test set
    score = model.evaluate(Xmat_test, Y_test)
    print('Test loss:', score[0])
    print('Test R^2:', score[1])

    # make a prediction
    Y_test_predict = model.predict(Xmat_test)
    Y_test_predict = pd.Series(Y_test_predict.transpose()[0], name='pred')
    Y_test = Y_test.reset_index(drop=True)
    df = pd.concat([Y_test, Y_test_predict], axis=1)

    df['absolute_diff'] = abs(df['final_view'] - df['pred'])

    df['percent_error'] = df['absolute_diff'] / df['final_view']

    df = df.sort_values(by=['final_view'], ascending=False)

    # print(df.to_string())

    df = df.sort_values(by=['pred'], ascending=False)

    # print(df.to_string())

    # print("avg percent error:", df['percent_error'].mean())



def build_neural_network(hp):
    INPUT_DIM = 29
    model = Sequential()

    # Tune the number of layers
    # for i in range(hp.Int("num_layers", 1, 3)):
    #     model.add(Dense(units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
    #                     input_dim=input_dim,
    #                     kernel_initializer='random_normal',
    #                     activation='relu'))
    #
    #     if hp.Boolean("dropout"):
    #         model.add(Dropout(rate=0.25))

    apply_dropout = hp.Boolean("dropout")
    dropout_prob = hp.Float("dropout_prob", min_value=0.1, max_value=0.5, step=0.1)

    # Hidden Layer 1
    model.add(Dense(units=hp.Int("First Hidden Layer", min_value=32, max_value=512, step=32),
                    input_dim=INPUT_DIM,
                    kernel_initializer='random_normal',
                    activation='relu'))

    if apply_dropout:
        model.add(Dropout(rate=dropout_prob))

    # Hidden Layer 2
    model.add(Dense(units=hp.Int("Second Hidden Layer", min_value=16, max_value=256, step=16),
                    kernel_initializer='random_normal',
                    activation='relu'))

    if apply_dropout:
        model.add(Dropout(rate=dropout_prob))

    # Output Layer
    model.add(Dense(1, activation='linear'))

    # Learning rate choices: 0.0001, 0.001, 0.01
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    sgd = SGD(learning_rate=learning_rate, clipvalue=0.5)

    model.compile(loss='mse', optimizer=sgd, metrics=[RSquare()])

    return model


def main():

    seed_value = 42

    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    # Set Tensorflow global random seed
    set_seed(42)

    # 5. Configure a new global `tensorflow` session

    # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    # K.set_session(sess)
    # for later versions:
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)



    # All of those are pandas objects
    Xmat_train_and_val, Y_train_and_val, Xmat_train, Xmat_val, Xmat_test, Y_train, Y_val, Y_test = load_bilibili_data(abalation=True)

    # ns_Xmat_train_and_val, ns_Y_train_and_val, ns_Xmat_train, ns_Xmat_val, ns_Xmat_test, ns_Y_train, ns_Y_val, ns_Y_test = load_bilibili_data(standardize=False)

    # Create a list where train data indices are -1 and validation data indices are 0
    split_index = [-1 if x in Xmat_train.index else 0 for x in Xmat_train_and_val.index]

    # grid_search_neural_network(Xmat_train, Y_train, Xmat_val, Y_val)
    print(Xmat_train.columns)

    # fit_linear_regression(Xmat_train, Y_train, Xmat_val, Y_val)
    #
    # fit_best_random_forest(Xmat_train, Y_train, Xmat_val, Y_val, Xmat_train_and_val, Y_train_and_val, split_index)

    # fit_polynomial_regression(ns_Xmat_train_and_val, ns_Y_train_and_val, split_index)

    # fit_random_forest(Xmat_train_and_val, Y_train_and_val, split_index)

    # fit_neural_network_keras(Xmat_train, Y_train, Xmat_val, Y_val)

    test_final_model(Xmat_train, Y_train, Xmat_val, Y_val, Xmat_test, Y_test)



main()
