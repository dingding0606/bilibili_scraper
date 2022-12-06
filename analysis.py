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
from sklearn.ensemble import RandomForestRegressor

def main():

    # All of those are Numpy objects
    Xmat_train, Xmat_val, Xmat_test, Y_train, Y_val, Y_test = load_bilibili_data()

    # ==============
    # BASELINE MODEL
    # ==============
    baseline_model = LinearRegression()
    baseline_model.fit(Xmat_train, Y_train)

    # Obtain the coefficient of determination by calling the model with the score() function, then print the coefficient:
    r_squared_train = baseline_model.score(Xmat_train, Y_train)
    print('R-sqaured on training set:', r_squared_train)

    r_squared_val = baseline_model.score(Xmat_val, Y_val)
    print('R-sqaured on validation set:', r_squared_val)


    # =====================
    # POLYNOMIAL REGRESSION
    # =====================

    # steps = [
    #     ('poly', PolynomialFeatures(degree=2)),
    #     ('model', Ridge(alpha=700)) #Lasso(alpha=0.9, max_iter=10000, fit_intercept=True))
    # ]
    #
    # pipeline = Pipeline(steps)
    #
    # pipeline.fit(Xmat_train, Y_train)
    #
    # print('Training score: {}'.format(pipeline.score(Xmat_train, Y_train)))
    # print('Validation score: {}'.format(pipeline.score(Xmat_val, Y_val)))
    for i in range(29, 40):
        print("max_features = ", i)
        forest = RandomForestRegressor(n_estimators=100, max_features=i)
        forest.fit(Xmat_train, Y_train)
        print('Training score: {}'.format(forest.score(Xmat_train, Y_train)))
        print('Validation score: {}'.format(forest.score(Xmat_val, Y_val)))
        print("\n")
    # poly_reg = PolynomialFeatures(degree=2)
    # Xmat_train_poly = poly_reg.fit_transform(Xmat_train)
    # poly_reg_model = LinearRegression()
    # poly_reg_model.fit(Xmat_train_poly, Y_train)
    #
    # r_squared_train = poly_reg_model.score(Xmat_train_poly, Y_train)
    # print('R-sqaured on training set:', r_squared_train)
    #
    # r_squared_val = poly_reg_model.score(poly_reg.fit_transform(Xmat_val), Y_val)
    # print('R-sqaured on validation set:', r_squared_val)


main()
