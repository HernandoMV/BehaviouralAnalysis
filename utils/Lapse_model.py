#!/usr/bin/env bash

import autograd
import autograd.numpy as np
import autograd.scipy.stats as st
import autograd.scipy.special as sp
import scipy.optimize as opt

import pandas as pd
import matplotlib.pyplot as plt


def predict_lapse_model(X, alpha, beta, slope, intercept):
    """predict from a logistic regression model with lapse component"""
    x = slope * X + intercept
    return alpha + (beta - alpha) * sp.expit(-x)


def fit_lapse_model(X, y, n_restart=50, seed=12345):
    """fit a logistic regression model with lapse component"""

    def transform(x):
        return (
            st.norm.cdf(x[0]), st.norm.cdf(x[0] + np.exp(x[1])), x[2], x[3]
        )

    def loss(params):
        preds = predict_lapse_model(X, *transform(params))
        return -np.mean(y * np.log(preds) + (1. - y) * np.log(1 - preds))

    value_n_grad = autograd.value_and_grad(loss)

    np.random.seed(seed)  # fixed seed for reproducibility
    fun = np.inf
    for i in range(n_restart):
        x0 = np.random.randn(4)
        new_res = opt.minimize(value_n_grad, x0, jac=True)
        if new_res.fun < fun:
            fun, res = new_res.fun, new_res

    return transform(res.x), res


if __name__ == "__main__":
    # # generate fake data to test
    # X = np.random.rand(200)
    # y_data = predict_lapse_model(X, 0.1, 0.8, 10., -5)
    # y = y_true > 0.5  # TODO sample, not thresholding

    # load Hernando's data
    dset = pd.read_pickle('WithLoveForMaxime.pkl')
    X = np.array(dset.Difficulty)
    y_data = dset.groupby('Difficulty').agg(lambda x: np.mean(x == 2))
    y = np.array(dset.Choice == 2)

    # fit the model parameters and print optimization results
    params, res = fit_lapse_model(X, y)
    print(res)

    # display results
    plt.figure()
    plt.plot(y_data, 'o')
    xs = np.linspace(min(X) - 0.5, max(X) + 0.5)
    plt.plot(xs, predict_lapse_model(xs, *params))
    plt.show()
