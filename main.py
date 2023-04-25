import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import linregress, randint, uniform
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from mlutils.regression import (
    regression_box_plot,
    regression_cdf_plot,
    regression_cooks_calculate,
    regression_cooks_plot,
    regression_density_plot,
    regression_df,
    regression_hyperparams,
    regression_importance_calculate,
    regression_importance_plot,
    regression_leverage_calculate,
    regression_leverage_plot,
    regression_metrics,
    regression_prob_plot,
    regression_residual_plot,
    regression_scale_loc_plot,
    regression_scatter_plot,
    scoring,
)

REFIT = "rmse"
SEED = 1990
FOLDS = 10

X, y = make_regression(n_samples=1000, n_features=4, n_informative=2, noise=1, random_state=SEED)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=SEED)

# SIMPLE FIT
rgr = LGBMRegressor()
rgr.fit(X_train, y_train)
y_train_pred = rgr.predict(X_train)

# RandomizedSearchCV FIT
estimator = ElasticNet()
search = RandomizedSearchCV(
    estimator=estimator,
    param_distributions=regression_hyperparams(estimator),
    scoring=scoring,
    refit=REFIT,
    cv=FOLDS,
    return_train_score=True,
    random_state=SEED,
)

start = time.time()
search.fit(X_train, y_train)
end = time.time()
print(f"Elapsed time {end-start} seconds")

y_train_pred = search.best_estimator_.predict(X_train)

# RESULTS

pd.DataFrame(search.cv_results_)

regression_leverage_calculate(X_train)
regression_cooks_calculate(X_train, y_train, y_train_pred)

regression_df(X_train, y_train, y_train_pred)
regression_metrics(y_train, y_train_pred)
regression_prob_plot(y_train, y_train_pred)
regression_residual_plot(y_train, y_train_pred)
regression_scale_loc_plot(y_train, y_train_pred)
regression_cooks_plot(X_train, y_train, y_train_pred)
regression_leverage_plot(X_train)
regression_scatter_plot(y_train, y_train_pred)
regression_density_plot(y_train, y_train_pred)
regression_cdf_plot(y_train, y_train_pred)
regression_box_plot(y_train, y_train_pred)

a, b, c = regression_importance_calculate(search.best_estimator_)
a, b, c = regression_importance_calculate(rgr)

regression_importance_plot(rgr)
regression_importance_plot(search.best_estimator_)
