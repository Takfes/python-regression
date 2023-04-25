import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from lightgbm import LGBMRegressor
from scipy.stats import linregress, randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    explained_variance_score,
    make_scorer,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_gamma_deviance,
    mean_poisson_deviance,
    mean_squared_error,
    mean_squared_log_error,
    mean_tweedie_deviance,
    median_absolute_error,
    r2_score,
)
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

# TODO
"""
Partial regression plot: A partial regression plot is a plot of the residuals from a regression of the response variable on a single predictor variable versus the residuals from a regression of the same response variable on all of the predictor variables. It can be used to assess the effect of individual predictor variables on the response variable.
Component-residual plot: A component-residual plot is a plot of the residuals from a regression of the response variable on a single predictor variable versus the values of the predictor variable. It can be used to identify nonlinearity or heteroscedasticity in the relationship between the predictor variable and the response variable.
"""


FIGSIZE = (8, 5)


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


symmetric_mean_absolute_percentage_scorer = make_scorer(
    symmetric_mean_absolute_percentage_error, greater_is_better=False
)

scoring = {
    "explained_variance": make_scorer(explained_variance_score),
    "r2": make_scorer(r2_score),
    "rmse": "neg_root_mean_squared_error",
    "mse": "neg_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "mad": "neg_median_absolute_error",
    # "msle": "neg_mean_squared_log_error",
    "max_error": "max_error",
    "mean_absolute_percentage_error": make_scorer(mean_absolute_percentage_error),
    "symmetric_mean_absolute_percentage_error": make_scorer(
        symmetric_mean_absolute_percentage_error
    ),
    # "mean_poisson_deviance": make_scorer(mean_poisson_deviance),
    # "mean_gamma_deviance": make_scorer(mean_gamma_deviance),
    # "mean_tweedie_deviance": make_scorer(mean_tweedie_deviance),
}


def regression_hyperparams(estimator):
    if estimator.__class__.__name__ == "LinearRegression":
        # Define the hyperparameter grid for LinearRegression
        param_grid = {}

    elif estimator.__class__.__name__ == "Ridge":
        # Define the hyperparameter grid for Ridge
        param_grid = {
            "alpha": uniform(0, 10),
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
        }

    elif estimator.__class__.__name__ == "Lasso":
        # Define the hyperparameter grid for Lasso
        param_grid = {
            "alpha": uniform(0, 10),
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "selection": ["cyclic", "random"],
        }

    elif estimator.__class__.__name__ == "ElasticNet":
        # Define the hyperparameter grid for ElasticNet
        param_grid = {
            "alpha": uniform(0, 10),
            "l1_ratio": uniform(0, 1),
            "fit_intercept": [True, False],
            "selection": ["cyclic", "random"],
        }

    elif estimator.__class__.__name__ == "RandomForestRegressor":
        # Define the hyperparameter grid for RandomForestRegressor
        param_grid = {
            "n_estimators": randint(10, 500),
            "max_depth": [None] + list(range(5, 50, 5)),
            "max_features": ["auto", "sqrt"],
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 20),
            "bootstrap": [True, False],
        }

    elif estimator.__class__.__name__ == "XGBoostRegressor":
        # Define the hyperparameter grid for XGBoostRegressor
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 1],
            "subsample": [0.5, 0.75, 1],
            "colsample_bytree": [0.5, 0.75, 1],
            "gamma": [0, 0.1, 1],
            "reg_alpha": [0, 0.1, 1],
            "reg_lambda": [0, 0.1, 1],
            "min_child_weight": [1, 3, 5],
        }
    elif estimator.__class__.__name__ == "LGBMRegressor":
        # Define the hyperparameter grid for LGBMRegressor
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 1],
            "subsample": [0.5, 0.75, 1],
            "colsample_bytree": [0.5, 0.75, 1],
            "reg_alpha": [0, 0.1, 1],
            "reg_lambda": [0, 0.1, 1],
            "min_child_weight": [1, 3, 5],
        }
    elif estimator.__class__.__name__ == "KNeighborsRegressor":
        # Define the hyperparameter grid for KNeighborsRegressor
        param_grid = {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": [10, 20, 30, 40, 50],
            "p": [1, 2],
        }

    else:
        raise ValueError(f"Unsupported estimator: {estimator.__class__.__name__}")

    return param_grid


def regression_importance_calculate(estimator):
    """
    Extracts the coefficients or feature importances from a linear or tree-based model.

    Args:
    - estimator: A trained regression estimator object (e.g., LinearRegression, RandomForestRegressor)

    Returns:
    - coefficients: The model coefficients or feature importances, depending on the estimator
    """
    modelname = estimator.__class__.__name__
    if hasattr(estimator, "coef_"):
        importance = estimator.coef_
        importance_type = "Coefficients"
    elif hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
        importance_type = "Feature Importance"
    else:
        print("Model does not have coefficients or feature importances.")
        return None, None, None
    return importance, importance_type, modelname


def regression_importance_plot(
    estimator, feature_names=None, figsize=(8, 6), show=True, save_fig=None
):
    """
    Plot the feature importances or coefficients of a regression estimator.

    Args:
    - estimator: A trained regression estimator object (e.g., LinearRegression, RandomForestRegressor)
    - feature_names: A list of feature names
    - figsize: A tuple indicating the figure size
    - show: A boolean indicating whether to show the plot or not
    - save_fig: A string indicating the file name to save the plot

    Returns:
    - None
    """
    importances, importance_type, model_name = regression_importance_calculate(estimator)
    if importances is None:
        return

    # Generate feature names if not provided
    if feature_names is None:
        n_features = len(importances)
        feature_names = [f"Feature_{i}" for i in range(n_features)]

    # Sort importances in descending order
    indices = np.argsort(importances)[::-1]
    importances_sorted = [importances[i] for i in indices]
    feature_names_sorted = [feature_names[i] for i in indices]

    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(feature_names_sorted, importances_sorted)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importances")

    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight")

    if show:
        plt.show()


def regression_metrics(y_true, y_pred):
    metrics = {}
    metrics["Explained Variance"] = explained_variance_score(y_true, y_pred)
    metrics["R2"] = r2_score(y_true, y_pred)
    metrics["RMSE"] = mean_squared_error(y_true, y_pred, squared=False)
    metrics["MSE"] = mean_squared_error(y_true, y_pred)
    metrics["MAE"] = mean_absolute_error(y_true, y_pred)
    metrics["MAD"] = median_absolute_error(y_true, y_pred)
    # metrics["MSLE"] = mean_squared_log_error(y_true, y_pred)
    metrics["MaxError"] = max_error(y_true, y_pred)
    metrics["MAPE"] = mean_absolute_percentage_error(y_true, y_pred)
    metrics["sMAPE"] = symmetric_mean_absolute_percentage_error(y_true, y_pred)

    # metrics["Mean Poisson Deviance"] = mean_poisson_deviance(y_true, y_pred)
    # metrics["Mean Gamma Deviance"] = mean_gamma_deviance(y_true, y_pred)
    # metrics["Mean Tweedie Deviance"] = mean_tweedie_deviance(y_true, y_pred)
    return metrics


def regression_df(X, y_true, y_pred):
    cooks, _, _ = regression_cooks_calculate(X, y_true, y_pred)
    leverage, _ = regression_leverage_calculate(X)
    df = (
        pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "cooks": cooks, "leverage": leverage})
        .assign(error=lambda x: x["y_true"] - x["y_pred"])
        .assign(abs_error=lambda x: abs(x["error"]))
    )
    return df


def regression_prob_plot(y_true, y_pred, figsize=FIGSIZE, show=True, save_fig=None):
    """
    Residual plot: A residual plot is a scatter plot of the residuals
    (i.e., the differences between the actual and predicted values)
    versus the predicted values. It can be used to identify patterns
    in the residuals, such as nonlinearity, heteroscedasticity,
    and outliers.
    """
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=figsize)
    stats.probplot(residuals, plot=ax, fit=True)
    ax.set_title("Probability Plot")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight")
    if show:
        plt.show()


def regression_residual_plot(y_true, y_pred, figsize=FIGSIZE, show=True, save_fig=None):
    """
    Plot the residuals of a regression model.
    Normal probability plot: A normal probability plot is a plot of
    the residuals versus their expected values under the assumption
    that the residuals are normally distributed. It can be used to
    assess the normality of the residuals.

    Args:
    - y_true (numpy array): array of true target values
    - y_pred (numpy array): array of predicted target values
    - figsize (tuple): size of the figure (default: FIGSIZE)
    - show (bool): whether to display the plot (default: True)
    - save_fig (str): filename to save the plot (default: None)

    Returns:
    - None
    """
    residuals = y_true - y_pred
    plt.figure(figsize=figsize)
    plt.scatter(y_pred, residuals, c="b", alpha=0.5)
    plt.axhline(y=0, color="r", linestyle="-")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight")
    if show:
        plt.show()


def regression_scale_location_plot(y_true, y_pred, figsize=FIGSIZE, show=True, save_fig=None):
    """
    Function to create a Scale-Location plot for a regression model.
    Scale-location plot: A scale-location plot is a plot of the
    square root of the absolute residuals versus the predicted values.
    It can be used to assess the homoscedasticity of the residuals.

    Parameters:
    y_true (array-like): true target values
    y_pred (array-like): predicted target values
    figsize (tuple, optional): figure size, default FIGSIZE
    show (bool, optional): whether to display the plot, default True
    save_fig (str, optional): file name to save the figure, default None

    Returns:
    fig (matplotlib figure object): the generated plot figure
    """
    # Calculate the residuals and standardized residuals
    residuals = y_true - y_pred
    std_residuals = residuals / np.sqrt(np.mean(np.square(residuals)))

    # Calculate the predicted values and square root of absolute residuals
    pred_values = y_pred
    sqrt_abs_res = np.sqrt(np.abs(residuals))

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(pred_values, sqrt_abs_res)

    # Add regression line to the plot
    x_min, x_max = ax.get_xlim()
    x_values = np.linspace(x_min, x_max, 100)
    y_values = np.sqrt(np.abs(np.mean(residuals))) * np.ones(100)
    ax.plot(x_values, y_values, "r--")

    # Add title and axis labels
    ax.set_title("Scale-Location Plot")
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("$\sqrt{|Standardized Residuals|}$")

    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def regression_scale_loc_plot(y_true, y_pred, figsize=FIGSIZE, show=True, save_fig=None):
    """
    Scale-location plot: A scale-location plot is a plot of the
    square root of the absolute residuals versus the predicted values.
    It can be used to assess the homoscedasticity of the residuals.

    Args:
    - y_true (numpy array): array of true target values
    - y_pred (numpy array): array of predicted target values
    - figsize (tuple, optional): size of the figure. Default is FIGSIZE
    - show (bool, optional): whether to display the plot. Default is True
    - save_fig (str, optional): name of the file to save the plot. Default is None

    Returns:
    - None
    """
    residuals = y_true - y_pred
    sqrt_abs_resid = np.sqrt(np.abs(residuals))
    fitted_vals = y_pred

    plt.figure(figsize=figsize)
    plt.scatter(fitted_vals, sqrt_abs_resid)
    plt.xlabel("Fitted Values")
    plt.ylabel("$\sqrt{|Residuals|}$")
    plt.title("Scale-Location Plot")

    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight")
    if show:
        plt.show()


def regression_cooks_calculate(X, y, y_pred):
    residuals = y - y_pred
    h = np.dot(X, np.dot(np.linalg.inv(np.dot(X.T, X)), X.T))
    leverage = np.diag(h)
    cooks_dist = (
        residuals**2 / (X.shape[1] * np.var(residuals)) * (leverage / (1 - leverage) ** 2)
    )
    n, p = X.shape
    influence_threshold = 4 / (n - p - 1)
    return (
        cooks_dist,
        influence_threshold,
        np.mean(np.where(cooks_dist >= influence_threshold, 1, 0)),
    )


def regression_cooks_plot(X, y, y_pred, figsize=FIGSIZE, show=True, save_fig=None):
    """
    Cook's distance plot: Cook's distance is a measure of the
    influence of individual observations on the regression coefficients.
    A Cook's distance plot is a plot of Cook's distances versus the
    observation index. It can be used to identify influential
    observations that may be driving the results.
    """
    cooks_dist, threshold, outlier_rate = regression_cooks_calculate(X, y, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    ax.stem(cooks_dist, markerfmt=" ")
    ax.axhline(y=threshold, linestyle="--", color="r")
    ax.set(
        title=f"Cook's Distance Plot - Outlier Rate {outlier_rate:.2%}",
        xlabel="Index",
        ylabel="Distance",
    )
    ax.tick_params(axis="x", labelrotation=45)
    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight")
    if show:
        plt.show()


def regression_leverage_calculate(X):
    # Calculate hat matrix
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    # Calculate leverage values
    leverage = np.diagonal(H)
    # Calculate threshold
    threshold = 2 * np.mean(leverage)
    return H.diagonal(), threshold


def regression_leverage_plot(X, figsize=FIGSIZE, show=True, save_fig=None):
    """
    Creates a leverage plot for a set of predictors.
    Leverage plot: A leverage plot is a plot of the leverage values
    (i.e., the diagonal elements of the hat matrix) versus the
    observation index. It can be used to identify observations
    that are outliers in terms of their predictor values.
    """

    # Calculate leverage values
    leverage, threshold = regression_leverage_calculate(X)

    # Plot results
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(range(len(leverage)), leverage, s=30, c="b", marker="o")
    ax.axhline(y=threshold, linestyle="--", color="r")
    ax.set(title="Leverage Plot", xlabel="Observation Index", ylabel="Leverage Value")
    ax.tick_params(axis="x", labelrotation=45)
    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight")
    if show:
        plt.show()


def regression_scatter_plot(y_true, y_pred, figsize=FIGSIZE, show=True, save_fig=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_true, y_pred)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "--k", alpha=0.5)

    # best fit line
    slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)
    line = slope * y_true + intercept
    ax.plot(y_true, line, "r", label="y={:.2f}x+{:.2f}".format(slope, intercept))

    ax.set(title="Actual vs Predicted Plot", xlabel="Actual", ylabel="Predicted")
    ax.legend()
    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight")
    if show:
        plt.show()


def regression_density_plot(y_true, y_pred, figsize=FIGSIZE, show=True, save_fig=None):
    fig, ax = plt.subplots(figsize=figsize)
    sns.kdeplot(y_true, ax=ax, label="y_true", fill=True, alpha=0.2)
    sns.kdeplot(y_pred, ax=ax, label="y_pred", fill=True, alpha=0.2)
    ax.set(title="Density Plot of y_true vs y_pred", xlabel="Value", ylabel="Density")
    plt.legend()
    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight")
    if show:
        plt.show()


def regression_cdf_plot(y_true, y_pred, figsize=FIGSIZE, show=True, save_fig=None):
    df = pd.DataFrame(
        {
            "y": np.hstack([y_true, y_pred]),
            "type": ["y_true" for _ in y_true] + ["y_pred" for _ in y_pred],
        }
    )
    # return df
    fig, ax = plt.subplots(figsize=figsize)
    sns.kdeplot(
        data=df,
        x="y",
        hue="type",
        label=["y_true", "y_pred"],
        cumulative=True,
        common_norm=False,
        common_grid=True,
    )
    ax.set(title="Density Plot of y_true vs y_pred", xlabel="Value", ylabel="Density")
    plt.legend()
    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight")
    if show:
        plt.show()


def regression_box_plot(y_true, y_pred, figsize=FIGSIZE, show=True, save_fig=None, h_adj=0.1):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})

    fig, ax = plt.subplots(figsize=figsize)
    df.plot(kind="box", ax=ax)
    ax.set(title="Box Plot of y_true vs y_pred", ylabel="Value")

    # Adding annotations
    for i, col in enumerate(["y_true", "y_pred"]):
        quartile1, quartile3 = np.percentile(df[col], [25, 75])
        iqr = quartile3 - quartile1
        lower_fence = quartile1 - 1.5 * iqr
        upper_fence = quartile3 + 1.5 * iqr
        median = np.median(df[col])
        ax.text(i + 1 + h_adj, lower_fence, f"Lower Fence: {lower_fence:.2f}")
        ax.text(i + 1 + h_adj, upper_fence, f"Upper Fence: {upper_fence:.2f}")
        ax.text(i + 1 + h_adj, quartile1 - iqr / 2, f"Q1: {quartile1:.2f}")
        ax.text(i + 1 + h_adj, quartile3 + iqr / 2, f"Q3: {quartile3:.2f}")
        ax.text(i + 1 + h_adj, median, f"Median: {median:.2f}")

    if save_fig:
        plt.savefig(save_fig, bbox_inches="tight")
    if show:
        plt.show()
