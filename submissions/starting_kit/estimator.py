import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

class Classifier(BaseEstimator):
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        y_pred = self.model.predict_proba(X)
        return y_pred


def compute_rolling(X_df, feature, time_window, aggreg, center=False):
    """
    For a given dataframe, compute the standard deviation over
    a defined period of time (time_window) of a defined feature

    Parameters
    ----------
    X : dataframe
    feature : str
        feature in the dataframe we wish to compute the rolling std from
    time_window : str
        string that defines the length of the time window passed to `rolling`
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """
    name = "_".join([feature, time_window, "std"])
    rolling = X_df[feature].rolling(time_window, center=center)
    if aggreg == "std":
        series = rolling.std()
    elif aggreg == "mean":
        series = rolling.std()
    elif aggreg == "min":
        series = rolling.min()
    elif aggreg == "max":
        series = rolling.max()
    series = series.ffill().bfill().astype(X_df[feature].dtype)
    return series


def compute_expanding(X_df, feature, aggreg, center=False):
    """
    For a given dataframe, compute the standard deviation over
    a defined period of time (time_window) of a defined feature

    Parameters
    ----------
    X : dataframe
    feature : str
        feature in the dataframe we wish to compute the rolling std from
    time_window : str
        string that defines the length of the time window passed to `rolling`
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """
    name = "_".join([feature, "std"])
    rolling = X_df[feature].expanding()
    if aggreg == "std":
        series = rolling.std()
    elif aggreg == "mean":
        series = rolling.std()
    elif aggreg == "min":
        series = rolling.min()
    elif aggreg == "max":
        series = rolling.max()
    series = series.ffill().bfill().astype(X_df[feature].dtype)
    return series


def compute_lagged_feature(X_df, feature, max_lag=100):
    """
    For a given dataframe, compute the shifted feature by the best lag

    Parameters
    ----------
    X : dataframe
    feature : str
        feature in the dataframe we wish to compute the shifted values
    max_lag : int
        maximum lags from which we are going to choose the best lag
    """
    name = "_".join([feature, "lagged"])
    autocorr = [X_df[feature].corr(X_df[feature].shift(lag)) for lag in range(1, max_lag + 1)]
    best_lag = np.array(autocorr).argmax() + 1
    series = X_df[feature].shift(best_lag)
    series = series.ffill().bfill().astype(X_df[feature].dtype)
    return series


class FeatureExtractor(BaseEstimator):
    def fit(self, X, y):
        return self

    def transform(self, X):
        window = "2h"
        dict_cols = {}
        X = X.drop(columns='Range F 14')
        X["beta_rolling_std"] = compute_rolling(X, 'Beta', window, "std")
        # X = X.drop(columns=['By_rms', 'Bx_rms', 'Bz_rms', 'Range F 1', 'Range F 8', 'Range F 9', 'Range F 12',
        #                     'Range F 14', 'Range F 2','Range F 3','Range F 4','Range F 5','Range F 6',
        #                     'Range F 7','Range F 11'])

        # for column in X.columns:
        #     # # Windowing
        #     dict_cols[f"{column}_rolling_std"] = compute_rolling(X, column, window, "std")
        #     dict_cols[f"{column}_rolling_mean"] = compute_rolling(X, column, window, "mean")
        #     # dict_cols[f"{column}_rolling_min"] = compute_rolling(X, column, window, "min")
        #     # dict_cols[f"{column}_rolling_max"] = compute_rolling(X, column, window, "max")
        #
        #     # Lags
        #     dict_cols[f"{column}_lags"] = compute_lagged_feature(X, column)
        #
        #     # # Expansion
        #     # dict_cols[f"{column}_expansion_mean"] = compute_expanding(X, column, "mean")
        #     # dict_cols[f"{column}_expansion_std"] = compute_expanding(X, column, "std")
        #     # dict_cols[f"{column}_expansion_min"] = compute_expanding(X, column, "min")
        #     # dict_cols[f"{column}_expansion_max"] = compute_expanding(X, column, "max")
        # return pd.concat([X, pd.DataFrame(dict_cols)], axis=1)
        return X

def get_estimator():

    feature_extractor = FeatureExtractor()
    #estimator = SVC(kernel="linear")
    #selector = RFECV(estimator, step=1, cv=5)
    #selector = SelectFromModel(estimator=estimator)
    #classifier = LogisticRegression(max_iter=1000)
    #classifier = RandomForestClassifier(random_state=42)
    #classifier = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    classifier = LogisticRegression(max_iter=1000, penalty='l1', solver='saga')
    pipe = make_pipeline(feature_extractor, StandardScaler(), classifier, verbose=True)
    return pipe
