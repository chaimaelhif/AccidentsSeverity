import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler


class FeaturePreprocessing(BaseEstimator):
    """
    Format and remove features
    """
    def __init__(self):
        self.columns_to_drop = []

    def fit(self, X, y):
        na_columns = X.columns[X.isnull().sum()/X.shape[0] > 0.9]
        self.columns_to_drop.extend(na_columns)

        columns_to_drop = ['id_vehicule',
                           'Num_Acc',
                           'adr',
                           'voie',
                           'an',
                           'hrmn']
        self.columns_to_drop.extend(columns_to_drop)

        cov_matrix = X.corr(numeric_only=True).abs()
        upper_tri = cov_matrix.where(np.triu(np.ones(cov_matrix.shape),
                                             k=1).astype(bool))
        columns_to_drop = [column
                           for column in upper_tri.columns
                           if any(upper_tri[column] > 0.85)
                           ]
        self.columns_to_drop.extend(columns_to_drop)
        return self

    def transform(self, X):
        # Transform hrmn
        X = X.copy()
        X['heure'] = [int(str(i).split(':')[0]) for i in X['hrmn']]
        X['hour_sin'] = np.sin(X.heure*(2.*np.pi/24))
        X['minutes'] = [int(str(i).split(':')[1]) for i in X['hrmn']]
        X['hour_cos'] = np.cos(X.heure*(2.*np.pi/24))
        X['minute_sin'] = np.sin(X.minutes*(2.*np.pi/60))
        X['minute_cos'] = np.cos(X.minutes*(2.*np.pi/60))

        del X['heure']
        del X['minutes']

        # Format other features
        X["lat"] = [float(str(i).replace(",", ".")) for i in X["lat"]]
        X["long"] = [float(str(i).replace(",", ".")) for i in X["long"]]
        X['larrout'] = X['larrout'].str.replace(",", ".").astype(float)

        # Remove columns
        X = X.drop(columns=self.columns_to_drop)

        # Convert strictly integer features to int
        float_col = X.select_dtypes("float").columns
        strict_int = [col
                      for col in float_col
                      if (X[col] == X[col].astype(int, errors='ignore')).all()
                      ]
        X[strict_int] = X[strict_int].astype(int, errors='ignore')

        # Convert features with more than unique values to float
        num_col = X.select_dtypes(exclude="object").columns
        num_col = [col for col in num_col if X[col].nunique() > 40]
        X[num_col] = X[num_col].astype("float", errors='ignore')
        return X


class CountOrdinalEncoder(OrdinalEncoder):
    """Encode categorical features as an integer array
    usint count information.
    """
    def __init__(self, categories='auto', dtype=np.float64):
        self.categories = categories
        self.dtype = dtype

    def fit(self, X, y=None):
        """Fit the OrdinalEncoder to X.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to determine the categories of each feature.

        Returns
        -------
        self
        """
        self.handle_unknown = 'use_encoded_value'
        self.unknown_value = np.nan
        super().fit(X)
        X_list, _, _ = self._check_X(X)
        # now we'll reorder by counts
        for k, cat in enumerate(self.categories_):
            counts = []
            for c in cat:
                counts.append(np.sum(X_list[k] == c))
            order = np.argsort(counts)
            self.categories_[k] = cat[order]
        return self


def get_estimator():
    feature_preproc = FeaturePreprocessing()

    transformer = ColumnTransformer(
        [
            ("Num_col",
             StandardScaler(),
             make_column_selector(dtype_include=float)
             ),
            ("Object_col",
             CountOrdinalEncoder(),
             make_column_selector(dtype_include=object)
             )
        ],
        remainder="passthrough"
    )
    classifier = HistGradientBoostingClassifier(random_state=42)
    pipe = Pipeline(steps=[
        ('feature_preprocessing', feature_preproc),
        ('transformer', transformer),
        ('classifier', classifier)
    ])
    return pipe
