from __future__ import annotations
import time
import pathlib
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Any, Literal
from retry import retry
from dataclasses import dataclass, asdict, field
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from sklearn.preprocessing import PowerTransformer as OriginalPowerTransformer
from scipy import optimize
from typing import cast

from pilot.c_ensemble import RandomForestCPilot
from pilot.Pilot import PILOT as PythonNumbaPILOT # Explicitly import Python/Numba version
from pilot import DEFAULT_DF_SETTINGS # Import if needed for default df_settings

class RobustPowerTransformer(OriginalPowerTransformer):
    """Apply a power transform featurewise to make data more Gaussian-like."""

    def _yeo_johnson_inverse_transform(self, x, lmbda):
        """Return inverse-transformed input x following Yeo-Johnson inverse transform."""
        x_inv = np.zeros_like(x)
        pos = x >= 0

        # when x >= 0
        if abs(lmbda) < np.spacing(1.0):
            x_inv[pos] = np.exp(x[pos]) - 1
        else:  # lmbda != 0
            # x_inv[pos] = np.power(x[pos] * lmbda + 1, 1 / lmbda) - 1  # Original line
            x_inv[pos] = np.exp(np.log1p(x[pos] * lmbda) / lmbda) - 1

        # when x < 0
        if abs(lmbda - 2) > np.spacing(1.0):
            # x_inv[~pos] = 1 - np.power(-(2 - lmbda) * x[~pos] + 1, 1 / (2 - lmbda))  # Original line
            x_inv[~pos] = 1 - np.exp(np.log1p(-(2 - lmbda) * x[~pos]) / (2 - lmbda))
        else:  # lmbda == 2
            x_inv[~pos] = 1 - np.exp(-x[~pos])

        return x_inv

    def _yeo_johnson_transform(self, x, lmbda):
        """Return transformed input x following Yeo-Johnson transform."""
        out = np.zeros_like(x)
        pos = x >= 0  # binary mask

        # when x >= 0
        if abs(lmbda) < np.spacing(1.0):
            out[pos] = np.log1p(x[pos])
        else:  # lmbda != 0
            # out[pos] = (np.power(x[pos] + 1, lmbda) - 1) / lmbda  # Original line
            out[pos] = (np.exp(np.log1p(x[pos]) * lmbda) - 1) / lmbda

        # when x < 0
        if abs(lmbda - 2) > np.spacing(1.0):
            # out[~pos] = -(np.power(-x[~pos] + 1, 2 - lmbda) - 1) / (2 - lmbda)  # Original line
            out[~pos] = -(np.exp(np.log1p(-x[~pos]) * (2 - lmbda)) - 1) / (2 - lmbda)
        else:  # lmbda == 2
            out[~pos] = -np.log1p(-x[~pos])

        return out

    def _yeo_johnson_optimize(self, x):
        """Find and return optimal lambda parameter of the Yeo-Johnson transform."""
        x_tiny = np.finfo(np.float64).tiny

        def _neg_log_likelihood(lmbda):
            """Compute the negative log likelihood of the observed data x."""
            x_trans = self._yeo_johnson_transform(x, lmbda)
            x_trans_var = x_trans.var()

            # Reject transformed data that would raise a RuntimeWarning in np.log
            if x_trans_var < x_tiny:
                return np.inf

            log_var = np.log(x_trans_var)
            # loglike = -n_samples / 2 * log_var  # Original line
            # loglike += (lmbda - 1) * (np.sign(x) * np.log1p(np.abs(x))).sum()  # Original line
            loglike = -0.5 * log_var
            loglike += (lmbda - 1) * (np.sign(x) * np.log1p(np.abs(x))).mean()
            loglike -= 0.01 * np.mean(np.abs(np.log(np.abs(x_trans_var[x_trans_var != 0]))))  # Regularize the exponent

            return -loglike

        # the computation of lambda is influenced by NaNs so we need to
        # get rid of them
        x = x[~np.isnan(x)]
        # choosing bracket -2, 2 like for boxcox
        lmbda = cast(float, optimize.brent(_neg_log_likelihood, brack=(-2, 2)))
        return lmbda


@dataclass
class Dataset:
    """A container for holding and managing all data versions and metadata.

    This class encapsulates the original features (X), one-hot encoded features,
    label-encoded features, the target variable (y), and associated metadata
    like encoders and categorical feature information.

    Attributes:
        id (int | str): A unique identifier for the dataset.
        name (str): The common name of the dataset.
        X (pd.DataFrame): The original feature matrix.
        X_oh_encoded (pd.DataFrame): The one-hot encoded feature matrix.
        X_label_encoded (pd.DataFrame): The label-encoded feature matrix.
        y (pd.Series): The target variable.
        cat_ids (list[int]): Column indices of categorical features in `X`.
        cat_names (list[str]): Column names of categorical features in `X`.
        oh_encoder (OneHotEncoder): The fitted one-hot encoder.
        label_encoders (dict[str, LabelEncoder]): A dictionary of fitted label encoders.
        rows_removed (int): The number of rows removed during preprocessing.
        cols_removed (int): The number of columns removed during preprocessing.
    """
    id: int | str
    name: str
    X: pd.DataFrame
    X_oh_encoded: pd.DataFrame
    X_label_encoded: pd.DataFrame
    y: pd.Series
    cat_ids: list[int]
    cat_names: list[str]
    oh_encoder: OneHotEncoder
    label_encoders: dict[str, LabelEncoder]
    rows_removed: int
    cols_removed: int

    def subset(self, idx: list[int]) -> Dataset:
        """Creates a new Dataset object containing a subset of the data.

        Args:
            idx (list[int]): A list of integer indices for row selection.

        Returns:
            Dataset: A new Dataset object with the specified subset of rows.
        """
        return Dataset(
            self.id,
            self.name,
            self.X.iloc[idx, :].copy(),
            self.X_oh_encoded.iloc[idx, :].copy(),
            self.X_label_encoded.iloc[idx, :].copy(),
            self.y.iloc[idx].copy(),
            self.cat_ids,
            self.cat_names,
            self.oh_encoder,
            self.label_encoders,
            self.rows_removed,
            self.cols_removed,
        )

    @property
    def categorical(self) -> np.ndarray:
        """Returns the indices of categorical features as a numpy array."""
        return np.array(self.cat_ids) if len(self.cat_ids) > 0 else np.array([-1])

    @property
    def n_features(self) -> int:
        """Returns the number of features in the original dataset `X`."""
        return self.X.shape[1]

    @property
    def n_samples(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.X)

    def summary(
        self,
        include_fields=[
            "id",
            "name",
            "n_samples",
            "n_features",
            "rows_removed",
            "cols_removed",
        ],
    ):
        """Generates a dictionary summarizing key dataset attributes.

        Args:
            include_fields (list[str]): A list of attribute names to include.

        Returns:
            dict: A dictionary containing the summary information.
        """
        return {field: getattr(self, field) for field in include_fields}

    def apply_transformer(self, feature_name: str, transformer: OriginalPowerTransformer | RobustPowerTransformer):
        """Applies a fitted transformer to a feature across all dataframes.

        This method transforms a specified column in-place in `X`, `X_oh_encoded`,
        and `X_label_encoded` and clips the results to avoid extreme values.

        Args:
            feature_name (str): The name of the column to transform.
            transformer (OriginalPowerTransformer | RobustPowerTransformer): The
                fitted transformer to apply.
        """
        self.X.loc[:, feature_name] = np.clip(
            np.nan_to_num(
                transformer.transform(self.X.loc[:, [feature_name]]).flatten(),
                posinf=0,
                neginf=0,
            ),
            -1e30,
            1e30,
        )
        self.X_oh_encoded.loc[:, feature_name] = np.clip(
            np.nan_to_num(
                transformer.transform(
                    self.X_oh_encoded.loc[:, [feature_name]]
                ).flatten(),
                posinf=0,
                neginf=0,
            ),
            -1e30,
            1e30,
        )
        self.X_label_encoded.loc[:, feature_name] = np.clip(
            np.nan_to_num(
                transformer.transform(
                    self.X_label_encoded.loc[:, [feature_name]]
                ).flatten(),
                posinf=0,
                neginf=0,
            ),
            -1e30,
            1e30,
        )


def fit_transformers(dataset: Dataset, problematic: bool):
    """Fits power transformers for all numerical columns in a dataset.

   Args:
       dataset (Dataset): The dataset containing the data to fit on.
       problematic (bool): If True, use `RobustPowerTransformer`. Otherwise,
           use the standard `OriginalPowerTransformer`.

   Returns:
       dict[str, OriginalPowerTransformer | RobustPowerTransformer]: A dictionary
           mapping column names to their fitted transformers.
   """
    transformers = {}

    # Use normal Yeo-Johnson transformation
    if not problematic:
        for col in dataset.X.columns:
            if col in dataset.cat_names:
                continue
            try:
                t = OriginalPowerTransformer().fit(dataset.X.loc[:, [col]])
                transformers[col] = t
            except ValueError as e:
                print(f"Could not fit transformer on column {col}, skipping.", e)
                continue
    # Use the bug-fixed version of Yeo-Johnson transformation
    else:
        for col in dataset.X.columns:
            if col in dataset.cat_names:
                continue
            try:
                t = RobustPowerTransformer().fit(dataset.X.loc[:, [col]])
                transformers[col] = t
            except ValueError as e:
                print(f"Could not fit transformer on column {col}, skipping.", e)
                continue    
    return transformers


@dataclass
class FitResult:
    """A container for storing the results of a model fitting experiment.

    Attributes:
        r2 (float): The R-squared score on the test set.
        mse (float): The mean squared error on the test set.
        mae (float): The median absolute error on the test set.
        fit_duration (float): The time taken to fit the model, in seconds.
        predict_duration (float): The time taken to generate predictions, in seconds.
        kwargs (dict[str, Any]): A dictionary for storing any extra model-specific
            results or hyperparameters.
    """
    r2: float
    mse: float
    mae: float
    fit_duration: float
    predict_duration: float
    kwargs: dict[str, Any] = field(default_factory=dict)

    def asdict(self):
        """Converts the FitResult object to a dictionary.

        This method expands the `kwargs` dictionary into the top-level
        dictionary for easier serialization (e.g., to a DataFrame).

        Returns:
            dict: The result data as a dictionary.
        """
        d = asdict(self)
        d.pop("kwargs")
        d.update(self.kwargs)
        return d

    
def _load_other_data(
    repo_id: str = "airfoil",
) -> Dataset:
    """Loads a dataset from a local CSV file and performs preprocessing.

    This is a private helper function to handle specific datasets stored locally.
    It reads a CSV, separates features and target, handles missing values by
    removing sparse columns and rows with NaNs, identifies categorical features,
    and creates both one-hot and label-encoded versions of the data.

    Args:
        repo_id (str, optional): The identifier for the dataset, used to find
            the CSV file. Defaults to "airfoil".

    Returns:
        Dataset: A fully populated Dataset object.
    """
    # Read the data
    csv_path = (
    pathlib.Path(__file__).parent.resolve() / "Data_folder_thesis" / f"{repo_id}_table.csv"
    )
    data = pd.read_csv(csv_path)

    # Make X and y
    X = data.drop(columns=["Target"])
    X = X.replace("?", np.nan)
    y = data["Target"].astype(np.float64)

    pd.options.mode.use_inf_as_na = True
    rows_removed = 0
    cols_removed = 0

    # Check which columns and rows need to be removed and remove them
    if X.isna().any().any() or y.isna().any():
        cols_to_remove = X.columns[X.isna().mean() > 0.5]
        X = X.drop(columns=cols_to_remove)
        rows_to_remove = X.index[X.isna().any(axis=1) | y.isna()]
        X = X.drop(index=rows_to_remove)
        y = y.loc[X.index]
        rows_removed = len(rows_to_remove)
        cols_removed = len(cols_to_remove)
        print(
            f"Removed {rows_removed} rows and {cols_removed} columns with missing values. "
            f"{len(X)} rows  and {X.shape[1]} columns remaining."
        )
    pd.options.mode.use_inf_as_na = False

    # Check which features are categorical or have few unique values
    cat_ids = [
        i
        for i, c in enumerate(X.columns)
        if (X[c].nunique() < 5) or (X.dtypes[c] == "O")
    ]
    cat_names = X.columns[cat_ids]

    # Make the one-hot encoded dataset
    oh_encoder = OneHotEncoder(sparse_output=False).fit(X[cat_names])
    X_oh_encoded = pd.concat(
        [
            X.drop(columns=cat_names),
            pd.DataFrame(
                oh_encoder.transform(X[cat_names]),
                columns=oh_encoder.get_feature_names_out(),
                index=X.index,
            ),
        ],
        axis=1,
    ).astype(np.float64)

    # Make the label encoded dataset
    label_encoders = {col: LabelEncoder().fit(X[col]) for col in cat_names}
    X_label_encoded = X.copy()
    for col, le in label_encoders.items():
        X_label_encoded.loc[:, col] = le.transform(X[col])
    X_label_encoded = X_label_encoded.astype(np.float64)

    return Dataset(
        id=f"other_{repo_id}",
        name=repo_id,
        X=X,
        X_oh_encoded=X_oh_encoded,
        X_label_encoded=X_label_encoded,
        y=y,
        cat_ids=cat_ids,
        cat_names=cat_names,
        oh_encoder=oh_encoder,
        label_encoders=label_encoders,
        rows_removed=rows_removed,
        cols_removed=cols_removed,
    )

@retry(ConnectionError, tries=5, delay=10)
def load_data(
    repo_id: int | str,
    ignore_feat: list[str] | None = None,
    use_download: bool = True,
    logtransform_target: bool = False,
    kind: Literal["uci", "pmlb", "other"] = "other",
) -> Dataset:
    """Loads a dataset, with options for caching and preprocessing.

    This function serves as the main entry point for loading data. It can
    load from a local cache (`.pkl` file) or generate the data by calling
    the appropriate helper functions based on the `kind` parameter.

    Args:
        repo_id (int | str): The identifier for the dataset.
        ignore_feat (list[str] | None, optional): List of features to ignore.
            Currently not implemented. Defaults to None.
        use_download (bool, optional): If True, attempts to load a cached version
            of the dataset first. Defaults to True.
        logtransform_target (bool, optional): If True, applies a log transform
            to the target variable. Currently not implemented. Defaults to False.
        kind (Literal["uci", "pmlb", "other"], optional): The source of the
            dataset. Defaults to "other".

    Returns:
        Dataset: The loaded and preprocessed dataset.
    """
    # See if the dataset is already loaded correctly
    if use_download:
        path = (
            pathlib.Path(__file__).parent.resolve() / "Data" / f"{kind}_{repo_id}.pkl"
        )
        if path.exists():
            print(f"Loading data from {path}")
            dataset = joblib.load(path)
            return dataset
        else:
            print(
                f"use_dowload was True, but path {path} does not exist. Trying to download."
            )
    # Load the dataset if not already done
    if kind == "other" :
        return _load_other_data(repo_id)
    else:
        raise ValueError(f"kind must be one of 'uci' or 'pmlb' but received {kind}")


def fit_cart(train_dataset: Dataset, test_dataset: Dataset, **init_kwargs
) -> FitResult:
    """Fits and evaluates a CART (DecisionTreeRegressor) model.

    Args:
        train_dataset (Dataset): The dataset to train the model on.
        test_dataset (Dataset): The dataset to evaluate the model on.
        **init_kwargs: Keyword arguments passed directly to the
            `DecisionTreeRegressor` constructor.

    Returns:
        FitResult: An object containing performance metrics and timings.
    """
    t1 = time.time()

    # Make and fit the CART model
    model = DecisionTreeRegressor(**init_kwargs)
    model.fit(train_dataset.X_oh_encoded, train_dataset.y)
    t2 = time.time()

    # Predict with the CART model
    try:
        y_pred = model.predict(test_dataset.X_oh_encoded)
    except:
        test_dataset.X_oh_encoded.to_csv("/tmp/data.csv", index=False)
        raise
    t3 = time.time()

    # Calculate metrics
    r2 = float(r2_score(test_dataset.y, y_pred))
    mse = float(mean_squared_error(test_dataset.y, y_pred))
    mae = float(median_absolute_error(test_dataset.y, y_pred))

    return FitResult(
        r2=r2, 
        mse=mse, 
        mae=mae, 
        fit_duration=t2 - t1, 
        predict_duration=t3 - t2,
        kwargs=init_kwargs
    )


def fit_cpilot(
    train_dataset: Dataset,
    test_dataset: Dataset,
    max_depth: int = 12,
    max_model_depth: int = 100,
    split_criterion: str = "BIC",
    min_sample_split: int = 10, 
    min_sample_leaf: int = 5,   
    step_size: int = 1,         
    random_state: int = 42,    
    truncation_factor: int = 3, 
    rel_tolerance: float = 0.0,   
    df_settings: dict | None = None, 
    regression_nodes: list[str] | None = None, 
    min_unique_values_regression: float = 5, 
    max_features_for_fit: float | str | None = None,
    compute_importances: bool = False,
    print_tree: bool = False,
    visualize_tree: bool = False,
    feature_names: list[str] = None,
    file_name_importance: str = "feature_importances.csv",
    **vis_kwargs, 
) -> FitResult:
    """Fits and evaluates a standard PILOT model.

    This function configures and runs the core PILOT algorithm on label-encoded
    data. It supports various hyperparameters to control tree growth, model
    selection, and feature subsampling.

    Args:
        train_dataset (Dataset): The dataset for training.
        test_dataset (Dataset): The dataset for evaluation.
        max_depth (int): Maximum depth of the main tree structure.
        max_model_depth (int): Maximum depth for the internal model-fitting recursion.
        split_criterion (str): The criterion for choosing the best split ('BIC', 'AIC', etc.).
        min_sample_split (int): The minimum number of samples required to split a node.
        min_sample_leaf (int): The minimum number of samples required at a leaf node.
        random_state (int): Seed for the random number generator.
        max_features_for_fit (float | str | None): The number/fraction of features to
            consider when looking for the best split.
        compute_importances (bool): If True, calculate and save feature importances.
        print_tree (bool): If True, print a textual representation of the final tree.
        visualize_tree (bool): If True, generate and save a visual plot of the tree.
        **vis_kwargs: Additional keyword arguments passed to the visualization function.

    Returns:
        FitResult: An object containing performance metrics and timings.
    """
    t1 = time.time()

    # Collect parameters for the PythonNumbaPILOT constructor
    constructor_params = {
        'max_depth': max_depth,
        'max_model_depth': max_model_depth,
        'split_criterion': split_criterion,
        'min_sample_split': min_sample_split,
        'min_sample_leaf': min_sample_leaf,
        'step_size': step_size,
        'random_state': random_state,
        'truncation_factor': truncation_factor,
        'rel_tolerance': rel_tolerance,
        'df_settings': df_settings if df_settings is not None else DEFAULT_DF_SETTINGS,
        'regression_nodes': regression_nodes, # Pass along, Pilot.py handles None
        'min_unique_values_regression': min_unique_values_regression,
    }

    # Make the PILOT model
    model = PythonNumbaPILOT(**constructor_params)

    # Prepare data for the PILOT model
    X_train_np = np.array(train_dataset.X_label_encoded.values, dtype=np.float64)
    y_train_np = train_dataset.y.values

    # Determine max features to consider in the fit
    actual_max_features_to_consider_in_fit = None # Default for Pilot.py's fit is all features
    if max_features_for_fit is not None:
        num_total_features = X_train_np.shape[1]
        if isinstance(max_features_for_fit, str) and max_features_for_fit == "sqrt":
            actual_max_features_to_consider_in_fit = int(np.sqrt(num_total_features))
        elif isinstance(max_features_for_fit, (int, float)):
            if max_features_for_fit <= 1.0 and max_features_for_fit > 0: # Fraction
                actual_max_features_to_consider_in_fit = int(max_features_for_fit * num_total_features)
            else: # Absolute number
                actual_max_features_to_consider_in_fit = int(max_features_for_fit)
        else: # Should not happen if type hints are followed
            actual_max_features_to_consider_in_fit = num_total_features 

        if actual_max_features_to_consider_in_fit is not None:
            actual_max_features_to_consider_in_fit = max(1, min(actual_max_features_to_consider_in_fit, num_total_features))
    
    # Fit the PILOT model
    model.fit(X_train_np, y_train_np, 
              categorical=train_dataset.categorical, 
              max_features_considered=actual_max_features_to_consider_in_fit,
              compute_importances=compute_importances,
              visualize_tree=visualize_tree, feature_names=feature_names, **vis_kwargs) 
    t2 = time.time()
    
    # Predict with the PILOT model
    X_test_np = np.array(test_dataset.X_label_encoded.values, dtype=np.float64)
    y_pred = model.predict(X_test_np)
    t3 = time.time()

    # If true, print a textual representation of the final tree.
    if print_tree:
        # Create a descriptive header for the printout
        print(f"\n--- Tree for pilot ---")

        # Get the feature names from the dataset to make the printout interpretable
        feature_names = train_dataset.X_label_encoded.columns.tolist()

        # Correctly CALL the print_tree method, which will internally use the new
        # `print_tree_multi` helper function.
        model.print_tree(feature_names=feature_names)

        print("--- End of tree ---\n")

    # If true, calculate and save feature importances.
    if compute_importances:
        model.save_importances_to_csv(feature_names=feature_names, filename=file_name_importance)

    # Calculate metrics
    r2 = float(r2_score(test_dataset.y, y_pred))
    mse = float(mean_squared_error(test_dataset.y, y_pred))
    mae = float(median_absolute_error(test_dataset.y, y_pred))

    return FitResult(
        r2=r2, mse=mse, mae=mae, 
        fit_duration=t2 - t1, 
        predict_duration=t3 - t2,
    )


def fit_random_forest(
    train_dataset: Dataset, test_dataset: Dataset, **init_kwargs
) -> FitResult:
    """Fits and evaluates a standard scikit-learn RandomForestRegressor.

    Args:
        train_dataset (Dataset): The dataset to train the model on.
        test_dataset (Dataset): The dataset to evaluate the model on.
        **init_kwargs: Keyword arguments passed to `RandomForestRegressor`.

    Returns:
        FitResult: An object containing performance metrics and timings.
    """
    t1 = time.time()

    # Make and fit the RF model
    model = RandomForestRegressor(**init_kwargs)
    model.fit(train_dataset.X_oh_encoded, train_dataset.y)
    t2 = time.time()

    # Predict with the RF model
    y_pred = model.predict(test_dataset.X_oh_encoded)
    t3 = time.time()

    # Calculate metrics
    r2 = float(r2_score(test_dataset.y, y_pred))
    mse = float(mean_squared_error(test_dataset.y, y_pred))
    mae = float(median_absolute_error(test_dataset.y, y_pred))

    return FitResult(
        r2=r2, mse=mse, mae=mae, fit_duration=t2 - t1, predict_duration=t3 - t2
    )


def fit_ridge(
    train_dataset: Dataset, test_dataset: Dataset, **init_kwargs
) -> FitResult:
    """Fits and evaluates a Ridge regression model.

    Args:
        train_dataset (Dataset): The dataset to train the model on.
        test_dataset (Dataset): The dataset to evaluate the model on.
        **init_kwargs: Keyword arguments passed to `Ridge`.

    Returns:
        FitResult: An object containing performance metrics and timings.
    """
    t1 = time.time()

    # Make and fit the Ridge model
    model = Ridge(**init_kwargs)
    model.fit(train_dataset.X_oh_encoded, train_dataset.y)
    t2 = time.time()

    # Predict with the Ridge model
    y_pred = model.predict(test_dataset.X_oh_encoded)
    t3 = time.time()

    # Calculate metrics
    r2 = float(r2_score(test_dataset.y, y_pred))
    mse = float(mean_squared_error(test_dataset.y, y_pred))
    mae = float(median_absolute_error(test_dataset.y, y_pred))

    return FitResult(
        r2=r2, mse=mse, mae=mae, fit_duration=t2 - t1, predict_duration=t3 - t2
    )


def fit_lasso(
    train_dataset: Dataset, test_dataset: Dataset, **init_kwargs
) -> FitResult:
    """Fits and evaluates a Lasso regression model.

    Args:
        train_dataset (Dataset): The dataset to train the model on.
        test_dataset (Dataset): The dataset to evaluate the model on.
        **init_kwargs: Keyword arguments passed to `Lasso`.

    Returns:
        FitResult: An object containing performance metrics and timings.
    """
    t1 = time.time()

    # Make and fit the Lasso model
    model = Lasso(**init_kwargs, max_iter=10000)
    model.fit(train_dataset.X_oh_encoded, train_dataset.y)
    t2 = time.time()

    # Predict with the Lasso model
    y_pred = model.predict(test_dataset.X_oh_encoded)
    t3 = time.time()

    # Calculate metrics
    r2 = float(r2_score(test_dataset.y, y_pred))
    mse = float(mean_squared_error(test_dataset.y, y_pred))
    mae = float(median_absolute_error(test_dataset.y, y_pred))

    return FitResult(
        r2=r2, mse=mse, mae=mae, fit_duration=t2 - t1, predict_duration=t3 - t2
    )


def fit_cpilot_forest(
    train_dataset: Dataset,
    test_dataset: Dataset,
    n_estimators: int = 10,
    max_depth: int = 12,
    max_model_depth: int = 100,
    min_sample_fit: int = 10,
    min_sample_alpha: int = 5,
    min_sample_leaf: int = 5,
    random_state: int = 42,
    n_features_tree: float | str = 1,
    n_features_node: float | str = 1,
    df_settings: dict[str, int] | None = None,
    rel_tolerance: float = 1e-2,
    precision_scale: float = 1e-10,
    max_pivot: int | float = None,
) -> FitResult:
    """Fits and evaluates a RAFFLE model.

    This is a random forest where each tree is a PILOT model.

    Args:
        train_dataset (Dataset): The dataset for training.
        test_dataset (Dataset): The dataset for evaluation.
        **init_kwargs: Keyword arguments passed to `RandomForestCPilot`.

    Returns:
        FitResult: An object containing performance metrics and timings, including
            average and max tree depths.
    """
    t1 = time.time()

    # Make and fit the RAFFLE model
    model = RandomForestCPilot(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_model_depth=max_model_depth,
        min_sample_fit=min_sample_fit,
        min_sample_alpha=min_sample_alpha,
        min_sample_leaf=min_sample_leaf,
        n_features_tree=n_features_tree,
        n_features_node=n_features_node,
        df_settings=df_settings,
        rel_tolerance=rel_tolerance,
        precision_scale=precision_scale,
        random_state=random_state,
        max_pivot=max_pivot,
    )
    model.fit(
        train_dataset.X_label_encoded.values,
        train_dataset.y.values,
        categorical_idx=train_dataset.categorical,
        n_workers=1,
    )
    t2 = time.time()

    # Predict with the RAFFLE model
    y_pred = model.predict(test_dataset.X_label_encoded.values)
    t3 = time.time()

    # Calculate metrics
    r2 = float(r2_score(test_dataset.y, y_pred))
    mse = float(mean_squared_error(test_dataset.y, y_pred))
    mae = float(median_absolute_error(test_dataset.y, y_pred))

    # Mean tree depth
    depths = pd.concat(
        [
            e.tree_summary()[["depth", "model_depth"]].max().to_frame().T
            for e in model.estimators
        ]
    ).agg(["mean", "max"])

    return FitResult(
        r2=r2,
        mse=mse,
        mae=mae,
        fit_duration=t2 - t1,
        predict_duration=t3 - t2,
        kwargs=dict(
            mean_tree_depth=depths.loc["mean", "depth"],
            max_tree_depth=depths.loc["max", "depth"],
            mean_tree_model_depth=depths.loc["mean", "model_depth"],
            max_tree_model_depth=depths.loc["max", "model_depth"],
        ),
    )


def fit_xgboost(
    train_dataset: Dataset,
    test_dataset: Dataset,
    max_depth: int = 6,
    max_node_features: float = 1,
    n_estimators: int = 100,
) -> FitResult:
    """Fits and evaluates an XGBoost Regressor model.

    Args:
        train_dataset (Dataset): The dataset for training.
        test_dataset (Dataset): The dataset for evaluation.
        **init_kwargs: Keyword arguments passed to `xgb.XGBRegressor`.

    Returns:
        FitResult: An object containing performance metrics and timings.
    """
    t1 = time.time()

    # Make and fit the XGB model
    model = xgb.XGBRegressor(
        max_depth=max_depth,
        colsample_bynode=max_node_features,
        n_estimators=n_estimators,
    )
    model.fit(
        train_dataset.X_oh_encoded.values,
        train_dataset.y.values,
    )
    t2 = time.time()

    # Predict with the XGB model
    y_pred = model.predict(test_dataset.X_oh_encoded.values)
    t3 = time.time()

    # Calculate metrics
    r2 = float(r2_score(test_dataset.y, y_pred))
    mse = float(mean_squared_error(test_dataset.y, y_pred))
    mae = float(median_absolute_error(test_dataset.y, y_pred))

    return FitResult(
        r2=r2, mse=mse, mae=mae, fit_duration=t2 - t1, predict_duration=t3 - t2
    )
    
def fit_ridge_pilot_ensemble(
    train_dataset: Dataset,
    test_dataset: Dataset,
    max_depth: int = 12,
    max_model_depth: int = 100,
    split_criterion: str = "BIC",
    min_sample_split: int = 10, 
    min_sample_leaf: int = 5,   
    step_size: int = 1,         
    random_state: int = 42,    
    truncation_factor: int = 3, 
    rel_tolerance: float = 0.0,   
    df_settings: dict | None = None, 
    regression_nodes: list[str] | None = None, 
    min_unique_values_regression: float = 5, 
    max_features_for_fit: float | str | None = None,
    alpha: float = 1.0,
    compute_importances: bool = False,
    print_tree: bool = False,
    visualize_tree: bool = False,
    feature_names: list[str] = None,
    file_name_importance: str = "feature_importances.csv",
    **vis_kwargs,
) -> FitResult:
    """Fits a two-stage ensemble of Ridge regression followed by PILOT.

    This method first fits a global Ridge model on the one-hot encoded data.
    Then, it fits a PILOT model on the residuals of the Ridge model using
    the label-encoded data. The final prediction is the sum of the predictions
    from both models.

    Args:
        train_dataset (Dataset): The dataset for training.
        test_dataset (Dataset): The dataset for evaluation.
        alpha (float): The regularization strength for the Ridge model.
        **pilot_kwargs: Keyword arguments passed to the `fit_cpilot` function.

    Returns:
        FitResult: Performance metrics for the combined ensemble model.
    """
    t1 = time.time()
    
    #Collect parameters for the PythonNumbaPILOT constructor
    constructor_params = {
        'max_depth': max_depth,
        'max_model_depth': max_model_depth,
        'split_criterion': split_criterion,
        'min_sample_split': min_sample_split,
        'min_sample_leaf': min_sample_leaf,
        'step_size': step_size,
        'random_state': random_state,
        'truncation_factor': truncation_factor,
        'rel_tolerance': rel_tolerance,
        'df_settings': df_settings if df_settings is not None else DEFAULT_DF_SETTINGS,
        'regression_nodes': regression_nodes, # Pass along, Pilot.py handles None
        'min_unique_values_regression': min_unique_values_regression,
    }
    
    X_train_np = np.array(train_dataset.X_label_encoded.values, dtype=np.float64)

    # Determine max features to consider in the fit
    actual_max_features_to_consider_in_fit = None # Default for Pilot.py's fit is all features
    if max_features_for_fit is not None:
        num_total_features = X_train_np.shape[1]
        if isinstance(max_features_for_fit, str) and max_features_for_fit == "sqrt":
            actual_max_features_to_consider_in_fit = int(np.sqrt(num_total_features))
        elif isinstance(max_features_for_fit, (int, float)):
            if max_features_for_fit <= 1.0 and max_features_for_fit > 0: # Fraction
                actual_max_features_to_consider_in_fit = int(max_features_for_fit * num_total_features)
            else: # Absolute number
                actual_max_features_to_consider_in_fit = int(max_features_for_fit)
        else: # Should not happen if type hints are followed
            actual_max_features_to_consider_in_fit = num_total_features 

        if actual_max_features_to_consider_in_fit is not None:
            actual_max_features_to_consider_in_fit = max(1, min(actual_max_features_to_consider_in_fit, num_total_features))

    # Make and fit the Ridge model, and predict with the model and calculate residuals
    model_ridge = Ridge(alpha)
    model_ridge.fit(train_dataset.X_oh_encoded, train_dataset.y)
    y_pred_ridge_train = model_ridge.predict(train_dataset.X_oh_encoded)
    res = np.array(train_dataset.y.values - y_pred_ridge_train)

    # Make and fit the PILOT model with residuals from Ridge model
    model_pilot = PythonNumbaPILOT(**constructor_params)
    model_pilot.fit(X_train_np, res, 
                    categorical=train_dataset.categorical,
                    max_features_considered=actual_max_features_to_consider_in_fit,
                    compute_importances=compute_importances, visualize_tree=visualize_tree,
                    feature_names=feature_names, pre_model=model_ridge, pre_model_name="Ridge",
                    pre_model_feature_names=pre_model_feature_names,
                    categorical_feature_names=categorical_feature_names, **vis_kwargs)
    
    t2 = time.time()

    # Predict with both models and combine
    X_test_np = np.array(test_dataset.X_label_encoded.values, dtype=np.float64)
    y_pred_ridge = model_ridge.predict(test_dataset.X_oh_encoded)
    y_pred_pilot = model_pilot.predict(X_test_np)
    y_pred = y_pred_ridge + y_pred_pilot
    
    t3 = time.time()

    # Print the tree if print_tree is true
    if print_tree:
        # Get all the necessary lists of names from your Dataset object
        pilot_feature_names = list(train_dataset.X_label_encoded.columns)
        pre_model_feature_names = list(train_dataset.X_oh_encoded.columns)
        categorical_feature_names = train_dataset.cat_names

        model_pilot.print_ensemble_model(
            pre_model=model_ridge,
            pre_model_name="Ridge",
            pilot_feature_names=pilot_feature_names,
            pre_model_feature_names=pre_model_feature_names,
            categorical_feature_names=categorical_feature_names
        )

    # Compute the importances if compute_importances is true
    if compute_importances:
        # Get all the necessary lists of names from your Dataset object
        pilot_feature_names = list(train_dataset.X_label_encoded.columns)
        pre_model_feature_names = list(train_dataset.X_oh_encoded.columns)
        categorical_feature_names = train_dataset.cat_names  # Get this from the dataset

        model_pilot.calculate_ensemble_importances(
            pre_model=model_ridge,
            y_train_original=train_dataset.y.values,
            y_pred_pre_model_train=y_pred_ridge_train,
            pilot_feature_names=pilot_feature_names,
            pre_model_feature_names=pre_model_feature_names,
            categorical_feature_names=categorical_feature_names,  # Pass the new arg
            filename=file_name_importance
        )

    # Calculate metrics
    r2 = float(r2_score(test_dataset.y, y_pred))
    mse = float(mean_squared_error(test_dataset.y, y_pred))
    mae = float(median_absolute_error(test_dataset.y, y_pred))

    return FitResult(
        r2=r2, mse=mse, mae=mae, fit_duration=t2 - t1, predict_duration=t3 - t2
    )
    
def fit_lasso_pilot_ensemble(
    train_dataset: Dataset,
    test_dataset: Dataset,
    max_depth: int = 12,
    max_model_depth: int = 100,
    split_criterion: str = "BIC",
    min_sample_split: int = 10, 
    min_sample_leaf: int = 5,   
    step_size: int = 1,         
    random_state: int = 42,    
    truncation_factor: int = 3, 
    rel_tolerance: float = 0.0,   
    df_settings: dict | None = None, 
    regression_nodes: list[str] | None = None, 
    min_unique_values_regression: float = 5, 
    max_features_for_fit: float | str | None = None,
    alpha: float = 0.01,
    compute_importances: bool = False,
    print_tree: bool = False,
    visualize_tree: bool = False,
    feature_names: list[str] = None,
    file_name_importance: str = "feature_importances.csv",
    **vis_kwargs,
) -> FitResult:
    """Fits a two-stage ensemble of Lasso regression followed by PILOT.

    This method first fits a global Lasso model on the one-hot encoded data.
    Then, it fits a PILOT model on the residuals of the Lasso model using
    the label-encoded data. The final prediction is the sum of the predictions
    from both models.

    Args:
        train_dataset (Dataset): The dataset for training.
        test_dataset (Dataset): The dataset for evaluation.
        alpha (float): The regularization strength for the Lasso model.
        **pilot_kwargs: Keyword arguments passed to the `fit_cpilot` function.

    Returns:
        FitResult: Performance metrics for the combined ensemble model.
    """
    t1 = time.time()
    
    #Collect parameters for the PythonNumbaPILOT constructor
    constructor_params = {
        'max_depth': max_depth,
        'max_model_depth': max_model_depth,
        'split_criterion': split_criterion,
        'min_sample_split': min_sample_split,
        'min_sample_leaf': min_sample_leaf,
        'step_size': step_size,
        'random_state': random_state,
        'truncation_factor': truncation_factor,
        'rel_tolerance': rel_tolerance,
        'df_settings': df_settings if df_settings is not None else DEFAULT_DF_SETTINGS,
        'regression_nodes': regression_nodes, # Pass along, Pilot.py handles None
        'min_unique_values_regression': min_unique_values_regression,
    }

    X_train_np = np.array(train_dataset.X_label_encoded.values, dtype=np.float64)

    # Determine max features to consider in the fit
    actual_max_features_to_consider_in_fit = None # Default for Pilot.py's fit is all features
    if max_features_for_fit is not None:
        num_total_features = X_train_np.shape[1]
        if isinstance(max_features_for_fit, str) and max_features_for_fit == "sqrt":
            actual_max_features_to_consider_in_fit = int(np.sqrt(num_total_features))
        elif isinstance(max_features_for_fit, (int, float)):
            if max_features_for_fit <= 1.0 and max_features_for_fit > 0: # Fraction
                actual_max_features_to_consider_in_fit = int(max_features_for_fit * num_total_features)
            else: # Absolute number
                actual_max_features_to_consider_in_fit = int(max_features_for_fit)
        else: # Should not happen if type hints are followed
            actual_max_features_to_consider_in_fit = num_total_features 

        if actual_max_features_to_consider_in_fit is not None:
            actual_max_features_to_consider_in_fit = max(1, min(actual_max_features_to_consider_in_fit, num_total_features))

    # Make and fit the Lasso model, and predict with the model and calculate residuals
    model_lasso = Lasso(alpha, max_iter=10000)
    model_lasso.fit(train_dataset.X_oh_encoded, train_dataset.y)
    y_pred_lasso_train = model_lasso.predict(train_dataset.X_oh_encoded)
    res = np.array(train_dataset.y.values - y_pred_lasso_train)

    # Make and fit the PILOT model with residuals from Lasso model
    model_pilot = PythonNumbaPILOT(**constructor_params)
    model_pilot.fit(X_train_np, res, 
                    categorical=train_dataset.categorical,
                    max_features_considered=actual_max_features_to_consider_in_fit,
                    compute_importances=compute_importances, visualize_tree=visualize_tree,
                    feature_names=feature_names, pre_model=model_lasso, pre_model_name="Lasso",
                    pre_model_feature_names=pre_model_feature_names,
                    categorical_feature_names=categorical_feature_names, **vis_kwargs)
    
    t2 = time.time()

    # Predict with both models and combine
    X_test_np = np.array(test_dataset.X_label_encoded.values, dtype=np.float64)
    y_pred_lasso = model_lasso.predict(test_dataset.X_oh_encoded)
    y_pred_pilot = model_pilot.predict(X_test_np)
    y_pred = y_pred_lasso + y_pred_pilot
    
    t3 = time.time()

    # Print the tree if print_tree is true
    if print_tree:
        # Get all the necessary lists of names from your Dataset object
        pilot_feature_names = list(train_dataset.X_label_encoded.columns)
        pre_model_feature_names = list(train_dataset.X_oh_encoded.columns)
        categorical_feature_names = train_dataset.cat_names

        model_pilot.print_ensemble_model(
            pre_model=model_lasso,
            pre_model_name="Lasso",
            pilot_feature_names=pilot_feature_names,
            pre_model_feature_names=pre_model_feature_names,
            categorical_feature_names=categorical_feature_names
        )

    # Compute the importances if compute_importances is true
    if compute_importances:
        # Get all the necessary lists of names from your Dataset object
        pilot_feature_names = list(train_dataset.X_label_encoded.columns)
        pre_model_feature_names = list(train_dataset.X_oh_encoded.columns)
        categorical_feature_names = train_dataset.cat_names  # Get this from the dataset

        model_pilot.calculate_ensemble_importances(
            pre_model=model_lasso,
            y_train_original=train_dataset.y.values,
            y_pred_pre_model_train=y_pred_lasso_train,
            pilot_feature_names=pilot_feature_names,
            pre_model_feature_names=pre_model_feature_names,
            categorical_feature_names=categorical_feature_names,  # Pass the new arg
            filename=file_name_importance
        )

    # Calculate the metrics
    r2 = float(r2_score(test_dataset.y, y_pred))
    mse = float(mean_squared_error(test_dataset.y, y_pred))
    mae = float(median_absolute_error(test_dataset.y, y_pred))
    return FitResult(
        r2=r2, mse=mse, mae=mae, fit_duration=t2 - t1, predict_duration=t3 - t2
    )
    
def fit_pilot_nlfs(
    train_dataset: Dataset,
    test_dataset: Dataset,
    max_depth: int = 12,
    max_model_depth: int = 100,
    split_criterion: str = "BIC",
    min_sample_split: int = 10, 
    min_sample_leaf: int = 5,   
    step_size: int = 1,         
    random_state: int = 42,    
    truncation_factor: int = 3, 
    rel_tolerance: float = 0.0,   
    df_settings: dict | None = None, 
    regression_nodes: list[str] | None = None, 
    min_unique_values_regression: float = 5, 
    max_features_for_fit: float | str | None = None,
    alpha: float = 0.005,
    nlfs_lars: bool = False,
    only_fallback: bool = False,
    compute_importances: bool = False,
    print_tree: bool = False,
    visualize_tree: bool = False,
    feature_names: list[str] = None,
    file_name_importance: str = "feature_importances.csv",
    **vis_kwargs,
) -> FitResult:
    """Fits and evaluates an NLFS-PILOT (Node-Level Feature Selection) model.

    This variant of PILOT uses Lasso with LARS or fixed alpha, or a fallback staregy
    at each node to  select features for the linear models.

    Args:
        train_dataset (Dataset): The dataset for training.
        test_dataset (Dataset): The dataset for evaluation.
        alpha (float): Regularization strength for the node-level Lasso.
        nlfs_lars (bool): If True, use LARS for feature selection at each node.
            Otherwise, use Lasso with fixed alpha.
        only_fallback (bool): If True, only use the fallback strategy for feature selection.
        **pilot_kwargs: Keyword arguments passed to the PILOT constructor and fit method.

    Returns:
        FitResult: Performance metrics for the NLFS-PILOT model.
    """
    t1 = time.time()

    # Check if that only lars or fallback strategy is used, never both
    if nlfs_lars and only_fallback:
        raise Exception("nlfs_lars and only_fallback can not be true at the same time!")

    #Collect parameters for the PythonNumbaPILOT constructor
    constructor_params = {
        'max_depth': max_depth,
        'max_model_depth': max_model_depth,
        'split_criterion': split_criterion,
        'min_sample_split': min_sample_split,
        'min_sample_leaf': min_sample_leaf,
        'step_size': step_size,
        'random_state': random_state,
        'truncation_factor': truncation_factor,
        'rel_tolerance': rel_tolerance,
        'df_settings': df_settings if df_settings is not None else DEFAULT_DF_SETTINGS,
        'regression_nodes': regression_nodes, # Pass along, Pilot.py handles None
        'min_unique_values_regression': min_unique_values_regression,
    }

    # Make the model
    model = PythonNumbaPILOT(**constructor_params)

    # NLFS-PILOT models use label encoded dataset
    X_train_np = np.array(train_dataset.X_label_encoded.values, dtype=np.float64)
    y_train_np = train_dataset.y.values

    # Determine max features to consider in the fit
    actual_max_features_to_consider_in_fit = None # Default for Pilot.py's fit is all features
    if max_features_for_fit is not None:
        num_total_features = X_train_np.shape[1]
        if isinstance(max_features_for_fit, str) and max_features_for_fit == "sqrt":
            actual_max_features_to_consider_in_fit = int(np.sqrt(num_total_features))
        elif isinstance(max_features_for_fit, (int, float)):
            if max_features_for_fit <= 1.0 and max_features_for_fit > 0: # Fraction
                actual_max_features_to_consider_in_fit = int(max_features_for_fit * num_total_features)
            else: # Absolute number
                actual_max_features_to_consider_in_fit = int(max_features_for_fit)
        else: # Should not happen if type hints are followed
            actual_max_features_to_consider_in_fit = num_total_features 

        if actual_max_features_to_consider_in_fit is not None:
            actual_max_features_to_consider_in_fit = max(1, min(actual_max_features_to_consider_in_fit, num_total_features))
    
    # Fit the model
    model.fit_nlfs(X_train_np, y_train_np, 
                categorical=train_dataset.categorical,
                max_features_considered=actual_max_features_to_consider_in_fit,
                alpha=alpha, nlfs_lars=nlfs_lars, only_fallback=only_fallback,
                compute_importances=compute_importances,
                visualize_tree=visualize_tree, feature_names=feature_names, **vis_kwargs)
    t2 = time.time()
    
    # Predict with the model
    X_test_np = np.array(test_dataset.X_label_encoded.values, dtype=np.float64)
    y_pred = model.predict(X_test_np)
    t3 = time.time()

    # Print the tree if print_tree is true
    if print_tree:
        # Create a descriptive header for the printout
        strategy_name = ""
        if only_fallback:
            strategy_name = "only_fallback"
        elif nlfs_lars:
            strategy_name = "lars"
        else:
            strategy_name = "prefix"

        print(f"\n--- Tree for pilot_nlfs ({strategy_name}) ---")

        # Get the feature names from the dataset to make the printout interpretable
        feature_names = train_dataset.X_label_encoded.columns.tolist()

        # Correctly CALL the print_tree method, which will internally use the new
        # `print_tree_multi` helper function.
        model.print_tree(feature_names=feature_names)

        print("--- End of tree ---\n")

    # Compute the importances if compute_importances is true
    if compute_importances:
        model.save_importances_to_csv(feature_names=feature_names, filename=file_name_importance)

    # Calculate metrics
    r2 = float(r2_score(test_dataset.y, y_pred))
    mse = float(mean_squared_error(test_dataset.y, y_pred))
    mae = float(median_absolute_error(test_dataset.y, y_pred))

    return FitResult(
        r2=r2, mse=mse, mae=mae, 
        fit_duration=t2 - t1, 
        predict_duration=t3 - t2,
    )

def fit_pilot_multi(
    train_dataset: Dataset,
    test_dataset: Dataset,
    max_depth: int = 12,
    max_model_depth: int = 100,
    split_criterion: str = "BIC",
    min_sample_split: int = 10, 
    min_sample_leaf: int = 5,   
    step_size: int = 1,         
    random_state: int = 42,    
    truncation_factor: int = 3, 
    rel_tolerance: float = 0.0,   
    df_settings: dict | None = None, 
    regression_nodes: list[str] | None = None, 
    min_unique_values_regression: float = 5, 
    max_features_for_fit: float | str | None = None,
    alpha: float = 0.01,
    multi_lars: bool = False,
    finalist_s: bool = True,
    finalist_d: bool = False,
    per_feature: bool = False,
    full_multi: bool = False,
    compute_importances: bool = False,
    print_tree: bool = False ,
    visualize_tree: bool = False,
    feature_names: list[str] = None,
    file_name_importance: str = "feature_importances.csv",
    **vis_kwargs, 
) -> FitResult:
    """Fits and evaluates a Multi-PILOT (Multivariate Split) model.

    This PILOT variant considers multivariate splits, where the split condition
    itself is a linear combination of features, found using Lasso with LARS or a fixed alpha.
    Exactly one splitting strategy must be chosen.

    Args:
        train_dataset (Dataset): The dataset for training.
        test_dataset (Dataset): The dataset for evaluation.
        alpha (float): Regularization strength for finding the multivariate split.
        multi_lars (bool): If True, use LARS to find the split; otherwise, use Lasso with fixed alpha.
        finalist_s (bool): 'Finalist-S' strategy.
        finalist_d (bool): 'Finalist-D' strategy.
        per_feature (bool): 'Per-Feature' strategy.
        full_multi (bool): 'Full-Multi' strategy.
        **pilot_kwargs: Keyword arguments passed to the PILOT constructor and fit method.

    Returns:
        FitResult: Performance metrics for the Multi-PILOT model.
    """
    t1 = time.time()

    # Count how many are set to True.
    strategies = [finalist_s, finalist_d, per_feature, full_multi]
    num_strategies_selected = sum(strategies)
    
    # Check if the count is not equal to 1 -> Error
    if num_strategies_selected != 1:
        raise ValueError(
            "Exactly one MultiSplit strategy (finalist_s, finalist_d, per_feature, full_multi) "
            f"must be set to True. Found: {num_strategies_selected}."
        )

    # Collect parameters for the PythonNumbaPILOT constructor
    constructor_params = {
        'max_depth': max_depth,
        'max_model_depth': max_model_depth,
        'split_criterion': split_criterion,
        'min_sample_split': min_sample_split,
        'min_sample_leaf': min_sample_leaf,
        'step_size': step_size,
        'random_state': random_state,
        'truncation_factor': truncation_factor,
        'rel_tolerance': rel_tolerance,
        'df_settings': df_settings if df_settings is not None else DEFAULT_DF_SETTINGS,
        'regression_nodes': regression_nodes,
        'min_unique_values_regression': min_unique_values_regression,
    }

    # Make the model
    model = PythonNumbaPILOT(**constructor_params)

    # Multi-PILOT functions use label encoded dataset
    X_train_np = np.array(train_dataset.X_label_encoded.values, dtype=np.float64)
    y_train_np = train_dataset.y.values

    # Determine max features to consider in the fit
    actual_max_features_to_consider_in_fit = None 
    if max_features_for_fit is not None:
        num_total_features = X_train_np.shape[1]
        if isinstance(max_features_for_fit, str) and max_features_for_fit == "sqrt":
            actual_max_features_to_consider_in_fit = int(np.sqrt(num_total_features))
        elif isinstance(max_features_for_fit, (int, float)):
            if 0 < max_features_for_fit <= 1.0:
                actual_max_features_to_consider_in_fit = int(max_features_for_fit * num_total_features)
            else:
                actual_max_features_to_consider_in_fit = int(max_features_for_fit)
        else:
            actual_max_features_to_consider_in_fit = num_total_features 

        if actual_max_features_to_consider_in_fit is not None:
            actual_max_features_to_consider_in_fit = max(1, min(actual_max_features_to_consider_in_fit, num_total_features))

    # Fit the model
    model.fit_multi(
        X_train_np, y_train_np, 
        categorical=train_dataset.categorical, 
        max_features_considered=actual_max_features_to_consider_in_fit,
        alpha=alpha, multi_lars=multi_lars, finalist_s=finalist_s, 
        finalist_d=finalist_d, per_feature=per_feature, full_multi=full_multi,
        compute_importances=compute_importances,
        visualize_tree=visualize_tree, feature_names=feature_names, **vis_kwargs
    ) 
    t2 = time.time()

    # Predict with the model
    X_test_np = np.array(test_dataset.X_label_encoded.values, dtype=np.float64)
    y_pred = model.predict_multi(X_test_np)
    t3 = time.time()

    # Print the tree if print_tree is true
    if print_tree:
        # Create a descriptive header for the printout
        strategy_name = ""
        if finalist_s: strategy_name = "Finalist-S"
        elif finalist_d: strategy_name = "Finalist-D"
        elif per_feature: strategy_name = "Per-Feature"
        elif full_multi: strategy_name = "Full-Multi"
        
        lars_info = "LARS" if multi_lars else "Prefix"
        
        print(f"\n--- Tree for pilot_multi ({strategy_name}, {lars_info}) ---")
        
        # Get the feature names from the dataset to make the printout interpretable
        feature_names = train_dataset.X_label_encoded.columns.tolist()
        
        # Correctly CALL the print_tree method, which will internally use the new
        # `print_tree_multi` helper function.
        model.print_tree(feature_names=feature_names) 
        
        print("--- End of tree ---\n")

    # Compute the importances if compute_importances is true
    if compute_importances:
        model.save_importances_to_csv(feature_names=feature_names, filename=file_name_importance)

    # Calculate metrics
    r2 = float(r2_score(test_dataset.y, y_pred))
    mse = float(mean_squared_error(test_dataset.y, y_pred))
    mae = float(median_absolute_error(test_dataset.y, y_pred))

    return FitResult(
        r2=r2, mse=mse, mae=mae, 
        fit_duration=t2 - t1, 
        predict_duration=t3 - t2,
    )