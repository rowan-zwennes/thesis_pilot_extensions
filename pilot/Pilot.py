"""This module implements the PILOT (Piecewise-Linear Regression Tree) algorithm
and its extensions, including versions with Node-Level Feature Selection (NLFS)
and Multivariate splits. It is designed for high performance using Numba's JIT
compilation for its core computational routines.
"""

import warnings
import uuid
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import numba as nb
import numpy as np
import pandas as pd

from typing import Optional
from numba import jit, objmode
from .Tree import tree
from .Tree import tree_multi
from .Tree import visualize_pilot_tree 

from sklearn.base import BaseEstimator
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLarsIC
from sklearn.ensemble import ExtraTreesRegressor
from scipy.stats import f_oneway

REGRESSION_NODES = ["con", "lin", "blin", "pcon", "plin", "lasso_no_split", "lasso_split"]
NODE_PREFERENCE_ORDER = REGRESSION_NODES + ["pconc"]
DEFAULT_DF_SETTINGS = {"con": 1, "lin": 2, "pcon": 5, "blin": 5, "plin": 7, "pconc": 5, "lasso_no_split": 1, "lasso_split": 5}

@nb.njit()
def random_sample(a, k):
    """
    Selects a random sample of `k` elements from array `a` without replacement.

    This function is a Numba-compatible wrapper that uses `objmode` to fall back
    to NumPy's `random.choice` for the sampling operation, as it's not directly
    available in Numba's nopython mode.

    Args:
        a (np.ndarray): The array to sample from.
        k (int): The number of items to sample.

    Returns:
        np.ndarray: An array containing the `k` randomly selected items.
    """
    with objmode(a="int64[:]"):
        a = np.random.choice(a, size=k, replace=False)
    return a

@nb.njit(parallel=True)
def isin(a, b):
    """
    Computes an element-wise check if items in array `a` are present in array `b`.

    This is a Numba-optimized equivalent of `np.isin`, designed to be fast for
    use within JIT-compiled functions. It leverages a set for `b` for efficient
    lookups and is parallelized.

    Args:
        a (np.ndarray): The input array of elements to check.
        b (np.ndarray): The array of values to check for membership against.

    Returns:
        np.ndarray: A boolean array of the same shape as `a`, where `out[i]` is
            True if `a[i]` is in `b`, and False otherwise.
    """
    out = np.empty(a.shape[0], dtype=nb.boolean)
    b = set(b)
    for i in nb.prange(a.shape[0]):
        if a[i] in b:
            out[i] = True
        else:
            out[i] = False
    return out

@jit(nb.float64[:](nb.types.unicode_type, nb.int64, nb.float64[:], nb.int64[:], nb.int64))
def loss_fun(criteria, num, Rss, k: np.ndarray, coef_num: int = 0):
    """
    Calculates the loss for a model based on a specified information criterion.

    This function computes common criteria like AIC, AICc, and BIC, which are
    used to select the best model by balancing model fit (RSS) against model
    complexity (degrees of freedom).

    Args:
        criteria (str): The information criterion to use ('AIC', 'AICc', 'BIC').
        num (int): The total number of samples in the node.
        Rss (np.ndarray): The residual sum of squares of the model(s).
        k (np.ndarray): The (base) degrees of freedom for the model type(s).
        coef_num (int, optional): Additional degrees of freedom from selected
            coefficients (e.g., in a Lasso model). Defaults to 0.

    Returns:
        np.ndarray: An array containing the calculated loss value(s).
    """
    if criteria == "AIC":
        return num * np.log(Rss / num) + 2 * (k + coef_num)
    elif criteria == "AICc":
        return num * np.log(Rss / num) + 2 * (k + coef_num) + (2 * k**2 + 2 * k) / (num - k - 1)
    elif criteria == "BIC":
        return num * np.log(Rss / num) + np.log(num) * (k + coef_num)
    return np.array([0.0])


@jit(
    nb.types.Tuple(
        (
            nb.int64,
            nb.float64,
            nb.types.unicode_type,
            nb.float64[:],
            nb.float64[:],
            nb.float64[:],
            nb.int64[:],
        )
    )(
        nb.int64[:],  # index
        nb.typeof(["a", "b"]),  # regression_nodes
        nb.int64,  # n_features
        nb.int64[:, :],  # sorted_X_indices
        nb.float64[:, :],  # X
        nb.float64[:, :],  # y
        nb.types.unicode_type,  # split_criterion
        nb.int64,  # min_sample_leaf
        nb.int64[:],  # k_con
        nb.int64[:],  # k_lin
        nb.int64[:],  # k_split_nodes
        nb.int64[:],  # k_pconc
        nb.int64[:],  # categorical
        nb.int64,  # max_features_considered,
        nb.int64,  # min_unique_values_regression
    ),
    nopython=True,
)
def best_split(
    index,
    regression_nodes,
    n_features,
    sorted_X_indices,
    X,
    y,
    split_criterion,
    min_sample_leaf,
    k_con,
    k_lin,
    k_split_nodes,
    k_pconc,
    categorical,
    max_features_considered,
    min_unique_values_regression,
):
    """
    Finds the best univariate split or no-split model for a given node.

    This core Numba-jitted function iterates through a random subset of features
    to find the optimal split point and model type (e.g., constant, linear,
    piecewise constant) that minimizes the specified information criterion (e.g., BIC).
    It efficiently calculates model parameters and RSS using pre-computed moments
    and handles both numerical and categorical features.

    Args:
        index (np.ndarray): Indices of the data points in the current node.
        regression_nodes (list): A list of model types to consider (e.g., 'con', 'lin').
        n_features (int): Total number of features in the dataset.
        sorted_X_indices (np.ndarray): Pre-sorted indices of X for each feature.
        X (np.ndarray): The predictor data.
        y (np.ndarray): The response data.
        split_criterion (str): The information criterion to use for model selection.
        min_sample_leaf (int): The minimum number of samples in a child node.
        k_* (np.ndarray): Degrees of freedom for each model type.
        categorical (np.ndarray): Indices of categorical features.
        max_features_considered (int): The number of features to randomly sample.
        min_unique_values_regression (int): Min unique values for a linear model.

    Returns:
        tuple: A tuple containing the best model's properties:
            - best_feature (int): The index of the best feature for splitting.
            - best_pivot (float): The value to split on for a numerical feature.
            - best_node (str): The type of the best model found (e.g., 'pcon').
            - lm_L (np.ndarray): [coef, intercept] for the left child model.
            - lm_R (np.ndarray): [coef, intercept] for the right child model.
            - interval (np.ndarray): The range [min, max] of the predictor in the node.
            - pivot_c (np.ndarray): The categorical levels for the left child.
    """

    # Initialize variables, should be consistent with the variable type
    best_pivot = -1.0
    best_node = ""
    best_loss = -1.0
    best_feature = -1
    lm_L = np.array([0.0, 0.0])
    lm_R = np.array([0.0, 0.0])
    interval = np.array([-np.inf, np.inf])
    pivot_c = np.array([0])

    # Initialize the coef and intercept for 'blin'/'plin'/'pcon'
    l = (
        1 * ("blin" in regression_nodes)
        + 1 * ("plin" in regression_nodes)
        + 1 * ("pcon" in regression_nodes)
    )
    coef = np.zeros((l, 2)) * np.nan
    intercept = np.zeros((l, 2)) * np.nan

    # Search for the best split among all features, negelecting the indices column
    for feature_id in random_sample(np.arange(1, n_features + 1), max_features_considered):
        # Get sorted X, y
        idx = sorted_X_indices[feature_id - 1]
        idx = idx[isin(idx, index)]
        X_sorted, y_sorted = X[idx].copy(), y[idx].copy()

        # Initialize possible pivots
        possible_p = np.unique(X_sorted[:, feature_id])
        lenp = len(possible_p)

        if feature_id - 1 not in categorical:
            num = np.array([0, X_sorted.shape[0]])

            # Store entries of the Gram and moment matrices
            Moments = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [
                        np.sum(X_sorted[:, feature_id]),
                        np.sum(X_sorted[:, feature_id] ** 2),
                        np.sum(X_sorted[:, feature_id].copy().reshape(-1, 1) * y_sorted),
                        np.sum(y_sorted),
                        np.sum(y_sorted**2),
                    ],
                ]
            )

            # CON:
            if "con" in regression_nodes:
                intercept_con = Moments[1, 3] / num[1]
                coef_con = 0
                # Compute the RSS and the loss according to the information criterion
                rss = (
                    Moments[1, 4] + (num[1] * intercept_con**2) - 2 * intercept_con * Moments[1, 3]
                )
                loss = loss_fun(
                    criteria=split_criterion,
                    num=num[1],
                    Rss=np.array([rss]),
                    k=k_con,
                    coef_num=0,
                )
                # Update best_loss immediately
                if best_node == "" or loss.item() < best_loss:
                    best_node = "con"
                    best_loss = loss.item()
                    best_feature = feature_id
                    interval = np.array([possible_p[0], possible_p[-1]])
                    lm_L = np.array([coef_con, intercept_con])

            # LIN:
            if "lin" in regression_nodes and lenp >= min_unique_values_regression:
                var = num[1] * Moments[1, 1] - Moments[1, 0] ** 2
                # In case a constant feature
                if var == 0:
                    coef_lin = 0
                else:
                    coef_lin = (num[1] * Moments[1, 2] - Moments[1, 0] * Moments[1, 3]) / var
                intercept_lin = (Moments[1, 3] - coef_lin * Moments[1, 0]) / num[1]
                # Compute the RSS and the loss according to the information criterion
                rss = (
                    Moments[1, 4]
                    + (num[1] * intercept_lin**2)
                    + (2 * coef_lin * intercept_lin * Moments[1, 0])
                    + coef_lin**2 * Moments[1, 1]
                    - 2 * intercept_lin * Moments[1, 3]
                    - 2 * coef_lin * Moments[1, 2]
                )
                loss = loss_fun(
                    criteria=split_criterion,
                    num=num[1],
                    Rss=np.array([rss]),
                    k=k_lin,
                    coef_num=0,
                )
                # Update best_loss immediately
                if best_loss == "" or loss.item() < best_loss:
                    best_node = "lin"
                    best_loss = loss.item()
                    best_feature = feature_id
                    interval = np.array([possible_p[0], possible_p[-1]])
                    lm_L = np.array([coef_lin, intercept_lin])

            # For blin, we need to maintain another Gram/moment matrices and the knot xi
            if "blin" in regression_nodes:
                # Moments need to be updated for blin:
                # [sum(x-xi)+, sum[(x-xi)+]**2, sum[x(x-xi)+], sum[y(x-xi)+]]
                XtX = np.array(
                    [
                        [
                            np.float64(num.sum()),
                            Moments[:, 0].sum(),
                            Moments[:, 0].sum(),
                        ],
                        [Moments[:, 0].sum(), Moments[:, 1].sum(), Moments[:, 1].sum()],
                        [Moments[:, 0].sum(), Moments[:, 1].sum(), Moments[:, 1].sum()],
                    ]
                )
                XtY = np.array([[Moments[1, 3]], [Moments[1, 2]], [Moments[1, 2]]])
                pre_pivot = 0.0

            # pcon, blin and plin: try each possible split and
            # find the best one the last number are never used for split
            for p in range(possible_p.shape[0] - 1):
                # The pointer to select the column of coef and intercept
                i = 0
                pivot = possible_p[p]
                # Update cases in the left region
                index_add = X_sorted[:, feature_id] == pivot
                X_add = X_sorted[index_add, feature_id]
                y_add = y_sorted[index_add]

                # BLIN:
                if "blin" in regression_nodes:
                    # First maintain xi
                    xi = pivot - pre_pivot

                    # Update XtX and XtY
                    XtX += np.array(
                        [
                            [0.0, 0.0, -xi * num[1]],
                            [0.0, 0.0, -xi * Moments[1, 0]],
                            [
                                -xi * num[1],
                                -xi * Moments[1, 0],
                                xi**2 * num[1] - 2 * xi * XtX[0, 2],
                            ],
                        ]
                    )
                    XtY += np.array([[0.0], [0.0], [-xi * Moments[1, 3]]])

                    # Useless to check the first pivot or partition that
                    # leads to less than min_sample_leaf samples
                    if (
                        pivot != possible_p[0]
                        and p >= 1
                        and lenp >= min_unique_values_regression
                        and np.linalg.det(XtX) > 0.001
                        and num[0] + X_add.shape[0] >= min_sample_leaf
                        and num[1] - X_add.shape[0] >= min_sample_leaf
                    ):
                        coefs = np.linalg.solve(XtX, XtY).flatten()
                        coef[i, :] = np.array([coefs[1], coefs[1] + coefs[2]])
                        intercept[i, :] = np.array([coefs[0], coefs[0] - coefs[2] * pivot])
                    i += 1  # we add a dimension to the coef and intercept arrays
                    pre_pivot = pivot

                # Update num after blin is fitted
                num += np.array([1, -1]) * X_add.shape[0]

                # First update moments then check if this pivot is eligable for a pcon/plin split
                Moments_add = np.array(
                    [
                        np.sum(X_add),
                        np.sum(X_add**2),
                        np.sum(X_add.reshape(-1, 1) * y_add),
                        np.sum(y_add),
                        np.sum(y_add**2),
                    ]
                )
                Moments += Moments_add * np.array([[1.0], [-1.0]])

                # Negelect ineligable split
                if num[0] < min_sample_leaf:
                    continue
                elif num[1] < min_sample_leaf:
                    break

                # 'pcon' fit
                if "pcon" in regression_nodes:
                    coef[i, :] = np.array([0, 0])
                    intercept[i, :] = (Moments[:, 3]) / num
                    i += 1  # we add a dimension to the coef and intercept arrays

                # 'plin' for the first split candidate is equivalent to 'pcon'
                if (
                    "plin" in regression_nodes
                    and p
                    >= min_unique_values_regression
                    - 1  # Number of unique values smaller than current value
                    and lenp - p
                    >= min_unique_values_regression  # number of unique values larger than current value
                    and 0 not in num * Moments[:, 1] - Moments[:, 0] ** 2
                ):
                    # coef and intercept are vectors of dimension 1
                    # have to reshape X column in order to get correct cross product
                    # the intercept should be divided by the total number of samples
                    coef[i, :] = (num * Moments[:, 2] - Moments[:, 0] * Moments[:, 3]) / (
                        num * Moments[:, 1] - Moments[:, 0] ** 2
                    )
                    intercept[i, :] = (Moments[:, 3] - coef[i, :] * Moments[:, 0]) / num

                # Compute the rss and loss of the above 3 methods
                # The dimension rss is between 1 and 3 (depending on the regression_nodes)
                rss = (
                    Moments[:, 4]
                    + (num * intercept**2)
                    + (2 * coef * intercept * Moments[:, 0])
                    + coef**2 * Moments[:, 1]
                    - 2 * intercept * Moments[:, 3]
                    - 2 * coef * Moments[:, 2]
                ).sum(axis=1)

                # If no fit is done, continue
                if np.isnan(rss).all():
                    continue

                # Update the best loss
                rss = np.maximum(10**-8, rss)
                loss = loss_fun(
                    criteria=split_criterion,
                    num=num.sum(),
                    Rss=rss,
                    k=k_split_nodes,
                    coef_num=0,
                )

                if ~np.isnan(loss).all() and (best_node == "" or np.nanmin(loss) < best_loss):
                    best_loss = np.nanmin(loss)
                    index_min = np.where(loss == best_loss)[0][0]
                    add_index = 1 * ("lin" in regression_nodes) + 1 * ("con" in regression_nodes)
                    best_node = regression_nodes[add_index + index_min]
                    best_feature = feature_id  # asigned but will not be used for 'lin'
                    interval = np.array([possible_p[0], possible_p[-1]])
                    best_pivot = pivot
                    lm_L = np.array([coef[index_min, 0], intercept[index_min, 0]])
                    lm_R = np.array([coef[index_min, 1], intercept[index_min, 1]])

            continue

        # CATEGORICAL VARIABLES
        mean_vec = np.zeros(lenp)
        num_vec = np.zeros(lenp)
        for i in range(lenp):
            # Mean values of the response w.r.t. each level
            mean_vec[i] = np.mean(y_sorted[X_sorted[:, feature_id] == possible_p[i]])
            # Number of elements at each level
            num_vec[i] = y_sorted[X_sorted[:, feature_id] == possible_p[i]].shape[0]

        # Sort unique values w.r.t. the mean of the responses
        mean_idx = np.argsort(mean_vec)
        num_vec = num_vec[mean_idx]
        sum_vec = mean_vec[mean_idx] * num_vec
        possible_p = possible_p[mean_idx]

        # Loop over the sorted possible_p and find the best partition
        num = np.array([0.0, X_sorted.shape[0]])
        sum_all = np.array([0, np.sum(y_sorted)])
        for i in range(lenp - 1):
            # Update the sum and num
            sum_all += np.array([1.0, -1.0]) * sum_vec[i]
            num += np.array([1.0, -1.0]) * num_vec[i]
            # Find the indices of the elements in the left node
            sub_index = isin(X_sorted[:, feature_id], possible_p[: i + 1])
            # Compute the rss
            rss = np.sum((y_sorted[sub_index] - sum_all[0] / num[0]) ** 2) + np.sum(
                (y_sorted[~sub_index] - sum_all[1] / num[1]) ** 2
            )
            rss = np.maximum(10**-8, rss)
            loss = loss_fun(
                criteria=split_criterion,
                num=num.sum(),
                Rss=np.array([rss]),
                k=k_pconc,
                coef_num=0,
            )
            if best_node == "" or loss.item() < best_loss:
                best_feature = feature_id
                best_node = "pconc"
                best_loss = loss.item()
                lm_L = np.array([0, sum_all[0] / num[0]])
                lm_R = np.array([0, sum_all[1] / num[1]])
                pivot_c = possible_p[: i + 1].copy()
                pivot_c = pivot_c.astype(np.int64)

    return best_feature, best_pivot, best_node, lm_L, lm_R, interval, pivot_c

def best_split_nlfs(
    index,
    regression_nodes,
    n_features,
    sorted_X_indices,
    X,
    y,
    split_criterion,
    min_sample_leaf,
    k_con,
    k_lin,
    k_split_nodes,
    k_pconc,
    categorical,
    max_features_considered,
    min_unique_values_regression,
    alpha,
    nlfs_lars,
    should_lars,
    only_fallback,
):
     """
    Finds the best split using Node-Level Feature Selection (NLFS).

    This function extends `best_split` by first running a feature selection
    model (Lasso with fixed alpha or LARS) on the data within the current node. It then performs
    the standard `best_split` search but restricts it to the subset of features
    selected by NLFS. If NLFS selects no features, it falls back to a heuristic
    (top-k correlated features) or all features if the heuristic also fails.

    Args:
        (All args from `best_split`, plus):
        alpha (float): The regularization parameter for the node-level Lasso.
        nlfs_lars (bool): If True, use LARS instead of Lasso with fixed alpha for feature selection.
        should_lars (bool): A flag indicating if LARS should be attempted.
        only_fallback (bool): If True, only use the fallback strategy for feature selection.

    Returns:
        tuple: A tuple containing the best model's properties, same as `best_split`,
               plus the updated `should_lars` flag.
    """
    # Initialize variables, should be consistent with the variable type
    best_pivot = -1.0
    best_node = ""
    best_loss = -1.0
    best_feature = -1
    lm_L = np.array([0.0, 0.0])
    lm_R = np.array([0.0, 0.0])
    interval = np.array([-np.inf, np.inf])
    pivot_c = np.array([0])

    # Initialize the coef and intercept for 'blin'/'plin'/'pcon'
    l = (
        1 * ("blin" in regression_nodes)
        + 1 * ("plin" in regression_nodes)
        + 1 * ("pcon" in regression_nodes)
    )
    coef = np.zeros((l, 2)) * np.nan
    intercept = np.zeros((l, 2)) * np.nan
    
     # --- Part 1: Identify numerical feature INDICES (0-based original) ---
    all_original_feature_indices_0based = np.arange(n_features)
    is_numerical_mask = np.ones(n_features, dtype=np.bool_)
    if len(categorical) > 0 and not (len(categorical) == 1 and categorical[0] == -1):
        for cat_idx in categorical: 
            if 0 <= cat_idx < n_features:
                is_numerical_mask[cat_idx] = False
    numerical_feature_indices_0based = all_original_feature_indices_0based[is_numerical_mask]

    # --- Part 2: Determine selected_numerical_global_indices_0based ---
    
    # This variable will temporarily hold features selected by a successful Lasso/LARS run.
    # It's initialized as empty. If Lasso/LARS is successful and selects >0 features, this will be populated.
    # If Lasso/LARS is successful but selects 0 features, or if it fails/is skipped, this remains empty.
    _lasso_attempt_selected_features_0based = np.array([], dtype=np.int64)

    # Check if Lasso/LARS can be attempted
    can_attempt_nlfs = False
    can_attempt_lars = False
    if len(numerical_feature_indices_0based) > 0:
        numerical_cols_in_X_1based = numerical_feature_indices_0based + 1
        current_node_indices_arr = np.asarray(index, dtype=np.int64)

        if current_node_indices_arr.size > 0 and numerical_cols_in_X_1based.size > 0:
            X_node_numerical = X[np.ix_(current_node_indices_arr, numerical_cols_in_X_1based)]
        else:
            X_node_numerical = np.empty((0, len(numerical_feature_indices_0based)), dtype=X.dtype)
        
        y_node_target = y[current_node_indices_arr].ravel()

        # Only attempt NLFS if more than 10 observations and more than 0 features
        if X_node_numerical.shape[0] >= 10  and X_node_numerical.shape[1] > 0:
            can_attempt_nlfs = True

    # Only perform LARS if n > (p+1)
    if can_attempt_nlfs and X_node_numerical.shape[0] > X_node_numerical.shape[1] + 1:
        can_attempt_lars = True
    
    if can_attempt_nlfs and should_lars and not only_fallback:
        _temp_selected_indices_subset = np.array([], dtype=np.int64) # For objmode output
        nlfs_fit_successful = False
            
        try:
            lasso_kwargs = {'criterion': 'bic', 'max_iter': 500}
            with objmode(temp_indices_out='int64[:]'): # Define output var for objmode
                _selected_coefs_obj = None

                # NLFS with Lasso with fixed alpha
                if not nlfs_lars:
                    local_lasso = Lasso(alpha=alpha, max_iter=20000, copy_X=True, precompute=False,
                                        random_state=42, tol=1e-3, selection='cyclic')
                    local_lasso.fit(X_node_numerical, y_node_target)
                    _selected_coefs_obj = local_lasso.coef_
                # NLFS with LARS
                elif nlfs_lars and can_attempt_lars: 
                    lars_ic_model = LassoLarsIC(**lasso_kwargs)
                    lars_ic_model.fit(X_node_numerical, y_node_target)
                    _selected_coefs_obj = lars_ic_model.coef_
                
                if _selected_coefs_obj is not None and _selected_coefs_obj.size > 0:
                    temp_indices_out = np.where(_selected_coefs_obj != 0)[0]
                    nlfs_fit_successful = True # If no exception during fit/objmode
                else:
                    temp_indices_out = np.array([], dtype=np.int64)
            
            _temp_selected_indices_subset = temp_indices_out # Assign output from objmode

        except Exception as e: # Catch any error from Lasso/LARS fit or objmode
            print("An error occurred:", e)
            # _temp_selected_indices_subset remains empty
            # nlfs_fit_successful remains False
            pass 
            
        if nlfs_fit_successful and len(_temp_selected_indices_subset) > 0:
            _lasso_attempt_selected_features_0based = numerical_feature_indices_0based[_temp_selected_indices_subset]
        else:
            should_lars = False
        # If fit was successful but selected 0 features, _lasso_attempt_selected_features_0based remains empty.
        # If fit failed, _lasso_attempt_selected_features_0based also remains empty.

    # Now, decide the final selected_numerical_global_indices_0based
    if len(_lasso_attempt_selected_features_0based) > 0:
        # NLFS was attempted, was successful, and selected one or more features.
        selected_numerical_global_indices_0based = _lasso_attempt_selected_features_0based
    elif X_node_numerical.shape[1] < 6:
        # If NLFS was not succesful or selected 0 features
        # but number of features is smaller than 6, use them all
        selected_numerical_global_indices_0based = numerical_feature_indices_0based
    else:
        # Case: NLFS resulted in 0 numerical features (or was skipped/failed).
        # --- FALLBACK: Top K Numerical Features by Absolute Correlation ---
        # Default to no features selected by fallback
        selected_numerical_global_indices_0based = np.array([], dtype=np.int64)
        
        # We need X_node_numerical to have been formed and have >0 columns and >=2 rows.
        # numerical_feature_indices_0based tells us which original features these columns map to.
        if len(numerical_feature_indices_0based) > 0 and \
           'X_node_numerical' in locals() and X_node_numerical.shape[1] > 0 and \
           X_node_numerical.shape[0] >= 2: # Need at least 2 samples for correlation

            num_top_k = max(1, min(5, X_node_numerical.shape[1])) # Select 1 to 5 features, but not more than available
            
            correlations = np.zeros(X_node_numerical.shape[1]) # One per numerical feature column in X_node_numerical
            valid_correlations_exist = False

            for i in range(X_node_numerical.shape[1]):
                # Check for near-zero variance in the feature column
                # Use a small epsilon to avoid issues with floating point comparisons
                if (X_node_numerical[:, i].max() - X_node_numerical[:, i].min()) > 1e-9:
                    with objmode(corr_val='float64'): # Pearson correlation
                        # Ensure y_node_target also has variance for meaningful correlation
                        if (y_node_target.max() - y_node_target.min()) > 1e-9:
                            c = np.corrcoef(X_node_numerical[:, i], y_node_target.ravel())
                            # Handle potential NaNs from corrcoef (e.g., if one input is constant despite check)
                            corr_val = c[0, 1] if (c.shape == (2,2) and not np.isnan(c[0,1])) else 0.0
                        else:
                            corr_val = 0.0 # y is constant, no linear correlation possible
                    correlations[i] = np.abs(corr_val)
                    if correlations[i] > 1e-6: # Check if it's a non-trivial correlation
                        valid_correlations_exist = True
                else:
                    correlations[i] = 0.0 # No correlation if feature column is (near) constant
            
            if valid_correlations_exist:
                # Get indices of top K correlations (these are local indices within X_node_numerical's columns)
                # These indices correspond to the order in numerical_feature_indices_0based
                # if X_node_numerical was formed directly from numerical_feature_indices_0based.
                
                # argsort sorts in ascending order, so take from the end for largest absolute correlations
                sorted_corr_indices_local = np.argsort(correlations) 
                
                # Select up to num_top_k features that have a correlation > 1e-6
                top_k_candidates_local = []
                for local_idx in reversed(sorted_corr_indices_local): # Iterate from highest correlation
                    if correlations[local_idx] > 1e-6 and len(top_k_candidates_local) < num_top_k:
                        top_k_candidates_local.append(local_idx)
                    elif len(top_k_candidates_local) >= num_top_k:
                        break 
                
                if len(top_k_candidates_local) > 0:
                    # Map these local indices (within X_node_numerical columns) back to original global indices
                    selected_numerical_global_indices_0based = numerical_feature_indices_0based[np.array(top_k_candidates_local, dtype=np.int64)]
                else: 
                    selected_numerical_global_indices_0based = numerical_feature_indices_0based
            # If no valid correlations, selected_numerical_global_indices_0based remains empty

    # --- Part 3: Construct the final feature pool ---
    feature_pool_for_iteration_1based_set = set()

    # Add all original categorical features (1-based)
    if len(categorical) > 0 and not (len(categorical) == 1 and categorical[0] == -1):
        for cat_idx_0based in categorical:
            if 0 <= cat_idx_0based < n_features:
                feature_pool_for_iteration_1based_set.add(cat_idx_0based + 1)

    # Add selected_numerical_global_indices_0based (which is now from Lasso OR fallback)
    for num_idx_0based in selected_numerical_global_indices_0based:
        feature_pool_for_iteration_1based_set.add(num_idx_0based + 1)

    # Fallback for feature_pool_for_loop (this is the second level of fallback, if the set is still empty)
    if not feature_pool_for_iteration_1based_set and n_features > 0:
        feature_pool_for_loop = np.arange(1, n_features + 1, dtype=np.int64)
    elif not feature_pool_for_iteration_1based_set and n_features == 0:
        feature_pool_for_loop = np.array([], dtype=np.int64)
    else: # feature_pool_for_iteration_1based_set is NOT empty
        feature_pool_for_loop = np.array(list(feature_pool_for_iteration_1based_set), dtype=np.int64)

    
    if len(feature_pool_for_loop) > 0:
        # Search for the best split among the selected features
        for feature_id in feature_pool_for_loop:
            # Get sorted X, y
            idx = sorted_X_indices[feature_id - 1]
            idx = idx[isin(idx, index)]
            X_sorted, y_sorted = X[idx].copy(), y[idx].copy()
    
            # Initialize possible pivots
            possible_p = np.unique(X_sorted[:, feature_id])
            lenp = len(possible_p)
    
            if feature_id - 1 not in categorical:
                num = np.array([0, X_sorted.shape[0]])
    
                # store entries of the Gram and moment matrices
                Moments = np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [
                            np.sum(X_sorted[:, feature_id]),
                            np.sum(X_sorted[:, feature_id] ** 2),
                            np.sum(X_sorted[:, feature_id].copy().reshape(-1, 1) * y_sorted),
                            np.sum(y_sorted),
                            np.sum(y_sorted**2),
                        ],
                    ]
                )
    
                # CON:
                if "con" in regression_nodes:
                    intercept_con = Moments[1, 3] / num[1]
                    coef_con = 0
                    # Compute the RSS and the loss according to the information criterion
                    rss = (
                        Moments[1, 4] + (num[1] * intercept_con**2) - 2 * intercept_con * Moments[1, 3]
                    )
                    loss = loss_fun(
                        criteria=split_criterion,
                        num=num[1],
                        Rss=np.array([rss]),
                        k=k_con,
                        coef_num=0,
                    )
                    # Update best_loss immediately
                    if best_node == "" or loss.item() < best_loss:
                        best_node = "con"
                        best_loss = loss.item()
                        best_feature = feature_id
                        interval = np.array([possible_p[0], possible_p[-1]])
                        lm_L = np.array([coef_con, intercept_con])
    
                # LIN:
                if "lin" in regression_nodes and lenp >= min_unique_values_regression:
                    var = num[1] * Moments[1, 1] - Moments[1, 0] ** 2
                    # In case a constant feature
                    if var == 0:
                        coef_lin = 0
                    else:
                        coef_lin = (num[1] * Moments[1, 2] - Moments[1, 0] * Moments[1, 3]) / var
                    intercept_lin = (Moments[1, 3] - coef_lin * Moments[1, 0]) / num[1]
                    # Compute the RSS and the loss according to the information criterion
                    rss = (
                        Moments[1, 4]
                        + (num[1] * intercept_lin**2)
                        + (2 * coef_lin * intercept_lin * Moments[1, 0])
                        + coef_lin**2 * Moments[1, 1]
                        - 2 * intercept_lin * Moments[1, 3]
                        - 2 * coef_lin * Moments[1, 2]
                    )
                    loss = loss_fun(
                        criteria=split_criterion,
                        num=num[1],
                        Rss=np.array([rss]),
                        k=k_lin,
                        coef_num=0
                    )
                    # Update best_loss immediately
                    if best_loss == "" or loss.item() < best_loss:
                        best_node = "lin"
                        best_loss = loss.item()
                        best_feature = feature_id
                        interval = np.array([possible_p[0], possible_p[-1]])
                        lm_L = np.array([coef_lin, intercept_lin])
    
                # For blin, we need to maintain another Gram/moment matrices and the knot xi
                if "blin" in regression_nodes:
                    # Moments need to be updated for blin:
                    # [sum(x-xi)+, sum[(x-xi)+]**2, sum[x(x-xi)+], sum[y(x-xi)+]]
                    XtX = np.array(
                        [
                            [
                                np.float64(num.sum()),
                                Moments[:, 0].sum(),
                                Moments[:, 0].sum(),
                            ],
                            [Moments[:, 0].sum(), Moments[:, 1].sum(), Moments[:, 1].sum()],
                            [Moments[:, 0].sum(), Moments[:, 1].sum(), Moments[:, 1].sum()],
                        ]
                    )
                    XtY = np.array([[Moments[1, 3]], [Moments[1, 2]], [Moments[1, 2]]])
                    pre_pivot = 0.0
    
                # pcon, blin and plin: try each possible split and
                # find the best one the last number are never used for split
                for p in range(possible_p.shape[0] - 1):
                    # The pointer to select the column of coef and intercept
                    i = 0
                    pivot = possible_p[p]
                    # Update cases in the left region
                    index_add = X_sorted[:, feature_id] == pivot
                    X_add = X_sorted[index_add, feature_id]
                    y_add = y_sorted[index_add]
    
                    # BLIN:
                    if "blin" in regression_nodes:
                        # First maintain xi
                        xi = pivot - pre_pivot
    
                        # Update XtX and XtY
                        XtX += np.array(
                            [
                                [0.0, 0.0, -xi * num[1]],
                                [0.0, 0.0, -xi * Moments[1, 0]],
                                [
                                    -xi * num[1],
                                    -xi * Moments[1, 0],
                                    xi**2 * num[1] - 2 * xi * XtX[0, 2],
                                ],
                            ]
                        )
                        XtY += np.array([[0.0], [0.0], [-xi * Moments[1, 3]]])
    
                        # Useless to check the first pivot or partition that
                        # leads to less than min_sample_leaf samples
                        if (
                            pivot != possible_p[0]
                            and p >= 1
                            and lenp >= min_unique_values_regression
                            and np.linalg.det(XtX) > 0.001
                            and num[0] + X_add.shape[0] >= min_sample_leaf
                            and num[1] - X_add.shape[0] >= min_sample_leaf
                        ):
                            coefs = np.linalg.solve(XtX, XtY).flatten()
                            coef[i, :] = np.array([coefs[1], coefs[1] + coefs[2]])
                            intercept[i, :] = np.array([coefs[0], coefs[0] - coefs[2] * pivot])
                        i += 1  # we add a dimension to the coef and intercept arrays
                        pre_pivot = pivot
    
                    # Update num after blin is fitted
                    num += np.array([1, -1]) * X_add.shape[0]
    
                    # First update moments then check if this pivot is eligable for a pcon/plin split
                    Moments_add = np.array(
                        [
                            np.sum(X_add),
                            np.sum(X_add**2),
                            np.sum(X_add.reshape(-1, 1) * y_add),
                            np.sum(y_add),
                            np.sum(y_add**2),
                        ]
                    )
                    Moments += Moments_add * np.array([[1.0], [-1.0]])
    
                    # Negelect ineligable split
                    if num[0] < min_sample_leaf:
                        continue
                    elif num[1] < min_sample_leaf:
                        break
    
                    # 'pcon' fit
                    if "pcon" in regression_nodes:
                        coef[i, :] = np.array([0, 0])
                        intercept[i, :] = (Moments[:, 3]) / num
                        i += 1  # We add a dimension to the coef and intercept arrays
    
                    # 'plin' for the first split candidate is equivalent to 'pcon'
                    if (
                        "plin" in regression_nodes
                        and p
                        >= min_unique_values_regression
                        - 1  # Number of unique values smaller than current value
                        and lenp - p
                        >= min_unique_values_regression  # Number of unique values larger than current value
                        and 0 not in num * Moments[:, 1] - Moments[:, 0] ** 2
                    ):
                        # coef and intercept are vectors of dimension 1
                        # have to reshape X column in order to get correct cross product
                        # the intercept should be divided by the total number of samples
                        coef[i, :] = (num * Moments[:, 2] - Moments[:, 0] * Moments[:, 3]) / (
                            num * Moments[:, 1] - Moments[:, 0] ** 2
                        )
                        intercept[i, :] = (Moments[:, 3] - coef[i, :] * Moments[:, 0]) / num
    
                    # Compute the rss and loss of the above 3 methods
                    # The dimension rss is between 1 and 3 (depending on the regression_nodes)
                    rss = (
                        Moments[:, 4]
                        + (num * intercept**2)
                        + (2 * coef * intercept * Moments[:, 0])
                        + coef**2 * Moments[:, 1]
                        - 2 * intercept * Moments[:, 3]
                        - 2 * coef * Moments[:, 2]
                    ).sum(axis=1)
    
                    # If no fit is done, continue
                    if np.isnan(rss).all():
                        continue
    
                    # Update the best loss
                    rss = np.maximum(10**-8, rss)
                    loss = loss_fun(
                        criteria=split_criterion,
                        num=num.sum(),
                        Rss=rss,
                        k=k_split_nodes,
                        coef_num=0,
                    )
    
                    if ~np.isnan(loss).all() and (best_node == "" or np.nanmin(loss) < best_loss):
                        best_loss = np.nanmin(loss)
                        index_min = np.where(loss == best_loss)[0][0]
                        add_index = 1 * ("lin" in regression_nodes) + 1 * ("con" in regression_nodes)
                        best_node = regression_nodes[add_index + index_min]
                        best_feature = feature_id  # asigned but will not be used for 'lin'
                        interval = np.array([possible_p[0], possible_p[-1]])
                        best_pivot = pivot
                        lm_L = np.array([coef[index_min, 0], intercept[index_min, 0]])
                        lm_R = np.array([coef[index_min, 1], intercept[index_min, 1]])
    
                continue
    
            # CATEGORICAL VARIABLES
            mean_vec = np.zeros(lenp)
            num_vec = np.zeros(lenp)
            for i in range(lenp):
                # Mean values of the response w.r.t. each level
                mean_vec[i] = np.mean(y_sorted[X_sorted[:, feature_id] == possible_p[i]])
                # Number of elements at each level
                num_vec[i] = y_sorted[X_sorted[:, feature_id] == possible_p[i]].shape[0]
    
            # Sort unique values w.r.t. the mean of the responses
            mean_idx = np.argsort(mean_vec)
            num_vec = num_vec[mean_idx]
            sum_vec = mean_vec[mean_idx] * num_vec
            possible_p = possible_p[mean_idx]
    
            # Loop over the sorted possible_p and find the best partition
            num = np.array([0.0, X_sorted.shape[0]])
            sum_all = np.array([0, np.sum(y_sorted)])
            for i in range(lenp - 1):
                # Update the sum and num
                sum_all += np.array([1.0, -1.0]) * sum_vec[i]
                num += np.array([1.0, -1.0]) * num_vec[i]
                # Find the indices of the elements in the left node
                sub_index = isin(X_sorted[:, feature_id], possible_p[: i + 1])
                # Compute the rss
                rss = np.sum((y_sorted[sub_index] - sum_all[0] / num[0]) ** 2) + np.sum(
                    (y_sorted[~sub_index] - sum_all[1] / num[1]) ** 2
                )
                rss = np.maximum(10**-8, rss)
                loss = loss_fun(
                    criteria=split_criterion,
                    num=num.sum(),
                    Rss=np.array([rss]),
                    k=k_pconc,
                    coef_num=0,
                )
                if best_node == "" or loss.item() < best_loss:
                    best_feature = feature_id
                    best_node = "pconc"
                    best_loss = loss.item()
                    lm_L = np.array([0, sum_all[0] / num[0]])
                    lm_R = np.array([0, sum_all[1] / num[1]])
                    pivot_c = possible_p[: i + 1].copy()
                    pivot_c = pivot_c.astype(np.int64)

    return best_feature, best_pivot, best_node, lm_L, lm_R, interval, pivot_c, should_lars
    
def best_split_multi(
    index,
    regression_nodes,
    n_features,
    sorted_X_indices,
    X,
    y,
    split_criterion,
    min_sample_leaf,
    k_con,
    k_lin,
    k_split_nodes,
    k_pconc,
    k_lns,
    k_ls,
    categorical,
    max_features_considered,
    min_unique_values_regression,
    alpha,
    multi_lars,
    finalist_s,
    finalist_d,
    per_feature,
    full_multi,
):
    """
    Finds the best split, considering both univariate and multivariate models.

    This is the most advanced split-finding function. It evaluates all standard
    univariate models (from `best_split`) and also considers replacing them with
    more complex multivariate models (fit using Lasso/LARS). It supports
    several strategies for how and when to introduce these multivariate models.

    Args:
        (All args from `best_split`, plus):
        k_lns, k_ls (np.ndarray): Degrees of freedom for multivariate nodes.
        alpha (float): Regularization parameter for multivariate models.
        multi_lars (bool): If True, use LARS; otherwise, use Lasso with fixed alpha.
        finalist_s, finalist_d, per_feature, full_multi (bool): Flags to
            select the specific multivariate strategy to use.

    Returns:
        tuple: A tuple containing the best model's properties, including
            new parameters for the multivariate models if one was chosen:
            - (All outputs from `best_split`)
            - best_multi_indices_L/R (np.ndarray): Indices of features in the MV model.
            - best_multi_coeffs_L/R (np.ndarray): Coefficients for the MV model.
            - best_multi_intercept_L/R (float): Intercept for the MV model.
            - best_node_prev (str): The original node type if it was upgraded.
    """
    # Initialize variables, should be consistent with the variable type
    best_pivot = -1.0
    best_node = ""
    best_loss = -1.0
    best_feature = -1
    lm_L = np.array([0.0, 0.0])
    lm_R = np.array([0.0, 0.0])
    interval = np.array([-np.inf, np.inf])
    pivot_c = np.array([0])
    
    # --- MULTIVARIATE INITIALIZATIONS ---
    best_multi_indices_L = np.array([], dtype=np.int64)
    best_multi_coeffs_L = np.array([], dtype=np.float64)
    best_multi_intercept_L = 0.0
    
    best_multi_indices_R = np.array([], dtype=np.int64)
    best_multi_coeffs_R = np.array([], dtype=np.float64)
    best_multi_intercept_R = 0.0 
    
    best_node_prev = ""
    best_loss_split = -1.0
    best_feature_split = -1
    best_pivot_split = -1
    pivot_c_split = np.array([0])

    # Initialize the coef and intercept for 'blin'/'plin'/'pcon'
    l = (
        1 * ("blin" in regression_nodes)
        + 1 * ("plin" in regression_nodes)
        + 1 * ("pcon" in regression_nodes)
    )
    coef = np.zeros((l, 2)) * np.nan
    intercept = np.zeros((l, 2)) * np.nan  
    
     # --- Part 1: Identify numerical feature INDICES (0-based original) ---
    all_original_feature_indices_0based = np.arange(n_features)
    is_numerical_mask = np.ones(n_features, dtype=np.bool_)
    if len(categorical) > 0 and not (len(categorical) == 1 and categorical[0] == -1):
        for cat_idx in categorical: 
            if 0 <= cat_idx < n_features:
                is_numerical_mask[cat_idx] = False
    numerical_feature_indices_0based = all_original_feature_indices_0based[is_numerical_mask]

    # --- Part 2: ---
    can_attempt_lasso = False
    if len(numerical_feature_indices_0based) > 0:
        numerical_cols_in_X_1based = numerical_feature_indices_0based + 1
        current_node_indices_arr = np.asarray(index, dtype=np.int64)

        if current_node_indices_arr.size > 0 and numerical_cols_in_X_1based.size > 0:
            X_node_numerical = X[np.ix_(current_node_indices_arr, numerical_cols_in_X_1based)]
        else:
            X_node_numerical = np.empty((0, len(numerical_feature_indices_0based)), dtype=X.dtype)
        
        y_node_target = y[current_node_indices_arr].ravel()

        if X_node_numerical.shape[0] >= 10  and X_node_numerical.shape[1] > 0:
            can_attempt_lasso = True 
            
    # Determine (max) 5 features with highest importance from heuristics
    # for per_feature and full_multi strategies
    top_k_feature_indices_0based = np.array([], dtype=np.int64)        
    if (per_feature or full_multi) and can_attempt_lasso and ("lasso_split" in regression_nodes):
        # --- Guardrail: If there are very few features, just use all of them ---
        if n_features <= 5:
            top_k_feature_indices_0based = np.arange(n_features)
        else:
            # --- Heuristic Selection: Use scouts to find the most promising features ---
            # First, derive the categorical indices from the numerical ones
            is_categorical_mask = ~isin(np.arange(n_features), numerical_feature_indices_0based)
            categorical_indices_0based = np.arange(n_features)[is_categorical_mask]

            # Use objmode to run scikit-learn and scipy functions
            with objmode(top_k_indices_out='int64[:]'):
                # --- 1. Numerical Scout ---
                numerical_scout_indices = np.array([], dtype=np.int64)
                if X_node_numerical.shape[1] > 0:
                    # Configure a cheap, fast scout model
                    scout_model = ExtraTreesRegressor(
                        n_estimators=20,
                        max_depth=4,
                        n_jobs=1,
                        random_state=42
                    )
                    scout_model.fit(X_node_numerical, y_node_target)
                    importances = scout_model.feature_importances_
                    
                    # Select up to top 4 numerical features
                    num_top_numerical = min(4, len(importances))
                    top_k_local_indices = np.argsort(importances)[-num_top_numerical:]
                    
                    # Map local numerical indices back to original global indices
                    numerical_scout_indices = numerical_feature_indices_0based[top_k_local_indices]

                # --- 2. Categorical Scout ---
                categorical_scout_indices = np.array([], dtype=np.int64)
                if len(categorical_indices_0based) > 0:
                    X_node = X[index, :] # Get the full data for the node
                    all_f_scores = []

                    for cat_idx in categorical_indices_0based:
                        # Group the target 'y' by the levels of the categorical feature
                        unique_levels = np.unique(X_node[:, cat_idx + 1]) # +1 to account for index column
                        groups = [y_node_target[X_node[:, cat_idx + 1] == level] for level in unique_levels]
                        
                        # ANOVA requires at least 2 groups with more than 1 sample each
                        valid_groups = [g for g in groups if len(g) > 1]
                        if len(valid_groups) > 1:
                            f_stat, _ = f_oneway(*valid_groups)
                            all_f_scores.append(f_stat)
                        else:
                            all_f_scores.append(0.0) # No predictive power if it can't be tested
                    
                    # Select the single best categorical feature
                    if len(all_f_scores) > 0:
                        top_categorical_local_index = np.argmax(np.array(all_f_scores))
                        categorical_scout_indices = np.array([categorical_indices_0based[top_categorical_local_index]], dtype=np.int64)

                # --- 3. Combine Finalists ---
                # Use np.union1d to get a sorted, unique array of the top features
                top_k_indices_out = np.union1d(numerical_scout_indices, categorical_scout_indices)

            # Assign the result from objmode back to our Numba variable
            top_k_feature_indices_0based = top_k_indices_out.astype(np.int64)  
    
    # Search for the best split among all features, negelecting the indices column
    for feature_id in random_sample(np.arange(1, n_features + 1), max_features_considered):
        # Get sorted X, y
        idx = sorted_X_indices[feature_id - 1]
        idx = idx[isin(idx, index)]
        X_sorted, y_sorted = X[idx].copy(), y[idx].copy()

        # Initialize possible pivots
        possible_p = np.unique(X_sorted[:, feature_id])
        lenp = len(possible_p)
        
        # Initialize for per_feature strategy
        best_loss_feat = -1.0
        best_node_feat = ""
        best_pivot_feat = -1
        pivot_c_feat = np.array([0])

        if feature_id - 1 not in categorical:
            num = np.array([0, X_sorted.shape[0]])

            # store entries of the Gram and moment matrices
            Moments = np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [
                        np.sum(X_sorted[:, feature_id]),
                        np.sum(X_sorted[:, feature_id] ** 2),
                        np.sum(X_sorted[:, feature_id].copy().reshape(-1, 1) * y_sorted),
                        np.sum(y_sorted),
                        np.sum(y_sorted**2),
                    ],
                ]
            )

            # CON:
            if "con" in regression_nodes:
                intercept_con = Moments[1, 3] / num[1]
                coef_con = 0
                # Compute the RSS and the loss according to the information criterion
                rss = (
                    Moments[1, 4] + (num[1] * intercept_con**2) - 2 * intercept_con * Moments[1, 3]
                )
                loss = loss_fun(
                    criteria=split_criterion,
                    num=num[1],
                    Rss=np.array([rss]),
                    k=k_con,
                    coef_num=0,
                )
                # Update best_loss immediately
                if best_node == "" or loss.item() < best_loss:
                    best_node = "con"
                    best_loss = loss.item()
                    best_feature = feature_id
                    interval = np.array([possible_p[0], possible_p[-1]])
                    lm_L = np.array([coef_con, intercept_con])

            # LIN:
            if "lin" in regression_nodes and lenp >= min_unique_values_regression:
                var = num[1] * Moments[1, 1] - Moments[1, 0] ** 2
                # In case a constant feature
                if var == 0:
                    coef_lin = 0
                else:
                    coef_lin = (num[1] * Moments[1, 2] - Moments[1, 0] * Moments[1, 3]) / var
                intercept_lin = (Moments[1, 3] - coef_lin * Moments[1, 0]) / num[1]
                # Compute the RSS and the loss according to the information criterion
                rss = (
                    Moments[1, 4]
                    + (num[1] * intercept_lin**2)
                    + (2 * coef_lin * intercept_lin * Moments[1, 0])
                    + coef_lin**2 * Moments[1, 1]
                    - 2 * intercept_lin * Moments[1, 3]
                    - 2 * coef_lin * Moments[1, 2]
                )
                loss = loss_fun(
                    criteria=split_criterion,
                    num=num[1],
                    Rss=np.array([rss]),
                    k=k_lin,
                    coef_num=0,
                )
                # Update best_loss immediately
                if best_loss == "" or loss.item() < best_loss:
                    best_node = "lin"
                    best_loss = loss.item()
                    best_feature = feature_id
                    interval = np.array([possible_p[0], possible_p[-1]])
                    lm_L = np.array([coef_lin, intercept_lin])

            # For blin, we need to maintain another Gram/moment matrices and the knot xi
            if "blin" in regression_nodes:
                # Moments need to be updated for blin:
                # [sum(x-xi)+, sum[(x-xi)+]**2, sum[x(x-xi)+], sum[y(x-xi)+]]
                XtX = np.array(
                    [
                        [
                            np.float64(num.sum()),
                            Moments[:, 0].sum(),
                            Moments[:, 0].sum(),
                        ],
                        [Moments[:, 0].sum(), Moments[:, 1].sum(), Moments[:, 1].sum()],
                        [Moments[:, 0].sum(), Moments[:, 1].sum(), Moments[:, 1].sum()],
                    ]
                )
                XtY = np.array([[Moments[1, 3]], [Moments[1, 2]], [Moments[1, 2]]])
                pre_pivot = 0.0

            # pcon, blin and plin: try each possible split and
            # find the best one the last number are never used for split
            for p in range(possible_p.shape[0] - 1):
                # The pointer to select the column of coef and intercept
                i = 0
                pivot = possible_p[p]
                # Update cases in the left region
                index_add = X_sorted[:, feature_id] == pivot
                X_add = X_sorted[index_add, feature_id]
                y_add = y_sorted[index_add]

                # BLIN:
                if "blin" in regression_nodes:
                    # First maintain xi
                    xi = pivot - pre_pivot

                    # Update XtX and XtY
                    XtX += np.array(
                        [
                            [0.0, 0.0, -xi * num[1]],
                            [0.0, 0.0, -xi * Moments[1, 0]],
                            [
                                -xi * num[1],
                                -xi * Moments[1, 0],
                                xi**2 * num[1] - 2 * xi * XtX[0, 2],
                            ],
                        ]
                    )
                    XtY += np.array([[0.0], [0.0], [-xi * Moments[1, 3]]])

                    # Useless to check the first pivot or partition that
                    # leads to less than min_sample_leaf samples
                    if (
                        pivot != possible_p[0]
                        and p >= 1
                        and lenp >= min_unique_values_regression
                        and np.linalg.det(XtX) > 0.001
                        and num[0] + X_add.shape[0] >= min_sample_leaf
                        and num[1] - X_add.shape[0] >= min_sample_leaf
                    ):
                        coefs = np.linalg.solve(XtX, XtY).flatten()
                        coef[i, :] = np.array([coefs[1], coefs[1] + coefs[2]])
                        intercept[i, :] = np.array([coefs[0], coefs[0] - coefs[2] * pivot])
                    i += 1  # We add a dimension to the coef and intercept arrays
                    pre_pivot = pivot

                # Update num after blin is fitted
                num += np.array([1, -1]) * X_add.shape[0]

                # First update moments then check if this pivot is eligable for a pcon/plin split
                Moments_add = np.array(
                    [
                        np.sum(X_add),
                        np.sum(X_add**2),
                        np.sum(X_add.reshape(-1, 1) * y_add),
                        np.sum(y_add),
                        np.sum(y_add**2),
                    ]
                )
                Moments += Moments_add * np.array([[1.0], [-1.0]])

                # Negelect ineligable split
                if num[0] < min_sample_leaf:
                    continue
                elif num[1] < min_sample_leaf:
                    break

                # 'pcon' fit
                if "pcon" in regression_nodes:
                    coef[i, :] = np.array([0, 0])
                    intercept[i, :] = (Moments[:, 3]) / num
                    i += 1  # We add a dimension to the coef and intercept arrays

                # 'plin' for the first split candidate is equivalent to 'pcon'
                if (
                    "plin" in regression_nodes
                    and p
                    >= min_unique_values_regression
                    - 1  # Number of unique values smaller than current value
                    and lenp - p
                    >= min_unique_values_regression  # Number of unique values larger than current value
                    and 0 not in num * Moments[:, 1] - Moments[:, 0] ** 2
                ):
                    # coef and intercept are vectors of dimension 1
                    # have to reshape X column in order to get correct cross product
                    # the intercept should be divided by the total number of samples
                    coef[i, :] = (num * Moments[:, 2] - Moments[:, 0] * Moments[:, 3]) / (
                        num * Moments[:, 1] - Moments[:, 0] ** 2
                    )
                    intercept[i, :] = (Moments[:, 3] - coef[i, :] * Moments[:, 0]) / num

                # Compute the rss and loss of the above 3 methods
                # The dimension rss is between 1 and 3 (depending on the regression_nodes)
                rss = (
                    Moments[:, 4]
                    + (num * intercept**2)
                    + (2 * coef * intercept * Moments[:, 0])
                    + coef**2 * Moments[:, 1]
                    - 2 * intercept * Moments[:, 3]
                    - 2 * coef * Moments[:, 2]
                ).sum(axis=1)

                # If no fit is done, continue
                if np.isnan(rss).all():
                    continue

                # Update the best loss
                rss = np.maximum(10**-8, rss)
                loss = loss_fun(
                    criteria=split_criterion,
                    num=num.sum(),
                    Rss=rss,
                    k=k_split_nodes,
                    coef_num=0,
                )

                if ~np.isnan(loss).all() and (best_node == "" or np.nanmin(loss) < best_loss):
                    best_loss = np.nanmin(loss)
                    index_min = np.where(loss == best_loss)[0][0]
                    add_index = 1 * ("lin" in regression_nodes) + 1 * ("con" in regression_nodes)
                    best_node = regression_nodes[add_index + index_min]
                    best_feature = feature_id  # asigned but will not be used for 'lin'
                    interval = np.array([possible_p[0], possible_p[-1]])
                    best_pivot = pivot
                    lm_L = np.array([coef[index_min, 0], intercept[index_min, 0]])
                    lm_R = np.array([coef[index_min, 1], intercept[index_min, 1]])

                # Determine best split for finalist_d stategy
                if finalist_d:   
                    if ~np.isnan(loss).all() and (best_node_prev == "" or np.nanmin(loss) < best_loss_split):
                        best_loss_split = np.nanmin(loss)
                        index_min = np.where(loss == best_loss_split)[0][0]
                        add_index = 1 * ("lin" in regression_nodes) + 1 * ("con" in regression_nodes)
                        best_node_prev = regression_nodes[add_index + index_min]
                        best_feature_split = feature_id  
                        best_pivot_split = pivot

                # Determine best split for per_feature and full_multi stategies
                if (per_feature or full_multi) and (feature_id - 1) in top_k_feature_indices_0based:
                    if ~np.isnan(loss).all() and (best_node_feat == "" or np.nanmin(loss) < best_loss_feat):
                        best_loss_feat = np.nanmin(loss)
                        index_min = np.where(loss == best_loss_feat)[0][0]
                        add_index = 1 * ("lin" in regression_nodes) + 1 * ("con" in regression_nodes)
                        best_node_feat = regression_nodes[add_index + index_min]
                        best_pivot_feat = pivot   

            # Make the Lasso/LARS models for per_feature and full_multi stategies
            if (per_feature or full_multi) and (feature_id - 1) in top_k_feature_indices_0based:
                if best_node_feat != "":
                    left_indices = np.array([], dtype=np.int64)
                    right_indices = np.array([], dtype=np.int64)
                    pivot_possible = np.empty(0, dtype=np.float64) 
                                    
                    # Get the feature values for all data points in the current node
                    feature_values_in_node = X[index, feature_id]

                    # Determine the split points
                    if per_feature:
                        pivot_possible = np.array([best_pivot_feat], dtype=np.float64)
                    elif full_multi:
                        with objmode(quantile_pivots_out='float64[:]'):
                            feature_values_in_node = X[index, feature_id] # Get data inside objmode
                            q_pivots = np.quantile(feature_values_in_node, q=[0.25, 0.50, 0.75], interpolation='linear')
                            quantile_pivots_out = np.unique(q_pivots)
                        
                        # Get the maximum value in the current node for this feature
                        max_val = feature_values_in_node.max()
                        
                        # Only consider pivots that are strictly less than the max value.
                        # This guarantees that the right child will never be empty.
                        pivot_possible = quantile_pivots_out[quantile_pivots_out < max_val]
                               
                    for pivot_opt in pivot_possible:     
                        # Determine the indices for left and right child
                        left_mask = feature_values_in_node <= pivot_opt
                        left_indices = index[left_mask]
                        right_indices = index[~left_mask]

                        # Check if the sizes in both children are correct (>0)
                        if left_indices.size > 0 and right_indices.size > 0:
                            X_left_child = X[np.ix_(left_indices, numerical_cols_in_X_1based)]
                            y_left_child = y[left_indices].ravel()
                    
                            X_right_child = X[np.ix_(right_indices, numerical_cols_in_X_1based)]
                            y_right_child = y[right_indices].ravel()
                    
                            # Check if children have enough samples to fit Lasso
                            if X_left_child.shape[0] >= min_sample_leaf and X_right_child.shape[0] >= min_sample_leaf:
                                # --- Initialize variables for this split attempt ---
                                # Using np.nan helps catch errors if a fit fails
                                rss_L, rss_R = np.nan, np.nan
                                coeffs_L, coeffs_R = np.array([-1.0]), np.array([-1.0])
                                intercept_L, intercept_R = np.nan, np.nan
                                
                                # --- Fit Lasso on LEFT child ---
                                try:
                                
                                    if multi_lars:
                                        lasso_kwargs_L = {'criterion': 'bic', 'max_iter': 500}
                                        if X_left_child.shape[0] <= X_left_child.shape[1] + 1:
                                            noise_var_L = np.var(y_left_child)
                                            lasso_kwargs_L['noise_variance'] = noise_var_L + 1e-9
                                        
                                    with objmode(
                                        rss_L_out='float64', 
                                        coeffs_L_out='float64[:]', 
                                        intercept_L_out='float64'
                                    ):
                                        # This entire block runs in standard Python mode
                                        
                                        # 1. Create and fit the model
                                        if multi_lars:
                                            lasso_L_model = LassoLarsIC(**lasso_kwargs_L)
                                            lasso_L_model.fit(X_left_child, y_left_child)
                                        else:
                                            lasso_L_model = Lasso(alpha=alpha, max_iter=20000, copy_X=True, precompute=False, 
                                                        random_state=42, tol=1e-3, selection='cyclic')
                                            lasso_L_model.fit(X_left_child, y_left_child)
                                        
                                        # 2. Predict on the training data to calculate RSS
                                        if lasso_L_model is not None:
                                            y_pred_L = lasso_L_model.predict(X_left_child)
                                            
                                            # 3. Calculate RSS
                                            residuals_L = y_left_child - y_pred_L
                                            rss_L_out = np.sum(residuals_L**2)
                                            
                                            # 4. Extract model parameters to pass back to Numba
                                            coeffs_L_out = lasso_L_model.coef_
                                            intercept_L_out = lasso_L_model.intercept_
                                
                                    # Assign the results from objmode back to our Numba-compatible variables
                                    rss_L = rss_L_out
                                    coeffs_L = coeffs_L_out
                                    intercept_L = intercept_L_out
                                
                                except Exception as e:
                                    print("An error occurred:", e)
                                    # If the left fit fails for any reason, we can't proceed with this split.
                                    # We'll break out of this block; the `np.isnan(rss_L)` check later will handle it.
                                    pass
                                
                                # --- Fit Lasso on RIGHT child ---
                                # Only proceed if the left fit was successful
                                if not np.isnan(rss_L):
                                    try:
                                        if multi_lars:
                                            lasso_kwargs_R = {'criterion': 'bic', 'max_iter': 500}
                                            if X_right_child.shape[0] <= X_right_child.shape[1] + 1:
                                                noise_var_R = np.var(y_right_child)
                                                lasso_kwargs_R['noise_variance'] = noise_var_R + 1e-9
                                            
                                        with objmode(
                                            rss_R_out='float64', 
                                            coeffs_R_out='float64[:]', 
                                            intercept_R_out='float64'
                                        ):
                                            # 1. Create and fit the model
                                            if multi_lars:
                                                lasso_R_model = LassoLarsIC(**lasso_kwargs_R)
                                                lasso_R_model.fit(X_right_child, y_right_child)
                                            else:
                                                lasso_R_model = Lasso(alpha=alpha, max_iter=20000, copy_X=True, precompute=False, 
                                                        random_state=42, tol=1e-3, selection='cyclic')
                                                lasso_R_model.fit(X_right_child, y_right_child)
                                            
                                            # 2. Predict to calculate RSS
                                            if lasso_R_model is not None:
                                                y_pred_R = lasso_R_model.predict(X_right_child)
                                                
                                                # 3. Calculate RSS
                                                residuals_R = y_right_child - y_pred_R
                                                rss_R_out = np.sum(residuals_R**2)
                                                
                                                # 4. Extract model parameters
                                                coeffs_R_out = lasso_R_model.coef_
                                                intercept_R_out = lasso_R_model.intercept_
                                
                                        # Assign results back
                                        rss_R = rss_R_out
                                        coeffs_R = coeffs_R_out
                                        intercept_R = intercept_R_out
                                        
                                    except Exception as e:
                                        print("An error occurred:", e)
                                        pass
                                
                                # --- After both fits, calculate total loss and update the best model ---
                                # Check if both fits were successful before proceeding
                                if not np.isnan(rss_L) and not np.isnan(rss_R):
                                    
                                    # Calculate total RSS for the split
                                    total_rss_lasso_split = rss_L + rss_R
                                    total_rss_lasso_split = np.maximum(10**-8, total_rss_lasso_split)
                                
                                    # Get the number of selected coefficients for the penalty term
                                    num_coeffs_L = np.sum(coeffs_L != 0)
                                    num_coeffs_R = np.sum(coeffs_R != 0)

                                    # Only check the loss if there is a feature selected for one of the children models
                                    if (num_coeffs_L + num_coeffs_R) > 0:
                                        # Calculate the information criterion (loss) for this split
                                        loss_lasso_split = loss_fun(
                                            criteria=split_criterion,
                                            num=num.sum(),
                                            Rss=np.array([total_rss_lasso_split]),
                                            k=k_ls, # Base DF for a lasso split
                                            coef_num = num_coeffs_L + num_coeffs_R
                                        ).item()
                                        
                                        # Compare with the best model found so far
                                        if loss_lasso_split < best_loss:
                                            best_loss = loss_lasso_split
                                            best_node = "lasso_split"
                                            # best_feature and best_pivot are already set from the finalist split
                                            
                                            # Store the parameters for BOTH children
                                            mask_L = coeffs_L != 0
                                            best_multi_indices_L = numerical_feature_indices_0based[mask_L]
                                            best_multi_coeffs_L = coeffs_L[mask_L]
                                            best_multi_intercept_L = intercept_L
                                            
                                            mask_R = coeffs_R != 0
                                            best_multi_indices_R = numerical_feature_indices_0based[mask_R]
                                            best_multi_coeffs_R = coeffs_R[mask_R]
                                            best_multi_intercept_R = intercept_R
                                            
                                            best_pivot = pivot_opt
                                            best_feature = feature_id
                                            best_node_prev = ""       

            continue

        # CATEGORICAL VARIABLES
        mean_vec = np.zeros(lenp)
        num_vec = np.zeros(lenp)
        for i in range(lenp):
            # Mean values of the response w.r.t. each level
            mean_vec[i] = np.mean(y_sorted[X_sorted[:, feature_id] == possible_p[i]])
            # Number of elements at each level
            num_vec[i] = y_sorted[X_sorted[:, feature_id] == possible_p[i]].shape[0]

        # Sort unique values w.r.t. the mean of the responses
        mean_idx = np.argsort(mean_vec)
        num_vec = num_vec[mean_idx]
        sum_vec = mean_vec[mean_idx] * num_vec
        possible_p = possible_p[mean_idx]

        # Loop over the sorted possible_p and find the best partition
        num = np.array([0.0, X_sorted.shape[0]])
        sum_all = np.array([0, np.sum(y_sorted)])
        for i in range(lenp - 1):
            # Update the sum and num
            sum_all += np.array([1.0, -1.0]) * sum_vec[i]
            num += np.array([1.0, -1.0]) * num_vec[i]
            # Find the indices of the elements in the left node
            sub_index = isin(X_sorted[:, feature_id], possible_p[: i + 1])
            # Compute the rss
            rss = np.sum((y_sorted[sub_index] - sum_all[0] / num[0]) ** 2) + np.sum(
                (y_sorted[~sub_index] - sum_all[1] / num[1]) ** 2
            )
            rss = np.maximum(10**-8, rss)
            loss = loss_fun(
                criteria=split_criterion,
                num=num.sum(),
                Rss=np.array([rss]),
                k=k_pconc,
                coef_num=0,
            )
            if best_node == "" or loss.item() < best_loss:
                best_feature = feature_id
                best_node = "pconc"
                best_loss = loss.item()
                lm_L = np.array([0, sum_all[0] / num[0]])
                lm_R = np.array([0, sum_all[1] / num[1]])
                pivot_c = possible_p[: i + 1].copy()
                pivot_c = pivot_c.astype(np.int64)

            # Determine best categorical split for finalist_d strategy
            if finalist_d:
                if best_node_prev == "" or loss.item() < best_loss_split:
                    best_feature_split = feature_id
                    best_node_prev = "pconc"
                    best_loss_split = loss.item()
                    pivot_c_split = possible_p[: i + 1].copy()
                    pivot_c_split = pivot_c_split.astype(np.int64)

            # Determine best categorical split for per_feature and full_multi stategies
            if (per_feature or full_multi) and (feature_id - 1) in top_k_feature_indices_0based:
                if best_node_feat == "" or loss.item() < best_loss_feat:
                    best_loss_feat = loss.item()
                    best_node_feat = "pconc"
                    pivot_c_feat = possible_p[: i + 1].copy()
                    pivot_c_feat = pivot_c_feat.astype(np.int64)
                    
        if (per_feature or full_multi) and (feature_id - 1) in top_k_feature_indices_0based:
            if best_node_feat != "":
                left_indices = np.array([], dtype=np.int64)
                right_indices = np.array([], dtype=np.int64)
            
                # Get the feature values for all data points in the current node
                feature_values_in_node = X[index, feature_id]
                
                # Create a boolean mask for the left child
                left_mask = isin(feature_values_in_node, pivot_c_feat)
                
                # Apply the mask to the node's `index` to get the child indices
                left_indices = index[left_mask]
                right_indices = index[~left_mask]

                # Check if indices are correct (>0)
                if left_indices.size > 0 and right_indices.size > 0:
                    X_left_child = X[np.ix_(left_indices, numerical_cols_in_X_1based)]
                    y_left_child = y[left_indices].ravel()
            
                    X_right_child = X[np.ix_(right_indices, numerical_cols_in_X_1based)]
                    y_right_child = y[right_indices].ravel()
            
                    # Check if children have enough samples to fit Lasso
                    if X_left_child.shape[0] >= min_sample_leaf and X_right_child.shape[0] >= min_sample_leaf:
                        # --- Initialize variables for this split attempt ---
                        # Using np.nan helps catch errors if a fit fails
                        rss_L, rss_R = np.nan, np.nan
                        coeffs_L, coeffs_R = np.array([-1.0]), np.array([-1.0])
                        intercept_L, intercept_R = np.nan, np.nan
                        
                        # --- Fit Lasso on LEFT child ---
                        try:
                        
                            if multi_lars:
                                lasso_kwargs_L = {'criterion': 'bic', 'max_iter': 500}
                                if X_left_child.shape[0] <= X_left_child.shape[1] + 1:
                                    noise_var_L = np.var(y_left_child)
                                    lasso_kwargs_L['noise_variance'] = noise_var_L + 1e-9
                                
                            with objmode(
                                rss_L_out='float64', 
                                coeffs_L_out='float64[:]', 
                                intercept_L_out='float64'
                            ):
                                # This entire block runs in standard Python mode
                                
                                # 1. Create and fit the model
                                if multi_lars:
                                    lasso_L_model = LassoLarsIC(**lasso_kwargs_L)
                                    lasso_L_model.fit(X_left_child, y_left_child)
                                else:
                                    lasso_L_model = Lasso(alpha=alpha, max_iter=20000, copy_X=True, precompute=False, 
                                                random_state=42, tol=1e-3, selection='cyclic')
                                    lasso_L_model.fit(X_left_child, y_left_child)
                                
                                # 2. Predict on the training data to calculate RSS
                                if lasso_L_model is not None:
                                    y_pred_L = lasso_L_model.predict(X_left_child)
                                    
                                    # 3. Calculate RSS
                                    residuals_L = y_left_child - y_pred_L
                                    rss_L_out = np.sum(residuals_L**2)
                                    
                                    # 4. Extract model parameters to pass back to Numba
                                    coeffs_L_out = lasso_L_model.coef_
                                    intercept_L_out = lasso_L_model.intercept_
                        
                            # Assign the results from objmode back to our Numba-compatible variables
                            rss_L = rss_L_out
                            coeffs_L = coeffs_L_out
                            intercept_L = intercept_L_out
                        
                        except Exception as e:
                            print("An error occurred:", e)
                            # If the left fit fails for any reason, we can't proceed with this split.
                            # We'll break out of this block; the `np.isnan(rss_L)` check later will handle it.
                            pass
                        
                        # --- Fit Lasso on RIGHT child ---
                        # Only proceed if the left fit was successful
                        if not np.isnan(rss_L):
                            try:
                                if multi_lars:
                                    lasso_kwargs_R = {'criterion': 'bic', 'max_iter': 500}
                                    if X_right_child.shape[0] <= X_right_child.shape[1] + 1:
                                        noise_var_R = np.var(y_right_child)
                                        lasso_kwargs_R['noise_variance'] = noise_var_R + 1e-9
                                    
                                with objmode(
                                    rss_R_out='float64', 
                                    coeffs_R_out='float64[:]', 
                                    intercept_R_out='float64'
                                ):
                                    # 1. Create and fit the model
                                    if multi_lars:
                                        lasso_R_model = LassoLarsIC(**lasso_kwargs_R)
                                        lasso_R_model.fit(X_right_child, y_right_child)
                                    else:
                                        lasso_R_model = Lasso(alpha=alpha, max_iter=20000, copy_X=True, precompute=False, 
                                                random_state=42, tol=1e-3, selection='cyclic')
                                        lasso_R_model.fit(X_right_child, y_right_child)
                                    
                                    # 2. Predict to calculate RSS
                                    if lasso_R_model is not None:
                                        y_pred_R = lasso_R_model.predict(X_right_child)
                                        
                                        # 3. Calculate RSS
                                        residuals_R = y_right_child - y_pred_R
                                        rss_R_out = np.sum(residuals_R**2)
                                        
                                        # 4. Extract model parameters
                                        coeffs_R_out = lasso_R_model.coef_
                                        intercept_R_out = lasso_R_model.intercept_
                        
                                # Assign results back
                                rss_R = rss_R_out
                                coeffs_R = coeffs_R_out
                                intercept_R = intercept_R_out
                                
                            except Exception as e:
                                print("An error occurred:", e)
                                pass
                        
                        # --- After both fits, calculate total loss and update the best model ---
                        # Check if both fits were successful before proceeding
                        if not np.isnan(rss_L) and not np.isnan(rss_R):
                            
                            # Calculate total RSS for the split
                            total_rss_lasso_split = rss_L + rss_R
                            total_rss_lasso_split = np.maximum(10**-8, total_rss_lasso_split)
                        
                            # Get the number of selected coefficients for the penalty term
                            num_coeffs_L = np.sum(coeffs_L != 0)
                            num_coeffs_R = np.sum(coeffs_R != 0)

                            # Only check the loss if there is a feature selected for one of the children models
                            if (num_coeffs_L + num_coeffs_R) > 0:
                                # Calculate the information criterion (loss) for this split
                                loss_lasso_split = loss_fun(
                                    criteria=split_criterion,
                                    num=num.sum(), # num.sum() is correct for a split model
                                    Rss=np.array([total_rss_lasso_split]),
                                    k=k_ls, # Base DF for a lasso split
                                    coef_num = num_coeffs_L + num_coeffs_R
                                ).item()
                                
                                # Compare with the best model found so far
                                if loss_lasso_split < best_loss:
                                    best_loss = loss_lasso_split
                                    best_node = "lasso_split"
                                    # best_feature and best_pivot are already set from the finalist split
                                    
                                    # Store the parameters for BOTH children
                                    mask_L = coeffs_L != 0
                                    best_multi_indices_L = numerical_feature_indices_0based[mask_L]
                                    best_multi_coeffs_L = coeffs_L[mask_L]
                                    best_multi_intercept_L = intercept_L
                                    
                                    mask_R = coeffs_R != 0
                                    best_multi_indices_R = numerical_feature_indices_0based[mask_R]
                                    best_multi_coeffs_R = coeffs_R[mask_R]
                                    best_multi_intercept_R = intercept_R
                                    
                                    best_feature = feature_id
                                    best_node_prev = "pconc"
                                    pivot_c = pivot_c_feat             
                
    # Finalist_s stategy
    if finalist_s:            
        best_node_prev = best_node           
        if best_node == "con" or best_node =="lin":
            if can_attempt_lasso and ("lasso_no_split" in regression_nodes):
    
                # --- Fit the LassoLarsIC model ---
                lasso_model = None # Initialize
                rss_lasso = -1.0
                num_lasso_coeffs = 0

                try: # Wrap in try/except to handle potential convergence errors in Lasso
                    coeffs_out, intercept_out = None, None

                    # If n < (p+1) then estimate variance for LARS
                    if multi_lars:
                        lasso_kwargs = {'criterion': 'bic', 'max_iter': 500}
                        if X_node_numerical.shape[0] <= X_node_numerical.shape[1] + 1:
                            noise_var_estimate = np.var(y_node_target)
                            lasso_kwargs['noise_variance'] = noise_var_estimate + 1e-9
                    
                    with objmode(
                        # Define outputs for objmode context
                        intercept_out='float64', 
                        coeffs_out='float64[:]',
                        y_pred_out='float64[:]'
                    ): 

                        # Fit the Lasso/LARS model
                        if multi_lars:
                            # We need to run the fit and predict in objmode
                            temp_lasso = LassoLarsIC(**lasso_kwargs)
                            temp_lasso.fit(X_node_numerical, y_node_target)
                        else:
                            temp_lasso = Lasso(alpha=alpha, max_iter=20000, copy_X=True, precompute=False, 
                                            random_state=42, tol=1e-3, selection='cyclic')
                            temp_lasso.fit(X_node_numerical, y_node_target)
                        
                        if temp_lasso is not None:
                            # Extract results to pass back to Numba-compatible variables
                            intercept_out = temp_lasso.intercept_
                            coeffs_out = temp_lasso.coef_
                            y_pred_out = temp_lasso.predict(X_node_numerical)
            
                    # --- Calculate RSS from predictions ---
                    residuals_lasso = y_node_target - y_pred_out
                    rss_lasso = np.sum(residuals_lasso**2)
                    rss_lasso = np.maximum(10**-8, rss_lasso) # Prevent log(0)
            
                    # --- Calculate Degrees of Freedom for Lasso ---
                    # The penalty is based on the number of selected features.
                    num_lasso_coeffs = np.sum(coeffs_out != 0)

                    # Only check loss if number of features that are selected is larger than 0
                    if num_lasso_coeffs > 0:
                        # --- Calculate the Loss (BIC/AIC) for the Lasso model ---
                        num_samples_in_node = X_node_numerical.shape[0]
                        loss_lasso = loss_fun(
                            criteria=split_criterion,
                            num=num_samples_in_node,
                            Rss=np.array([rss_lasso]),
                            k=k_lns, # Base degrees of freedom for this model type
                            coef_num=num_lasso_coeffs # Additional penalty for selected coeffs
                        ).item() # .item() to get a single float value
                
                        # --- Compare with the best model found so far ---
                        if loss_lasso < best_loss:
                            best_loss = loss_lasso
                            best_node = "lasso_no_split"
                            best_feature = -1 # No splitting feature
                            best_pivot = -1.0
                        
                            selected_mask = coeffs_out != 0
                            best_multi_indices_L = numerical_feature_indices_0based[selected_mask]
                            best_multi_coeffs_L = coeffs_out[selected_mask]
                            best_multi_intercept_L = intercept_out
                            
                except Exception as e:
                    print("An error occurred:", e)
                    # If Lasso fails to fit, we just skip it
                    pass
        elif best_node in ["pcon", "plin", "blin", "pconc"]:
            if can_attempt_lasso and ("lasso_split" in regression_nodes):
            
                left_indices = np.array([], dtype=np.int64)
                right_indices = np.array([], dtype=np.int64)

                # CASE 1: The best univariate split was CATEGORICAL
                if best_node == "pconc":
                    # Partition data based on the levels in `pivot_c`
                    # Get the feature values for all data points in the current node
                    feature_values_in_node = X[index, best_feature]
                    
                    # Create a boolean mask for the left child
                    left_mask = isin(feature_values_in_node, pivot_c)
                    
                    # Apply the mask to the node's `index` to get the child indices
                    left_indices = index[left_mask]
                    right_indices = index[~left_mask]
            
                # CASE 2: The best univariate split was NUMERICAL
                elif best_node in ["pcon", "plin", "blin"]:
                    # Use the logic you already perfected for numerical pivots.
                    # Get the feature values for all data points in the current node
                    feature_values_in_node = X[index, best_feature]
                    
                    # Create a boolean mask and apply it
                    left_mask = feature_values_in_node <= best_pivot
                    left_indices = index[left_mask]
                    right_indices = index[~left_mask]

                #  Check if indices are correct (>0)
                if left_indices.size > 0 and right_indices.size > 0:
                    X_left_child = X[np.ix_(left_indices, numerical_cols_in_X_1based)]
                    y_left_child = y[left_indices].ravel()
            
                    X_right_child = X[np.ix_(right_indices, numerical_cols_in_X_1based)]
                    y_right_child = y[right_indices].ravel()
            
                    # Check if children have enough samples to fit Lasso
                    if X_left_child.shape[0] >= min_sample_leaf and X_right_child.shape[0] >= min_sample_leaf:
                        # --- Initialize variables for this split attempt ---
                        # Using np.nan helps catch errors if a fit fails
                        rss_L, rss_R = np.nan, np.nan
                        coeffs_L, coeffs_R = np.array([-1.0]), np.array([-1.0])
                        intercept_L, intercept_R = np.nan, np.nan
                        
                        # --- Fit Lasso on LEFT child ---
                        try:
                        
                            if multi_lars:
                                lasso_kwargs_L = {'criterion': 'bic', 'max_iter': 500}
                                if X_left_child.shape[0] <= X_left_child.shape[1] + 1:
                                    noise_var_L = np.var(y_left_child)
                                    lasso_kwargs_L['noise_variance'] = noise_var_L + 1e-9
                                
                            with objmode(
                                rss_L_out='float64', 
                                coeffs_L_out='float64[:]', 
                                intercept_L_out='float64'
                            ):
                                # This entire block runs in standard Python mode
                                
                                # 1. Create and fit the model
                                if multi_lars:
                                    lasso_L_model = LassoLarsIC(**lasso_kwargs_L)
                                    lasso_L_model.fit(X_left_child, y_left_child)
                                else:
                                    lasso_L_model = Lasso(alpha=alpha, max_iter=20000, copy_X=True, precompute=False, 
                                                random_state=42, tol=1e-3, selection='cyclic')
                                    lasso_L_model.fit(X_left_child, y_left_child)
                                
                                # 2. Predict on the training data to calculate RSS
                                if lasso_L_model is not None:
                                    y_pred_L = lasso_L_model.predict(X_left_child)
                                    
                                    # 3. Calculate RSS
                                    residuals_L = y_left_child - y_pred_L
                                    rss_L_out = np.sum(residuals_L**2)
                                    
                                    # 4. Extract model parameters to pass back to Numba
                                    coeffs_L_out = lasso_L_model.coef_
                                    intercept_L_out = lasso_L_model.intercept_
                        
                            # Assign the results from objmode back to our Numba-compatible variables
                            rss_L = rss_L_out
                            coeffs_L = coeffs_L_out
                            intercept_L = intercept_L_out
                        
                        except Exception as e:
                            print("An error occurred:", e)
                            # If the left fit fails for any reason, we can't proceed with this split.
                            # We'll break out of this block; the `np.isnan(rss_L)` check later will handle it.
                            pass
                        
                        # --- Fit Lasso on RIGHT child ---
                        # Only proceed if the left fit was successful
                        if not np.isnan(rss_L):
                            try:
                                if multi_lars:
                                    lasso_kwargs_R = {'criterion': 'bic', 'max_iter': 500}
                                    if X_right_child.shape[0] <= X_right_child.shape[1] + 1:
                                        noise_var_R = np.var(y_right_child)
                                        lasso_kwargs_R['noise_variance'] = noise_var_R + 1e-9
                                    
                                with objmode(
                                    rss_R_out='float64', 
                                    coeffs_R_out='float64[:]', 
                                    intercept_R_out='float64'
                                ):
                                    # 1. Create and fit the model
                                    if multi_lars:
                                        lasso_R_model = LassoLarsIC(**lasso_kwargs_R)
                                        lasso_R_model.fit(X_right_child, y_right_child)
                                    else:
                                        lasso_R_model = Lasso(alpha=alpha, max_iter=20000, copy_X=True, precompute=False, 
                                                random_state=42, tol=1e-3, selection='cyclic')
                                        lasso_R_model.fit(X_right_child, y_right_child)
                                    
                                    # 2. Predict to calculate RSS
                                    if lasso_R_model is not None:
                                        y_pred_R = lasso_R_model.predict(X_right_child)
                                        
                                        # 3. Calculate RSS
                                        residuals_R = y_right_child - y_pred_R
                                        rss_R_out = np.sum(residuals_R**2)
                                        
                                        # 4. Extract model parameters
                                        coeffs_R_out = lasso_R_model.coef_
                                        intercept_R_out = lasso_R_model.intercept_
                        
                                # Assign results back
                                rss_R = rss_R_out
                                coeffs_R = coeffs_R_out
                                intercept_R = intercept_R_out
                                
                            except Exception as e:
                                print("An error occurred:", e)
                                pass
                        
                        # --- After both fits, calculate total loss and update the best model ---
                        # Check if both fits were successful before proceeding
                        if not np.isnan(rss_L) and not np.isnan(rss_R):
                            
                            # Calculate total RSS for the split
                            total_rss_lasso_split = rss_L + rss_R
                            total_rss_lasso_split = np.maximum(10**-8, total_rss_lasso_split)
                        
                            # Get the number of selected coefficients for the penalty term
                            num_coeffs_L = np.sum(coeffs_L != 0)
                            num_coeffs_R = np.sum(coeffs_R != 0)

                            # Only check the loss if there is a feature selected for one of the children models
                            if (num_coeffs_L + num_coeffs_R) > 0:
                                # Calculate the information criterion (loss) for this split
                                loss_lasso_split = loss_fun(
                                    criteria=split_criterion,
                                    num=num.sum(), # num.sum() is correct for a split model
                                    Rss=np.array([total_rss_lasso_split]),
                                    k=k_ls, # Base DF for a lasso split
                                    coef_num = num_coeffs_L + num_coeffs_R
                                ).item()
                                
                                # Compare with the best model found so far
                                if loss_lasso_split < best_loss:
                                    best_loss = loss_lasso_split
                                    best_node = "lasso_split"
                                    # best_feature and best_pivot are already set from the finalist split
                                    
                                    # Store the parameters for BOTH children
                                    mask_L = coeffs_L != 0
                                    best_multi_indices_L = numerical_feature_indices_0based[mask_L]
                                    best_multi_coeffs_L = coeffs_L[mask_L]
                                    best_multi_intercept_L = intercept_L
                                    
                                    mask_R = coeffs_R != 0
                                    best_multi_indices_R = numerical_feature_indices_0based[mask_R]
                                    best_multi_coeffs_R = coeffs_R[mask_R]
                                    best_multi_intercept_R = intercept_R

    # Finalist_d split model
    elif finalist_d:
        if can_attempt_lasso and ("lasso_split" in regression_nodes):
            if best_node_prev != "":
            
                left_indices = np.array([], dtype=np.int64)
                right_indices = np.array([], dtype=np.int64)
                
                    
                # CASE 1: The best univariate split was CATEGORICAL
                if best_node_prev == "pconc":
                    # Partition data based on the levels in `pivot_c`
                    # `isin` is perfect for this.
                    # Note: We operate on the full `index` of the current node.
                    
                    # Get the feature values for all data points in the current node
                    feature_values_in_node = X[index, best_feature_split]
                    
                    # Create a boolean mask for the left child
                    left_mask = isin(feature_values_in_node, pivot_c_split)
                    
                    # Apply the mask to the node's `index` to get the child indices
                    left_indices = index[left_mask]
                    right_indices = index[~left_mask]
            
                # CASE 2: The best univariate split was NUMERICAL
                elif best_node_prev in ["pcon", "plin", "blin"]:
                    # Use the logic you already perfected for numerical pivots.
                    # No need to re-sort, can operate directly on the node's data.
                    
                    # Get the feature values for all data points in the current node
                    feature_values_in_node = X[index, best_feature_split]
                    
                    # Create a boolean mask and apply it
                    left_mask = feature_values_in_node <= best_pivot_split
                    left_indices = index[left_mask]
                    right_indices = index[~left_mask]
                    
                # --- Now, `left_indices` and `right_indices` are correctly populated ---
                # The rest of the logic can proceed identically for both cases.
          
                if left_indices.size > 0 and right_indices.size > 0:
                    X_left_child = X[np.ix_(left_indices, numerical_cols_in_X_1based)]
                    y_left_child = y[left_indices].ravel()
            
                    X_right_child = X[np.ix_(right_indices, numerical_cols_in_X_1based)]
                    y_right_child = y[right_indices].ravel()
            
                    # Check if children have enough samples to fit Lasso
                    if X_left_child.shape[0] >= min_sample_leaf and X_right_child.shape[0] >= min_sample_leaf:
                        # --- Initialize variables for this split attempt ---
                        # Using np.nan helps catch errors if a fit fails
                        rss_L, rss_R = np.nan, np.nan
                        coeffs_L, coeffs_R = np.array([-1.0]), np.array([-1.0])
                        intercept_L, intercept_R = np.nan, np.nan
                        
                        # --- Fit Lasso on LEFT child ---
                        try:
                        
                            if multi_lars:
                                lasso_kwargs_L = {'criterion': 'bic', 'max_iter': 500}
                                if X_left_child.shape[0] <= X_left_child.shape[1] + 1:
                                    noise_var_L = np.var(y_left_child)
                                    lasso_kwargs_L['noise_variance'] = noise_var_L + 1e-9
                                
                            with objmode(
                                rss_L_out='float64', 
                                coeffs_L_out='float64[:]', 
                                intercept_L_out='float64'
                            ):
                                # This entire block runs in standard Python mode
                                
                                # 1. Create and fit the model
                                if multi_lars:
                                    lasso_L_model = LassoLarsIC(**lasso_kwargs_L)
                                    lasso_L_model.fit(X_left_child, y_left_child)
                                else:
                                    lasso_L_model = Lasso(alpha=alpha, max_iter=20000, copy_X=True, precompute=False, 
                                                random_state=42, tol=1e-3, selection='cyclic')
                                    lasso_L_model.fit(X_left_child, y_left_child)
                                
                                # 2. Predict on the training data to calculate RSS
                                if lasso_L_model is not None:
                                    y_pred_L = lasso_L_model.predict(X_left_child)
                                    
                                    # 3. Calculate RSS
                                    residuals_L = y_left_child - y_pred_L
                                    rss_L_out = np.sum(residuals_L**2)
                                    
                                    # 4. Extract model parameters to pass back to Numba
                                    coeffs_L_out = lasso_L_model.coef_
                                    intercept_L_out = lasso_L_model.intercept_
                        
                            # Assign the results from objmode back to our Numba-compatible variables
                            rss_L = rss_L_out
                            coeffs_L = coeffs_L_out
                            intercept_L = intercept_L_out
                        
                        except Exception as e:
                            print("An error occurred:", e)
                            # If the left fit fails for any reason, we can't proceed with this split.
                            # We'll break out of this block; the `np.isnan(rss_L)` check later will handle it.
                            pass
                        
                        # --- Fit Lasso on RIGHT child ---
                        # Only proceed if the left fit was successful
                        if not np.isnan(rss_L):
                            try:
                                if multi_lars:
                                    lasso_kwargs_R = {'criterion': 'bic', 'max_iter': 500}
                                    if X_right_child.shape[0] <= X_right_child.shape[1] + 1:
                                        noise_var_R = np.var(y_right_child)
                                        lasso_kwargs_R['noise_variance'] = noise_var_R + 1e-9
                                    
                                with objmode(
                                    rss_R_out='float64', 
                                    coeffs_R_out='float64[:]', 
                                    intercept_R_out='float64'
                                ):
                                    # 1. Create and fit the model
                                    if multi_lars:
                                        lasso_R_model = LassoLarsIC(**lasso_kwargs_R)
                                        lasso_R_model.fit(X_right_child, y_right_child)
                                    else:
                                        lasso_R_model = Lasso(alpha=alpha, max_iter=20000, copy_X=True, precompute=False, 
                                                random_state=42, tol=1e-3, selection='cyclic')
                                        lasso_R_model.fit(X_right_child, y_right_child)
                                    
                                    # 2. Predict to calculate RSS
                                    if lasso_R_model is not None:
                                        y_pred_R = lasso_R_model.predict(X_right_child)
                                        
                                        # 3. Calculate RSS
                                        residuals_R = y_right_child - y_pred_R
                                        rss_R_out = np.sum(residuals_R**2)
                                        
                                        # 4. Extract model parameters
                                        coeffs_R_out = lasso_R_model.coef_
                                        intercept_R_out = lasso_R_model.intercept_
                        
                                # Assign results back
                                rss_R = rss_R_out
                                coeffs_R = coeffs_R_out
                                intercept_R = intercept_R_out
                                
                            except Exception as e:
                                print("An error occurred:", e)
                                pass
                        
                        # --- After both fits, calculate total loss and update the best model ---
                        # Check if both fits were successful before proceeding
                        if not np.isnan(rss_L) and not np.isnan(rss_R):
                            
                            # Calculate total RSS for the split
                            total_rss_lasso_split = rss_L + rss_R
                            total_rss_lasso_split = np.maximum(10**-8, total_rss_lasso_split)
                        
                            # Get the number of selected coefficients for the penalty term
                            num_coeffs_L = np.sum(coeffs_L != 0)
                            num_coeffs_R = np.sum(coeffs_R != 0)

                            # Only check the loss if there is a feature selected for one of the children models
                            if (num_coeffs_L + num_coeffs_R) > 0:
                                # Calculate the information criterion (loss) for this split
                                loss_lasso_split = loss_fun(
                                    criteria=split_criterion,
                                    num=num.sum(), # num.sum() is correct for a split model
                                    Rss=np.array([total_rss_lasso_split]),
                                    k=k_ls, # Base DF for a lasso split
                                    coef_num = num_coeffs_L + num_coeffs_R
                                ).item()
                                
                                # Compare with the best model found so far
                                if loss_lasso_split < best_loss:
                                    best_loss = loss_lasso_split
                                    best_node = "lasso_split"
                                    # best_feature and best_pivot are already set from the finalist split
                                    
                                    # Store the parameters for BOTH children
                                    mask_L = coeffs_L != 0
                                    best_multi_indices_L = numerical_feature_indices_0based[mask_L]
                                    best_multi_coeffs_L = coeffs_L[mask_L]
                                    best_multi_intercept_L = intercept_L
                                    
                                    mask_R = coeffs_R != 0
                                    best_multi_indices_R = numerical_feature_indices_0based[mask_R]
                                    best_multi_coeffs_R = coeffs_R[mask_R]
                                    best_multi_intercept_R = intercept_R
                                    
                                    best_feature=best_feature_split
                                    
                                    if best_node_prev == "pconc":
                                        pivot_c = pivot_c_split
                                        best_pivot = -1
                                    else:
                                        best_pivot = best_pivot_split
                                        pivot_c = np.array([0])
                                
    # Finalist_d, per_feature and full_multi no_split model
    if finalist_d or per_feature or full_multi:
        if can_attempt_lasso and ("lasso_no_split" in regression_nodes):
    
                # --- Fit the LassoLarsIC model ---
                lasso_model = None # Initialize
                rss_lasso = -1.0
                num_lasso_coeffs = 0

                try: # Wrap in try/except to handle potential convergence errors in Lasso
                    coeffs_out, intercept_out = None, None
                    
                    if multi_lars:
                        lasso_kwargs = {'criterion': 'bic', 'max_iter': 500}
                        if X_node_numerical.shape[0] <= X_node_numerical.shape[1] + 1:
                            noise_var_estimate = np.var(y_node_target)
                            lasso_kwargs['noise_variance'] = noise_var_estimate + 1e-9
                    
                    with objmode(
                        # Define outputs for objmode context
                        intercept_out='float64', 
                        coeffs_out='float64[:]',
                        y_pred_out='float64[:]'
                    ): 
                        
                        if multi_lars:
                            # We need to run the fit and predict in objmode
                            temp_lasso = LassoLarsIC(**lasso_kwargs)
                            temp_lasso.fit(X_node_numerical, y_node_target)
                        else:
                            temp_lasso = Lasso(alpha=alpha, max_iter=20000, copy_X=True, precompute=False, 
                                            random_state=42, tol=1e-3, selection='cyclic')
                            temp_lasso.fit(X_node_numerical, y_node_target)
                        
                        if temp_lasso is not None:
                            # Extract results to pass back to Numba-compatible variables
                            intercept_out = temp_lasso.intercept_
                            coeffs_out = temp_lasso.coef_
                            y_pred_out = temp_lasso.predict(X_node_numerical)
            
                    # --- Calculate RSS from predictions ---
                    residuals_lasso = y_node_target - y_pred_out
                    rss_lasso = np.sum(residuals_lasso**2)
                    rss_lasso = np.maximum(10**-8, rss_lasso) # Prevent log(0)
            
                    # --- Calculate Degrees of Freedom for Lasso ---
                    # The penalty is based on the number of selected features.
                    num_lasso_coeffs = np.sum(coeffs_out != 0)
                    #print(f"nodes chosen for lasso_no_split: {num_lasso_coeffs}")

                    # Only check the loss if number of features selected is more than 0
                    if num_lasso_coeffs > 0:
                        # --- Calculate the Loss (BIC/AIC) for the Lasso model ---
                        num_samples_in_node = X_node_numerical.shape[0]
                        loss_lasso = loss_fun(
                            criteria=split_criterion,
                            num=num_samples_in_node,
                            Rss=np.array([rss_lasso]),
                            k=k_lns, # Base degrees of freedom for this model type
                            coef_num=num_lasso_coeffs # Additional penalty for selected coeffs
                        ).item() # .item() to get a single float value
                
                        # --- Compare with the best model found so far ---
                        if loss_lasso < best_loss:
                            best_loss = loss_lasso
                            best_node = "lasso_no_split"
                            best_feature = -1 # No splitting feature
                            best_pivot = -1.0
                        
                            selected_mask = coeffs_out != 0
                            best_multi_indices_L = numerical_feature_indices_0based[selected_mask]
                            best_multi_coeffs_L = coeffs_out[selected_mask]
                            best_multi_intercept_L = intercept_out
                            
                except Exception as e:
                    print("An error occurred:", e)
                    # If Lasso fails to fit, we just skip it
                    pass

    # For safety
    if best_node != "lasso_no_split" and best_node != "lasso_split":
        # A standard univariate model won. Clear all multi-params.
        best_multi_indices_L = np.array([], dtype=np.int64)
        best_multi_coeffs_L = np.array([], dtype=np.float64)
        best_multi_intercept_L = 0.0 # Use 0.0 instead of NaN
        
        best_multi_indices_R = np.array([], dtype=np.int64)
        best_multi_coeffs_R = np.array([], dtype=np.float64)
        best_multi_intercept_R = 0.0 # Use 0.0 instead of NaN
    
    elif best_node == "lasso_no_split":
        # The no-split model won. Clear the right-side multi-params.
        best_multi_indices_R = np.array([], dtype=np.int64)
        best_multi_coeffs_R = np.array([], dtype=np.float64)
        best_multi_intercept_R = 0.0

    return (
        best_feature, 
        best_pivot, 
        best_node, 
        lm_L, 
        lm_R, 
        interval, 
        pivot_c, 
        best_multi_indices_L,
        best_multi_coeffs_L,
        best_multi_intercept_L,
        best_multi_indices_R,
        best_multi_coeffs_R,
        best_multi_intercept_R,
        best_node_prev
    )

class PILOT(BaseEstimator):
    """
    This is an implementation of the PILOT method.
    Attributes:
    -----------
    max_depth: int,
        the max depth allowed to grow in a tree.
    max_model_depth: int,
        same as max_depth but including linear nodes
    split_criterion: str,
        the criterion to split the tree,
        we have 'AIC'/'AICc'/'BIC'/'adjusted R^2', etc.
    regression_nodes: list,
        A list of regression models used.
        They are 'con', 'lin', 'blin', 'pcon', 'plin'.
    min_sample_split: int,
        the minimal number of samples required
        to split an internal node.
    min_sample_leaf: int,
        the minimal number of samples required
        to be at a leaf node.
    step_size: int,
        boosting step size.
    X: ndarray,
        2D float array of the predictors.
    y, y0: ndarray,
        2D float array of the responses.
    sorted_X_indices: ndarray,
        2D int array of sorted indices according to each feature.
    n_feature: int,
        number of features
    categorical: ndarray,
        1D int array indicating categorical predictors.
    model_tree: tree object,
        learned PILOT model tree.
    B1, B2: int
        upper and lower bound for the first truncation,
        learned from y.
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances. This is computed only when `compute_importances`
        is set to True in a `fit` method. The importances are based on the
        total reduction of RSS at each node.
    """

    def __init__(
        self,
        max_depth=12,
        max_model_depth=100,
        split_criterion="BIC",
        min_sample_split=10,
        min_sample_leaf=5,
        step_size=1,
        random_state=42,
        truncation_factor: int = 3,
        rel_tolerance: float = 0,
        df_settings: dict[str, int] | None = None,
        regression_nodes: list[str] | None = None,
        min_unique_values_regression: float = 5,
    ) -> None:
        """
        Here we input model parameters to build a tree,
        not all the parameters for split finding.
        parameters:
        -----------
        max_depth: int,
            the max depth allowed to grow in a tree.
        split_criterion: str,
            the criterion to split the tree,
            we have 'AIC'/'AICc'/'BIC'/'adjusted R^2', etc.
        min_sample_split: int,
            the minimal number of samples required
            to split an internal node.
        min_sample_leaf: int,
            the minimal number of samples required
            to be at a leaf node.
        step_size: int,
            boosting step size.
        random_state: int,
            Not used, added for compatibility with sklearn framework
        truncation_factor: float,
            By default, predictions are truncated at [-3B, 3B] where B = y_max = -y_min for centered data.
            The multiplyer (3 by default) can be adapted.
        rel_tolerance: float,
            Minimum percentage decrease in RSS in order for a linear node to be added (if 0, there is no restriction on the number of linear nodes).
            Used to avoid recursion errors.
        df_settings:
            Mapping from regression node type to the number of degrees of freedom for that node type.
        regression_nodes:
            List of node types to consider for numerical features. If None, all available regression nodes are considered
        """

        # initialize class attributes
        self.max_depth = max_depth
        self.max_model_depth = max_model_depth
        self.split_criterion = split_criterion
        self.regression_nodes = REGRESSION_NODES if regression_nodes is None else regression_nodes
        self.min_sample_split = min_sample_split
        self.min_sample_leaf = min_sample_leaf
        self.step_size = step_size
        self.random_state = random_state
        self.truncation_factor = truncation_factor
        self.rel_tolerance = rel_tolerance
        self.min_unique_values_regression = min_unique_values_regression

        # attributes used for fitting
        self.X = None
        self.y = None
        self.y0 = None
        self.sorted_X_indices = None
        self.ymean = None
        self.n_features = None
        self.max_features_considered = None
        self.categorical = np.array([-1])
        self.model_tree = None
        self.B1 = None
        self.B2 = None
        self.recursion_counter = {"lin": 0, "blin": 0, "pcon": 0, "plin": 0, "pconc": 0, "lasso_no_split": 0, "lasso_split": 0}
        self.tree_depth = 0
        self.model_depth = 0
        self.feature_importances_ = None 

        # Order of preference for regression nodes
        # This cannot be changed as best split relies on this specific order
        self.regression_nodes = [
            node for node in NODE_PREFERENCE_ORDER if node in self.regression_nodes
        ]

        # Degrees of freedom for each regression node
        self.k = DEFAULT_DF_SETTINGS.copy()
        if df_settings is not None:
            self.k.update(df_settings)

        # df need to be stored as separate numpy arrays for numba
        self.k = {k: np.array([v], dtype=np.int64) for k, v in self.k.items()}
        self.k_con = self.k["con"]
        self.k_lin = self.k["lin"]
        self.k_lns = self.k["lasso_no_split"]
        self.k_ls = self.k["lasso_split"]

        k_list = []
        if 'blin' in self.regression_nodes:
            k_list.append(self.k['blin'])
        if 'pcon' in self.regression_nodes:
            k_list.append(self.k['pcon'])
        if 'plin' in self.regression_nodes:
            k_list.append(self.k['plin'])
        
        self.k_split_nodes = np.concatenate(k_list)
        self.k_pconc = self.k["pconc"]

    def stop_criterion(self, tree_depth, model_depth, y):
         """
        Determines if the tree building process should stop at the current node.

        Stopping occurs if the maximum tree depth or model depth is reached, or
        if the number of samples in the node is too small for a split.

        Args:
            tree_depth (int): The current depth of split nodes.
            model_depth (int): The current total depth of all nodes.
            y (np.ndarray): The response data in the current node.

        Returns:
            bool: True if building should continue, False if it should stop.
        """
        if (
            (tree_depth >= self.max_depth)
            or (model_depth >= self.max_model_depth)
            or (y.shape[0] <= self.min_sample_split)
        ):
            return False
        return True

    def build_tree(self, tree_depth, model_depth, indices, rss, importances: Optional[np.ndarray] = None):
        """
        Recursively builds a standard PILOT tree.

        This function calls `best_split` to find the optimal model for the current
        node. If a split is chosen, it updates the residuals and recursively calls
        itself for the left and right child nodes.

        Args:
            tree_depth (int): The current depth of the tree's split nodes.
            model_depth (int): The current total depth of the tree.
            indices (np.ndarray): The indices of data points in the current node.
            rss (float): The residual sum of squares at the parent node.
            importances (np.ndarray, optional): Array to accumulate feature importances.

        Returns:
            tree: The constructed tree or subtree for the current node.
        """
        tree_depth += 1
        model_depth += 1
        # Fit models on the node
        best_feature, best_pivot, best_node, lm_l, lm_r, interval, pivot_c = best_split(
            indices,
            self.regression_nodes,
            self.n_features,
            self.sorted_X_indices,
            self.X,
            self.y,
            self.split_criterion,
            self.min_sample_leaf,
            self.k_con,
            self.k_lin,
            self.k_split_nodes,
            self.k_pconc,
            self.categorical,
            self.max_features_considered,
            self.min_unique_values_regression,
        )  # Find the best split

        if importances is not None:
            rss_con = np.sum((self.y[indices]-self.y[indices].mean())**2)
            
        # Stop fitting the tree
        if best_node == "":
            return tree(node="END", Rt=rss)
        elif best_node in ["con", "lin"]:
            # Do not include 'lin' and 'con' in the depth calculation
            tree_depth -= 1

        self.tree_depth = max(self.tree_depth, tree_depth)
        self.model_depth = max(self.model_depth, model_depth)

        # Build tree only if it doesn't meet the stop_criterion
        if self.stop_criterion(tree_depth, model_depth, self.y[indices]):
            # Define a new node
            # best_feature should - 1 because the 1st column is the indices
            node = tree(
                best_node,
                (best_feature - 1, best_pivot),
                lm_l,
                lm_r,
                Rt=rss,
                depth=tree_depth + 1,
                interval=interval,
                pivot_c=pivot_c,
            )

            # Update X and y by vectorization, reshape them to make sure their sizes are correct
            if best_node == "lin":
                rss_previous = np.sum(self.y[indices] ** 2)
                # Unpdate y
                raw_res = self.y[indices] - self.step_size * (
                    lm_l[0] * self.X[indices, best_feature].reshape(-1, 1) + lm_l[1]
                )
                # Truncate the prediction
                self.y[indices] = self.y0[indices] - np.maximum(
                    np.minimum(self.y0[indices] - raw_res, self.B1), self.B2
                )
                rss_new = np.sum(self.y[indices] ** 2)
                improvement = (rss_previous - rss_new) / rss_previous
                if improvement < self.rel_tolerance:
                    node.left = tree(node="END", Rt=np.sum(self.y[indices] ** 2))
                    return node

                else:
                    if importances is not None:
                        rss_lin = np.sum(raw_res**2)
                        importances[best_feature-1] += rss_con - rss_lin

                    self.recursion_counter[best_node] += 1
                    # Recursion
                    node.left = self.build_tree(
                        tree_depth=tree_depth,
                        model_depth=model_depth,
                        indices=indices,
                        rss=np.maximum(
                            0, np.sum((self.y[indices] - np.mean(self.y[indices])) ** 2)
                        ),
                        importances=importances
                    )

            elif best_node == "con":
                self.y[indices] -= self.step_size * (lm_l[1])

                # Stop the recursion
                node.left = tree(node="END", Rt=np.sum(self.y[indices] ** 2))
                return node

            else:
                # Find the indices for the cases in the left and right node
                if best_node == "pconc":
                    cond = isin(self.X[indices, best_feature], pivot_c)
                else:
                    cond = self.X[indices, best_feature] <= best_pivot
                indices_left = (self.X[indices][cond, 0]).astype(int)
                indices_right = (self.X[indices][~cond, 0]).astype(int)

                # Compute the importances
                if importances is not None and indices_left.size > 0 and indices_right.size > 0:
                    y_left = self.y[indices_left]
                    y_right = self.y[indices_right]

                    # 1. Calculate the raw residuals for the fitted model
                    rawres_left = (y_left - (lm_l[0] * self.X[indices_left, best_feature].reshape(-1, 1) + lm_l[1]))
                    rawres_right = (y_right - (lm_r[0] * self.X[indices_right, best_feature].reshape(-1, 1) + lm_r[1]))

                    # 2. Calculate the total RSS of the fitted model
                    rss_model = np.sum(rawres_left ** 2) + np.sum(rawres_right ** 2)

                    # 3. Calculate total improvement over the parent's constant model
                    total_improvement = rss_con - rss_model

                    # 4. Attribute the ENTIRE improvement to the single splitting/regression feature
                    if total_improvement > 0 and best_feature > 0:
                        importances[best_feature - 1] += total_improvement

                # Compute the raw and truncated predicrtion
                rawres_left = (
                    self.y[indices_left]
                    - (lm_l[0] * self.X[indices_left, best_feature].reshape(-1, 1) + lm_l[1])
                ).copy()
                self.y[indices_left] = self.y0[indices_left] - np.maximum(
                    np.minimum(self.y0[indices_left] - rawres_left, self.B1), self.B2
                )
                rawres_right = (
                    self.y[indices_right]
                    - (lm_r[0] * self.X[indices_right, best_feature].reshape(-1, 1) + lm_r[1])
                ).copy()
                self.y[indices_right] = self.y0[indices_right] - np.maximum(
                    np.minimum(self.y0[indices_right] - rawres_right, self.B1), self.B2
                )

                # Recursion
                try:
                    self.recursion_counter[best_node] += 1
                    node.left = self.build_tree(
                        tree_depth=tree_depth,
                        model_depth=model_depth,
                        indices=indices_left,
                        rss=np.maximum(
                            0,
                            np.sum((self.y[indices_left] - np.mean(self.y[indices_left])) ** 2),
                        ),
                        importances=importances
                    )

                    node.right = self.build_tree(
                        tree_depth=tree_depth,
                        model_depth=model_depth,
                        indices=indices_right,
                        rss=np.maximum(
                            0,
                            np.sum((self.y[indices_right] - np.mean(self.y[indices_right])) ** 2),
                        ),
                        importances=importances
                    )
                except RecursionError:
                    print(
                        f"ERROR: encountered recursion error, return END node. "
                        f"Current counter: {self.recursion_counter}"
                    )
                    return tree(node="END", Rt=rss)

        else:
            # Stop recursion if meeting the stopping criterion
            return tree(node="END", Rt=rss)

        return node

    def _validate_data(self):
        """
        Asserts that the data arrays have the correct dtypes for Numba.
        """
        assert np.issubdtype(
            self.sorted_X_indices.dtype, np.int64
        ), f"sorted_X_indices should be int64, but is {self.sorted_X_indices.dtype}"
        assert np.issubdtype(
            self.X.dtype, np.float64
        ), f"X should be float64, but is {self.X.dtype}"
        assert np.issubdtype(
            self.y.dtype, np.float64
        ), f"y should be float64, but is {self.y.dtype}"
        assert np.issubdtype(
            self.categorical.dtype, np.int64
        ), f"categorical should be int64, but is {self.categorical.dtype}"

    def fit(
        self,
        X,
        y,
        categorical=np.array([-1]),
        max_features_considered: Optional[int] = None,
        compute_importances: bool = False,
        visualize_tree: bool = False,
        feature_names: list[str] = None, 
        **vis_kwargs, 
    ):
        """
        Fits a standard PILOT model to the training data.

        This method prepares the data, including sorting and handling different
        data types, and then initiates the recursive tree-building process by
        calling `build_tree`.

        Args:
            X (array-like): The predictor variables.
            y (array-like): The response variable.
            categorical (np.ndarray, optional): Indices of categorical features.
            max_features_considered (int, optional): The number of features to
                randomly sample at each split. If None, all are used.
            compute_importances (bool): If True, feature importances will be calculated.
            visualize_tree (bool): If True, a visualization of the tree will be generated.
            feature_names (list, optional): Names of the features for visualization.
            **vis_kwargs: Additional keyword arguments for the visualization function.
        """
        # X and y should have the same size
        assert X.shape[0] == y.shape[0]
        
        X_original_for_vis = X.copy()

        # Switch pandas objects to numpy objects
        if isinstance(X, pd.core.frame.DataFrame):
            if feature_names is None: 
                feature_names = list(X.columns)
            X = np.array(X)

        if isinstance(y, pd.core.frame.DataFrame):
            y = np.array(y)
        elif y.ndim == 1:
            y = y.reshape((-1, 1))

        # Define class attributes
        self.n_features = X.shape[1]
        self.max_features_considered = (
            min(max_features_considered, self.n_features)
            if max_features_considered is not None
            else self.n_features
        )
        n_samples = X.shape[0]
        self.categorical = categorical

        # Insert indices to the first column of X to memorize the indices
        self.X = np.c_[np.arange(0, n_samples, dtype=int), X]

        # Memorize the indices of the cases sorted along each feature
        # Do not sort the first column since they are just indices
        sorted_indices = np.array(
            [
                np.argsort(self.X[:, feature_id], axis=0).flatten()
                for feature_id in range(1, self.n_features + 1)
            ]
        )
        self.sorted_X_indices = (self.X[:, 0][sorted_indices]).astype(int)
        # ->(n_samples, n_features) 2D array

        # y should be remembered and modified during 'boosting'
        self.y = y.copy()  # calculate on y directly to save memory
        self.y0 = y.copy()  # for the truncation procudure
        padding = (self.truncation_factor - 1) * ((y.max() - y.min()) / 2)
        self.B1 = y.max() + padding  # compute the upper bound for the first truncation
        self.B2 = y.min() - padding  # compute the lower bound for the second truncation

        # If true, initiliaze arrays for importances
        if compute_importances:
            self.feature_importances_ = np.zeros(self.n_features)
            importances_array = self.feature_importances_
        else:
            self.feature_importances_ = None
            importances_array = None

        self._validate_data()
        # Build the tree, only need to take in the indices for X
        self.model_tree = self.build_tree(
            tree_depth=-1,
            model_depth=-1,
            indices=self.sorted_X_indices[0],
            rss=np.sum((y - y.mean()) ** 2),
            importances=importances_array
        )

        # If the first node is 'con'
        if self.model_tree.node == "END":
            self.ymean = y.mean()

        # If true, visualize the tree
        if visualize_tree:
            print("Model fitting complete. Generating tree visualization...")
            visualize_pilot_tree(
                model_tree=self.model_tree,
                training_data=np.array(X_original_for_vis), # Use the original, 0-based data
                feature_names=feature_names,
                **vis_kwargs # Pass along optional args like figsize and filename
            )

        return
        
    def fit_nlfs(
        self,
        X,
        y,
        categorical=np.array([-1]),
        max_features_considered: Optional[int] = None,
        alpha: float = 0.005,
        nlfs_lars: bool = False,
        only_fallback: bool = False,
        compute_importances: bool = False,
        visualize_tree: bool = False,          
        feature_names: list[str] = None,      
        **vis_kwargs,  
    ):
        """
        Fits a PILOT model with Node-Level Feature Selection (NLFS).

        This method orchestrates the fitting of the NLFS variant. It prepares the
        data and then calls `build_tree_nlfs` to start the recursive building
        process, passing along the NLFS-specific hyperparameters.

        Args:
            (All args from `fit`, plus):
            alpha (float): The regularization parameter for the node-level Lasso.
            nlfs_lars (bool): If True, use LARS instead of Lasso with fixed alpha for selection.
            only_fallback (bool): If True, only use only the fallback mechanism.
        """
        # X and y should have the same size
        assert X.shape[0] == y.shape[0]

        X_original_for_vis = X.copy() 
        
        if isinstance(X, pd.core.frame.DataFrame):
            # If feature_names aren't provided, get them from the DataFrame columns
            if feature_names is None: 
                feature_names = list(X.columns)
            X = np.array(X)


        if isinstance(y, pd.core.frame.DataFrame):
            y = np.array(y)
        elif y.ndim == 1:
            y = y.reshape((-1, 1))

        # Define class attributes
        self.n_features = X.shape[1]
        self.max_features_considered = (
            min(max_features_considered, self.n_features)
            if max_features_considered is not None
            else self.n_features
        )
        n_samples = X.shape[0]
        self.categorical = categorical

        # Insert indices to the first column of X to memorize the indices
        self.X = np.c_[np.arange(0, n_samples, dtype=int), X]

        # Memorize the indices of the cases sorted along each feature
        # Do not sort the first column since they are just indices
        sorted_indices = np.array(
            [
                np.argsort(self.X[:, feature_id], axis=0).flatten()
                for feature_id in range(1, self.n_features + 1)
            ]
        )
        self.sorted_X_indices = (self.X[:, 0][sorted_indices]).astype(int)
        # ->(n_samples, n_features) 2D array

        # y should be remembered and modified during 'boosting'
        self.y = y.copy()  # calculate on y directly to save memory
        self.y0 = y.copy()  # for the truncation procudure
        padding = (self.truncation_factor - 1) * ((y.max() - y.min()) / 2)
        self.B1 = y.max() + padding  # compute the upper bound for the first truncation
        self.B2 = y.min() - padding  # compute the lower bound for the second truncation

        # If true, initialize arrays for importances
        if compute_importances:
            self.feature_importances_ = np.zeros(self.n_features)
            importances_array = self.feature_importances_
        else:
            self.feature_importances_ = None
            importances_array = None

        self._validate_data()
        # Build the tree, only need to take in the indices for X
        self.model_tree = self.build_tree_nlfs(
            tree_depth=-1,
            model_depth=-1,
            indices=self.sorted_X_indices[0],
            rss=np.sum((y - y.mean()) ** 2),
            alpha=alpha,
            nlfs_lars=nlfs_lars,
            should_lars=True,
            only_fallback=only_fallback,
            importances=importances_array
        )

        # if the first node is 'con'
        if self.model_tree.node == "END":
            self.ymean = y.mean()

        # If true, visualize the tree
        if visualize_tree:
            print("Model fitting complete. Generating tree visualization...")
            
            visualize_pilot_tree(
                model_tree=self.model_tree,
                training_data=np.array(X_original_for_vis), # Use the original, 0-based data
                feature_names=feature_names,
                **vis_kwargs # Pass along optional args like figsize and filename
            )

        return
        
    def build_tree_nlfs(self, tree_depth, model_depth, indices, rss, alpha, nlfs_lars, should_lars, only_fallback,
                        importances: Optional[np.ndarray] = None):
         """
        Recursively builds a PILOT tree with Node-Level Feature Selection (NLFS).

        This function calls `best_split_nlfs` to determine the best model,
        which internally performs feature selection before searching for a split.
        It then recurses on the resulting child nodes.

        Args:
            (All args from `build_tree`, plus):
            alpha (float): Regularization parameter for node-level Lasso.
            nlfs_lars (bool): Flag to use LARS instead of Lasso with fixed alpha.
            should_lars (bool): Flag indicating if LARS should be attempted.
            only_fallback (bool): Flag to only use only the fallback strategy.
        """
        tree_depth += 1
        model_depth += 1
        # Fit models on the node
        best_feature, best_pivot, best_node, lm_l, lm_r, interval, pivot_c, should_lars = best_split_nlfs(
            indices,
            self.regression_nodes,
            self.n_features,
            self.sorted_X_indices,
            self.X,
            self.y,
            self.split_criterion,
            self.min_sample_leaf,
            self.k_con,
            self.k_lin,
            self.k_split_nodes,
            self.k_pconc,
            self.categorical,
            self.max_features_considered,
            self.min_unique_values_regression,
            alpha,
            nlfs_lars,
            should_lars,
            only_fallback
        )  # Find the best split
        
        if importances is not None:
            rss_con = np.sum((self.y[indices]-self.y[indices].mean())**2)

        # Stop fitting the tree
        if best_node == "":
            return tree(node="END", Rt=rss)
        elif best_node in ["con", "lin"]:
            # do not include 'lin' and 'con' in the depth calculation
            tree_depth -= 1

        self.tree_depth = max(self.tree_depth, tree_depth)
        self.model_depth = max(self.model_depth, model_depth)

        # Build tree only if it doesn't meet the stop_criterion
        if self.stop_criterion(tree_depth, model_depth, self.y[indices]):
            # Define a new node
            # best_feature should - 1 because the 1st column is the indices
            node = tree(
                best_node,
                (best_feature - 1, best_pivot),
                lm_l,
                lm_r,
                Rt=rss,
                depth=tree_depth + 1,
                interval=interval,
                pivot_c=pivot_c,
            )

            # Update X and y by vectorization, reshape them to make sure their sizes are correct
            if best_node == "lin":
                rss_previous = np.sum(self.y[indices] ** 2)
                # Unpdate y
                raw_res = self.y[indices] - self.step_size * (
                    lm_l[0] * self.X[indices, best_feature].reshape(-1, 1) + lm_l[1]
                )
                # Truncate the prediction
                self.y[indices] = self.y0[indices] - np.maximum(
                    np.minimum(self.y0[indices] - raw_res, self.B1), self.B2
                )
                rss_new = np.sum(self.y[indices] ** 2)
                improvement = (rss_previous - rss_new) / rss_previous
                if improvement < self.rel_tolerance:
                    node.left = tree(node="END", Rt=np.sum(self.y[indices] ** 2))
                    return node

                else:
                    if importances is not None:
                        rss_lin = np.sum(raw_res**2)
                        importances[best_feature-1] += rss_con - rss_lin

                    self.recursion_counter[best_node] += 1
                    # Recursion
                    node.left = self.build_tree_nlfs(
                        tree_depth=tree_depth,
                        model_depth=model_depth,
                        indices=indices,
                        rss=np.maximum(
                            0, np.sum((self.y[indices] - np.mean(self.y[indices])) ** 2)
                        ),
                        alpha=alpha,
                        nlfs_lars=nlfs_lars,
                        should_lars=should_lars,
                        only_fallback=only_fallback,
                        importances=importances
                    )

            elif best_node == "con":
                self.y[indices] -= self.step_size * (lm_l[1])

                # Stop the recursion
                node.left = tree(node="END", Rt=np.sum(self.y[indices] ** 2))
                return node

            else:
                # Find the indices for the cases in the left and right node
                if best_node == "pconc":
                    cond = isin(self.X[indices, best_feature], pivot_c)
                else:
                    cond = self.X[indices, best_feature] <= best_pivot
                indices_left = (self.X[indices][cond, 0]).astype(int)
                indices_right = (self.X[indices][~cond, 0]).astype(int)

                # Compute importances
                if importances is not None and indices_left.size > 0 and indices_right.size > 0:
                    y_left = self.y[indices_left]
                    y_right = self.y[indices_right]

                    # 1. Calculate the raw residuals for the fitted model
                    rawres_left = (y_left - (lm_l[0] * self.X[indices_left, best_feature].reshape(-1, 1) + lm_l[1]))
                    rawres_right = (y_right - (lm_r[0] * self.X[indices_right, best_feature].reshape(-1, 1) + lm_r[1]))

                    # 2. Calculate the total RSS of the fitted model
                    rss_model = np.sum(rawres_left ** 2) + np.sum(rawres_right ** 2)

                    # 3. Calculate total improvement over the parent's constant model
                    total_improvement = rss_con - rss_model

                    # 4. Attribute the ENTIRE improvement to the single splitting/regression feature
                    if total_improvement > 0 and best_feature > 0:
                        importances[best_feature - 1] += total_improvement

                # Compute the raw and truncated predicrtion
                rawres_left = (
                    self.y[indices_left]
                    - (lm_l[0] * self.X[indices_left, best_feature].reshape(-1, 1) + lm_l[1])
                ).copy()
                self.y[indices_left] = self.y0[indices_left] - np.maximum(
                    np.minimum(self.y0[indices_left] - rawres_left, self.B1), self.B2
                )
                rawres_right = (
                    self.y[indices_right]
                    - (lm_r[0] * self.X[indices_right, best_feature].reshape(-1, 1) + lm_r[1])
                ).copy()
                self.y[indices_right] = self.y0[indices_right] - np.maximum(
                    np.minimum(self.y0[indices_right] - rawres_right, self.B1), self.B2
                )

                # Recursion
                try:
                    self.recursion_counter[best_node] += 1
                    node.left = self.build_tree_nlfs(
                        tree_depth=tree_depth,
                        model_depth=model_depth,
                        indices=indices_left,
                        rss=np.maximum(
                            0,
                            np.sum((self.y[indices_left] - np.mean(self.y[indices_left])) ** 2),
                        ),
                        alpha=alpha,
                        nlfs_lars=nlfs_lars,
                        should_lars=should_lars,
                        only_fallback=only_fallback,
                        importances=importances
                    )

                    node.right = self.build_tree_nlfs(
                        tree_depth=tree_depth,
                        model_depth=model_depth,
                        indices=indices_right,
                        rss=np.maximum(
                            0,
                            np.sum((self.y[indices_right] - np.mean(self.y[indices_right])) ** 2),
                        ),
                        alpha=alpha,
                        nlfs_lars=nlfs_lars,
                        should_lars=should_lars,
                        only_fallback=only_fallback,
                        importances=importances
                    )
                except RecursionError:
                    print(
                        f"ERROR: encountered recursion error, return END node. "
                        f"Current counter: {self.recursion_counter}"
                    )
                    return tree(node="END", Rt=rss)

        else:
            # Stop recursion if meeting the stopping criterion
            return tree(node="END", Rt=rss)

        return node
        
    def fit_multi(
        self,
        X,
        y,
        categorical=np.array([-1]),
        max_features_considered: Optional[int] = None,
        alpha: float=0.01,
        multi_lars: bool=False,
        finalist_s: bool=True,
        finalist_d: bool=False,
        per_feature: bool=False,
        full_multi: bool=False,
        compute_importances: bool = False,
        visualize_tree: bool = False, 
        feature_names: list[str] = None,
        **vis_kwargs, 
    ):
         """
        Fits a PILOT model with multivariate nodes (Multi-PILOT).

        This method orchestrates the fitting of the Multi-PILOT variant. It prepares
        the data and then calls `build_tree_multi`, passing along the parameters
        that control the multivariate splitting strategies.

        Args:
            (All args from `fit`, plus):
            alpha (float): Regularization parameter for multivariate models.
            multi_lars (bool): If True, use LARS; otherwise, use Lasso.
            finalist_s, finalist_d, per_feature, full_multi (bool): Flags to
                select the specific multivariate strategy to use.
        """
        # X and y should have the same size
        assert X.shape[0] == y.shape[0]

        X_original_for_vis = X.copy()
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = list(X.columns)
            X = np.array(X)

        if isinstance(y, pd.core.frame.DataFrame):
            y = np.array(y)
        elif y.ndim == 1:
            y = y.reshape((-1, 1))

        # Define class attributes
        self.n_features = X.shape[1]
        self.max_features_considered = (
            min(max_features_considered, self.n_features)
            if max_features_considered is not None
            else self.n_features
        )
        n_samples = X.shape[0]
        self.categorical = categorical

        # Insert indices to the first column of X to memorize the indices
        self.X = np.c_[np.arange(0, n_samples, dtype=int), X]

        # Memorize the indices of the cases sorted along each feature
        # Do not sort the first column since they are just indices
        sorted_indices = np.array(
            [
                np.argsort(self.X[:, feature_id], axis=0).flatten()
                for feature_id in range(1, self.n_features + 1)
            ]
        )
        self.sorted_X_indices = (self.X[:, 0][sorted_indices]).astype(int)
        # ->(n_samples, n_features) 2D array

        # y should be remembered and modified during 'boosting'
        self.y = y.copy()  # calculate on y directly to save memory
        self.y0 = y.copy()  # for the truncation procudure
        padding = (self.truncation_factor - 1) * ((y.max() - y.min()) / 2)
        self.B1 = y.max() + padding  # compute the upper bound for the first truncation
        self.B2 = y.min() - padding  # compute the lower bound for the second truncation

        # If true, initialize arrays for importances
        if compute_importances:
            self.feature_importances_ = np.zeros(self.n_features)
            importances_array = self.feature_importances_
        else:
            self.feature_importances_ = None 
            importances_array = None

        self._validate_data()
        # Build the tree, only need to take in the indices for X
        self.model_tree = self.build_tree_multi(
            tree_depth=-1,
            model_depth=-1,
            indices=self.sorted_X_indices[0].ravel(),
            rss=np.sum((y - y.mean()) ** 2),
            alpha=alpha,
            multi_lars=multi_lars,
            finalist_s=finalist_s, 
            finalist_d=finalist_d, 
            per_feature=per_feature, 
            full_multi=full_multi,
            importances=importances_array
        )

        # If the first node is 'con'
        if self.model_tree.node == "END":
            self.ymean = y.mean()

        # If true, visualize tree
        if visualize_tree:
            print("Model fitting complete. Generating tree visualization...")
    
            visualize_pilot_tree(
                model_tree=self.model_tree,
                training_data=np.array(X_original_for_vis),
                feature_names=feature_names,
                **vis_kwargs
            )
        
        return
        
    def build_tree_multi(self, tree_depth, model_depth, indices, rss, alpha, multi_lars, finalist_s, finalist_d, per_feature, full_multi,  importances: Optional[np.ndarray] = None):
         """
        Recursively builds a PILOT tree with support for multivariate nodes.

        This function calls `best_split_multi`, which can return either a standard
        univariate model or a more complex multivariate one. It correctly handles
        the parameters of both model types and recurses on the child nodes.

        Args:
            (All args from `build_tree`, plus):
            alpha (float): Regularization parameter for multivariate models.
            multi_lars (bool): Flag to use LARS instead of Lasso.
            finalist_s, finalist_d, per_feature, full_multi (bool): Flags to
                select the multivariate strategy.
        """
        tree_depth += 1
        model_depth += 1
        # Fit models on the node
        (best_feature, best_pivot, 
        best_node, lm_l, lm_r, 
        interval, pivot_c, 
        best_multi_indices_L,
        best_multi_coeffs_L,
        best_multi_intercept_L,
        best_multi_indices_R,
        best_multi_coeffs_R,
        best_multi_intercept_R,
        best_node_prev) = best_split_multi(
            indices,
            self.regression_nodes,
            self.n_features,
            self.sorted_X_indices,
            self.X,
            self.y,
            self.split_criterion,
            self.min_sample_leaf,
            self.k_con,
            self.k_lin,
            self.k_split_nodes,
            self.k_pconc,
            self.k_lns,
            self.k_ls,
            self.categorical,
            self.max_features_considered,
            self.min_unique_values_regression,
            alpha,
            multi_lars,
            finalist_s,
            finalist_d,
            per_feature,
            full_multi
        )  # Find the best split
        
        if importances is not None:
            rss_con = np.sum((self.y[indices]-self.y[indices].mean())**2)

        # Stop fitting the tree
        if best_node == "":
            return tree_multi(node="END", Rt=rss)
        elif best_node in ["con", "lin", "lasso_no_split"]:
            # Do not include 'lin' and 'con' in the depth calculation
            tree_depth -= 1

        self.tree_depth = max(self.tree_depth, tree_depth)
        self.model_depth = max(self.model_depth, model_depth)

        # Build tree only if it doesn't meet the stop_criterion
        if self.stop_criterion(tree_depth, model_depth, self.y[indices]):
            # Define a new node
            # best_feature should - 1 because the 1st column is the indices
            node = tree_multi(
                node=best_node,
                pivot=(best_feature - 1, best_pivot),
                lm_l=lm_l,
                lm_r=lm_r,
                Rt=rss,
                depth=tree_depth + 1,
                interval=interval,
                pivot_c=pivot_c,
                # --- PASS THE NEW MULTIVARIATE PARAMETERS ---
                multi_model_indices_L=best_multi_indices_L,
                multi_model_coeffs_L=best_multi_coeffs_L,
                multi_model_intercept_L=best_multi_intercept_L,
                multi_model_indices_R=best_multi_indices_R,
                multi_model_coeffs_R=best_multi_coeffs_R,
                multi_model_intercept_R=best_multi_intercept_R,
                best_node_prev=best_node_prev
            )
            

            # Update X and y by vectorization, reshape them to make sure their sizes are correct
            # Case 1: Univariate NO-SPLIT model ("lin")
            if best_node == "lin":

                rss_previous = np.sum(self.y[indices] ** 2)
                raw_res = self.y[indices] - self.step_size * (
                    lm_l[0] * self.X[indices, best_feature].reshape(-1, 1) + lm_l[1]
                )
                self.y[indices] = self.y0[indices] - np.maximum(
                    np.minimum(self.y0[indices] - raw_res, self.B1), self.B2
                )
                rss_new = np.sum(self.y[indices] ** 2)
                improvement = (rss_previous - rss_new) / (rss_previous + 1e-9)
                if improvement < self.rel_tolerance:
                    node.left = tree_multi(node="END", Rt=np.sum(self.y[indices] ** 2))
                    return node
                else:
                    if importances is not None:
                        rss_lin = np.sum(raw_res**2)
                        importances[best_feature-1] += rss_con - rss_lin

                    self.recursion_counter[best_node] += 1
                    # Recurse with the MULTI version of the function
                    node.left = self.build_tree_multi(
                        tree_depth=tree_depth,
                        model_depth=model_depth,
                        indices=indices,
                        rss=np.maximum(0, np.sum((self.y[indices] - np.mean(self.y[indices])) ** 2)),
                        alpha=alpha,
                        multi_lars=multi_lars,
                        finalist_s=finalist_s,
                        finalist_d=finalist_d,
                        per_feature=per_feature,
                        full_multi=full_multi,
                        importances=importances
                    )
            
            # Case 2: Univariate NO-SPLIT terminal model ("con")
            elif best_node == "con":
                self.y[indices] -= self.step_size * (lm_l[1])
                # Use MultiTree for consistency
                node.left = tree_multi(node="END", Rt=np.sum(self.y[indices] ** 2))
                return node
            
            # Case 3: NEW Multivariate NO-SPLIT model ("lasso_no_split")
            elif best_node == "lasso_no_split":
                # This logic is analogous to the "lin" case
                # Get feature values. Remember to use the 0-based indices + 1 for the internal self.X
                X_model_features = self.X[np.ix_(indices, best_multi_indices_L + 1)]
                
                # Calculate prediction using dot product and intercept
                prediction = (X_model_features @ best_multi_coeffs_L) + best_multi_intercept_L
                
                # Update residuals with truncation
                rss_previous = np.sum(self.y[indices] ** 2)
                raw_res = self.y[indices] - self.step_size * prediction.reshape(-1, 1)
                self.y[indices] = self.y0[indices] - np.maximum(
                    np.minimum(self.y0[indices] - raw_res, self.B1), self.B2
                )
                rss_new = np.sum(self.y[indices] ** 2)
                
                # Check for sufficient improvement
                improvement = (rss_previous - rss_new) / (rss_previous + 1e-9)
                if improvement < self.rel_tolerance:
                    node.left = tree_multi(node="END", Rt=np.sum(self.y[indices] ** 2))
                    return node
                else:
                    if importances is not None:
                        rss_lns = np.sum(raw_res ** 2)
                        improvement_lns = rss_con - rss_lns

                        if improvement_lns > 0:
                            # 4. Distribute the improvement based on coefficient magnitudes
                            abs_coeffs = np.abs(best_multi_coeffs_L)
                            sum_abs_coeffs = np.sum(abs_coeffs)

                            # Avoid division by zero if coeffs are somehow zero
                            if sum_abs_coeffs > 0:
                                # Calculate proportions for each feature
                                proportions = abs_coeffs / sum_abs_coeffs

                                # Calculate the share of improvement for each feature
                                improvements_per_feature = improvement_lns * proportions

                                # 5. Add these scores to the global importances array
                                importances[best_multi_indices_L] += improvements_per_feature

                    self.recursion_counter[best_node] += 1
                    # Recurse on the SAME indices with the MULTI version of the function
                    node.left = self.build_tree_multi(
                        tree_depth=tree_depth,
                        model_depth=model_depth,
                        indices=indices,
                        rss=np.maximum(0, np.sum((self.y[indices] - np.mean(self.y[indices])) ** 2)),
                        alpha=alpha,
                        multi_lars=multi_lars,
                        finalist_s=finalist_s,
                        finalist_d=finalist_d,
                        per_feature=per_feature,
                        full_multi=full_multi,
                        importances=importances
                    )
            
            # Case 4: ALL SPLIT models (covers "pcon", "plin", "blin", "pconc", and our new "lasso_split")
            else:
                # First, find the indices for the left and right children
                feature_values_in_node = self.X[indices, best_feature]
                if best_node == "pconc" or (best_node == "lasso_split" and best_node_prev == "pconc"):
                    left_mask = isin(feature_values_in_node, pivot_c)
                else: # All numerical splits
                    left_mask = feature_values_in_node <= best_pivot
                
                indices_left = indices[left_mask]
                indices_right = indices[~left_mask]

                # Compute importances
                if importances is not None and indices_left.size > 0 and indices_right.size > 0:

                    y_left = self.y[indices_left]
                    y_right = self.y[indices_right]

                    if best_node == "lasso_split":
                        # --- Logic for LASSO_SPLIT (Decomposition is REQUIRED) ---

                        # 1. Credit the Splitting Feature
                        rss_con_left = np.sum((y_left - y_left.mean()) ** 2)
                        rss_con_right = np.sum((y_right - y_right.mean()) ** 2)
                        split_improvement = rss_con - (rss_con_left + rss_con_right)

                        if split_improvement > 0 and best_feature > 0:
                            importances[best_feature - 1] += split_improvement

                        # 2. Credit the Left Child's Regression Features
                        X_model_features_L = self.X[np.ix_(indices_left, best_multi_indices_L + 1)]
                        prediction_L = (X_model_features_L @ best_multi_coeffs_L) + best_multi_intercept_L
                        rss_lasso_left = np.sum((y_left - prediction_L.reshape(-1, 1)) ** 2)
                        improvement_L = rss_con_left - rss_lasso_left
                        if improvement_L > 0:
                            abs_coeffs_L = np.abs(best_multi_coeffs_L)
                            if np.sum(abs_coeffs_L) > 0:
                                proportions = abs_coeffs_L / np.sum(abs_coeffs_L)
                                importances[best_multi_indices_L] += improvement_L * proportions

                        # 3. Credit the Right Child's Regression Features
                        X_model_features_R = self.X[np.ix_(indices_right, best_multi_indices_R + 1)]
                        prediction_R = (X_model_features_R @ best_multi_coeffs_R) + best_multi_intercept_R
                        rss_lasso_right = np.sum((y_right - prediction_R.reshape(-1, 1)) ** 2)
                        improvement_R = rss_con_right - rss_lasso_right
                        if improvement_R > 0:
                            abs_coeffs_R = np.abs(best_multi_coeffs_R)
                            if np.sum(abs_coeffs_R) > 0:
                                proportions = abs_coeffs_R / np.sum(abs_coeffs_R)
                                importances[best_multi_indices_R] += improvement_R * proportions

                    else:
                        # --- ALL OTHER UNIVARIATE SPLITS (pcon, plin, pconc, etc.) ---

                        # 1. Calculate the raw residuals for the fitted model
                        rawres_left = (y_left - (lm_l[0] * self.X[indices_left, best_feature].reshape(-1, 1) + lm_l[1]))
                        rawres_right = (y_right - (lm_r[0] * self.X[indices_right, best_feature].reshape(-1, 1) + lm_r[1]))

                        # 2. Calculate the total RSS of the fitted model
                        rss_model = np.sum(rawres_left ** 2) + np.sum(rawres_right ** 2)

                        # 3. Calculate total improvement over the parent's constant model
                        total_improvement = rss_con - rss_model

                        # 4. Attribute the ENTIRE improvement to the single splitting/regression feature
                        if total_improvement > 0 and best_feature > 0:
                            importances[best_feature - 1] += total_improvement
            
                # --- Update residuals for the LEFT child ---
                if best_node == "lasso_split":
                    # Use the multivariate model for prediction
                    X_model_features_L = self.X[np.ix_(indices_left, best_multi_indices_L + 1)]
                    prediction_L = (X_model_features_L @ best_multi_coeffs_L) + best_multi_intercept_L
                    rawres_left = self.y[indices_left] - prediction_L.reshape(-1, 1)

                else:
                    # Use the original univariate logic for pcon, plin, blin, etc.
                    rawres_left = (self.y[indices_left] - (lm_l[0] * self.X[indices_left, best_feature].reshape(-1, 1) + lm_l[1]))
                
                self.y[indices_left] = self.y0[indices_left] - np.maximum(
                    np.minimum(self.y0[indices_left] - rawres_left, self.B1), self.B2
                )
            
                # --- Update residuals for the RIGHT child ---
                if best_node == "lasso_split":
                    # Use the multivariate model for prediction
                    X_model_features_R = self.X[np.ix_(indices_right, best_multi_indices_R + 1)]
                    prediction_R = (X_model_features_R @ best_multi_coeffs_R) + best_multi_intercept_R
                    rawres_right = self.y[indices_right] - prediction_R.reshape(-1, 1)

                else:
                    # Use the original univariate logic
                    rawres_right = (self.y[indices_right] - (lm_r[0] * self.X[indices_right, best_feature].reshape(-1, 1) + lm_r[1]))
            
                self.y[indices_right] = self.y0[indices_right] - np.maximum(
                    np.minimum(self.y0[indices_right] - rawres_right, self.B1), self.B2
                )

                # --- Recurse on the new LEFT and RIGHT child indices ---
                try:
                    self.recursion_counter[best_node] += 1

                    node.left = self.build_tree_multi(
                        tree_depth=tree_depth,
                        model_depth=model_depth,
                        indices=indices_left,
                        rss=np.maximum(0, np.sum((self.y[indices_left] - np.mean(self.y[indices_left])) ** 2)),
                        alpha=alpha,
                        multi_lars=multi_lars,
                        finalist_s=finalist_s,
                        finalist_d=finalist_d,
                        per_feature=per_feature,
                        full_multi=full_multi,
                        importances=importances
                    )
                    
                    node.right = self.build_tree_multi(
                        tree_depth=tree_depth,
                        model_depth=model_depth,
                        indices=indices_right,
                        rss=np.maximum(0, np.sum((self.y[indices_right] - np.mean(self.y[indices_right])) ** 2)),
                        alpha=alpha,
                        multi_lars=multi_lars,
                        finalist_s=finalist_s,
                        finalist_d=finalist_d,
                        per_feature=per_feature,
                        full_multi=full_multi,
                        importances=importances
                    )

                except RecursionError:
                    print(f"ERROR: encountered recursion error, return END node. Current counter: {self.recursion_counter}")
                    return tree_multi(node="END", Rt=rss)
                    
        else:
            # stop recursion if meeting the stopping criterion
            return tree_multi(node="END", Rt=rss)
            
        return node
        
    def predict_multi(self, X, model=None, maxd=np.inf, **kwargs):
        """
        Generates predictions for new data using a fitted Multi-PILOT tree.

        This method traverses the `tree_multi` structure for each data point. It
        correctly handles all node types, including the new 'lasso_no_split'
        and 'lasso_split' nodes, by applying the appropriate multivariate
        linear models to generate the final prediction.

        Args:
            X (array-like): The new data samples to predict.
            model (tree_multi, optional): The fitted tree model. If None, uses
                `self.model_tree`.
            maxd (int, optional): The maximum depth to traverse for prediction.

        Returns:
            np.ndarray: The array of predicted values.
        """
        y_hat = []
        if model is None:
            model = self.model_tree
    
        if isinstance(X, pd.core.frame.DataFrame):
            X = np.array(X)
    
        if self.model_tree.node == "END":
            return np.ones(X.shape[0]) * self.ymean
    
        for row in range(X.shape[0]): 
            t = model
            y_hat_one = 0
            while t.node != "END" and t.depth < maxd:
                
                # --- HANDLE NO-SPLIT MODELS FIRST ---
                if t.node == "con":
                    # A constant model's prediction is simply its intercept.
                    # lm_l[0] is always 0 for a 'con' node.
                    y_hat_one += self.step_size * t.lm_l[1]
                    t = t.left
    
                elif t.node == "lin":
                    # Univariate linear model with input clamping.
                    pred = t.lm_l[0] * np.min([np.max([X[row, t.pivot[0]], t.interval[0]]), t.interval[1]]) + t.lm_l[1]
                    y_hat_one += self.step_size * pred
                    t = t.left
    
                elif t.node == "lasso_no_split":
                    # Check for valid multivariate parameters before predicting.
                    if t.multi_model_coeffs_L is not None and t.multi_model_intercept_L is not None:
                        feature_values = X[row, t.multi_model_indices_L]
                        # Handle case where no features were selected (empty arrays).
                        if feature_values.size > 0:
                            pred = np.sum(feature_values * t.multi_model_coeffs_L) + t.multi_model_intercept_L
                        else: # If no coeffs, prediction is just the intercept.
                            pred = t.multi_model_intercept_L
                        y_hat_one += self.step_size * pred
                    t = t.left
    
                # --- HANDLE ALL SPLIT MODELS ---
                else:
                    # First, determine the path (left or right)
                    go_left = False
                    if t.node == "pconc": # Categorical split
                        if np.isin(X[row, t.pivot[0]], t.pivot_c):
                            go_left = True
                    else: # All other numerical splits (pcon, plin, blin, lasso_split)
                        if X[row, t.pivot[0]] <= t.pivot[1]:
                            go_left = True
                    
                    # Now, calculate prediction based on the chosen path
                    if go_left:
                        if t.node == "lasso_split":
                            # Check for valid multivariate parameters
                            if t.multi_model_coeffs_L is not None and t.multi_model_intercept_L is not None:
                                feature_values = X[row, t.multi_model_indices_L]
                                if feature_values.size > 0:
                                    pred = np.sum(feature_values * t.multi_model_coeffs_L) + t.multi_model_intercept_L
                                else:
                                    pred = t.multi_model_intercept_L
                                y_hat_one += self.step_size * pred
                        else: # Univariate models on left path
                            pred = t.lm_l[0] * np.max([X[row, t.pivot[0]], t.interval[0]]) + t.lm_l[1]
                            y_hat_one += self.step_size * pred
                        t = t.left
                    else: # Go Right
                        if t.node == "lasso_split":
                            # Check for valid multivariate parameters
                            if t.multi_model_coeffs_R is not None and t.multi_model_intercept_R is not None:
                                feature_values = X[row, t.multi_model_indices_R]
                                if feature_values.size > 0:
                                    pred = np.sum(feature_values * t.multi_model_coeffs_R) + t.multi_model_intercept_R
                                else:
                                    pred = t.multi_model_intercept_R
                                y_hat_one += self.step_size * pred
                        else: # Univariate models on right path
                            pred = t.lm_r[0] * np.min([X[row, t.pivot[0]], t.interval[1]]) + t.lm_r[1]
                            y_hat_one += self.step_size * pred
                        t = t.right
    
                # Apply the overall prediction truncation after each step
                if y_hat_one > self.B1:
                    y_hat_one = self.B1
                elif y_hat_one < self.B2:
                    y_hat_one = self.B2
    
            y_hat.append(y_hat_one)
            
        return np.array(y_hat)

    def predict(self, X, model=None, maxd=np.inf, **kwargs):
        """
        Generates predictions for new data using a fitted standard PILOT tree.

        This method traverses the tree for each data point, accumulating the
        predictions from the models at each node along the path until a leaf
        is reached or `maxd` is exceeded.

        Args:
            X (array-like): The new data samples to predict.
            model (tree, optional): The fitted tree model. If None, uses
                `self.model_tree`.
            maxd (int, optional): The maximum depth to traverse for prediction.

        Returns:
            np.ndarray: The array of predicted values.
        """
        y_hat = []
        if model is None:
            model = self.model_tree

        if isinstance(X, pd.core.frame.DataFrame):
            X = np.array(X)

        if self.model_tree.node == "END":
            return np.ones(X.shape[0]) * self.ymean

        # The loop for each row in the test data
        for row in range(X.shape[0]):
            t = model
            y_hat_one = 0
            while t.node != "END" and t.depth < maxd:
                if t.node == "pconc":
                    if np.isin(X[row, t.pivot[0]], t.pivot_c):
                        y_hat_one += self.step_size * (t.lm_l[1])
                        t = t.left
                    else:
                        y_hat_one += self.step_size * (t.lm_r[1])
                        t = t.right

                # Go left if 'lin'
                elif t.node in ["lin", "con"] or X[row, t.pivot[0]] <= t.pivot[1]:
                    if t.node == "lin":
                        # Truncate both on the left and the right
                        y_hat_one += self.step_size * (
                            t.lm_l[0]
                            * np.min(
                                [
                                    np.max([X[row, t.pivot[0]], t.interval[0]]),
                                    t.interval[1],
                                ]
                            )
                            + t.lm_l[1]
                        )
                    else:
                        # Truncate on the left
                        y_hat_one += self.step_size * (
                            t.lm_l[0] * np.max([X[row, t.pivot[0]], t.interval[0]]) + t.lm_l[1]
                        )
                    t = t.left

                else:
                    y_hat_one += self.step_size * (
                        t.lm_r[0] * np.min([X[row, t.pivot[0]], t.interval[1]]) + t.lm_r[1]
                    )
                    t = t.right

                # Truncation
                if y_hat_one > self.B1:
                    y_hat_one = self.B1
                elif y_hat_one < self.B2:
                    y_hat_one = self.B2

            y_hat.append(y_hat_one)
        return np.array(y_hat)

    def _validate_X_predict(self, X, *args, **kwargs):
        return X
    
    def print_tree(self, level: int = 2, feature_names: list[str] | None = None) -> None:
        """
        Prints a textual representation of the fitted model tree.

        This method automatically detects whether the fitted tree is a standard
        or multivariate PILOT tree and calls the appropriate printing helper function.

        Args:
            level (int): The initial indentation level for printing.
            feature_names (list, optional): A list of feature names for more
                interpretable output.
        """
        if self.model_tree is None:
            print("Model has not been fitted yet.")
            return

        # Check if we are dealing with a multivariate tree
        if hasattr(self.model_tree, 'multi_model_indices_L'):
            print("--- Multivariate PILOT Tree ---")
            # Import the new function or define it in the same file
            print_tree_multi(self.model_tree, 0, feature_names)
        else:
            print("--- Univariate PILOT Tree ---")
            # Import or use the original function
            print_tree_inner_function(self.model_tree, level) # Assumes original function is available

    def get_model_summary_multi(self, feature_names: list[str] | None = None) -> pd.DataFrame:
        """
        Returns a detailed summary of the model tree as a pandas DataFrame.

        This method supports both standard and multivariate trees, capturing all
        relevant parameters for each node.

        Args:
            feature_names (list, optional): A list of feature names to include
                in the summary for better readability.

        Returns:
            pd.DataFrame: A DataFrame where each row represents a node in the tree.
        """
        if self.model_tree is None:
            print("Model has not been fitted yet.")
            return pd.DataFrame()

        summary = []
        # Check if we are dealing with a multivariate tree
        if hasattr(self.model_tree, 'multi_model_indices_L'):
            # Import the new function or define it in the same file
            tree_summary_multi(self.model_tree, 0, "root", None, summary)
        else:
            # Import or use the original function
            tree_summary(self.model_tree, 0, "root", None, summary)

        summary_df = pd.DataFrame(summary)
        if feature_names is not None and 'pivot_idx' in summary_df.columns:
            summary_df = summary_df.assign(
                pivot_name=lambda x: x.pivot_idx.map(
                    lambda y: feature_names[int(y)] if y is not None and not np.isnan(y) else None
                )
            )
        return summary_df

    def save_importances_to_csv(self, feature_names: list[str], filename: str = "feature_importances.csv",
                                previous_importances: dict[str, float] | None = None):
        """
        Normalizes the calculated feature importances so they sum to 1,
        pairs them with their names, and saves them to a CSV file.

        This method should only be called after fitting the model with
        `compute_importances=True`.

        Args:
            feature_names (list[str]): A list of strings containing the names
                of the features in the same order as the original data.
            filename (str, optional): The name of the CSV file to create.
                Defaults to "feature_importances.csv".
            previous_importances (dict, optional): A dictionary mapping feature names
            to their importance scores from a pre-model (e.g., Ridge/Lasso).
        """
        if self.feature_importances_ is not None:
            pilot_importances = self.feature_importances_
        else:
            # If PILOT didn't compute importances, start with an array of zeros.
            pilot_importances = np.zeros(len(feature_names))

        # Create a combined dictionary of importances.
        combined_importances = {name: 0.0 for name in feature_names}

        # Add PILOT's importances
        for i, name in enumerate(feature_names):
            combined_importances[name] += pilot_importances[i]

        # Add importances from the previous model
        if previous_importances:
            for name, importance in previous_importances.items():
                if name in combined_importances:
                    combined_importances[name] += importance
                else:
                    # This might happen if one-hot encoding creates new names
                    # For now, we can print a warning or handle as needed.
                    print(f"Warning: Feature '{name}' from pre-model not found in PILOT features.")

        # 3. Normalize the combined importances
        raw_scores = np.array(list(combined_importances.values()))
        total_importance = np.sum(raw_scores)

        if total_importance > 0:
            normalized_scores = raw_scores / total_importance
        else:
            normalized_scores = raw_scores

        # 4. Create and save the DataFrame
        importances_df = pd.DataFrame({
            'feature_name': list(combined_importances.keys()),
            'importance': normalized_scores
        })
        importances_df = importances_df.sort_values(by='importance', ascending=False)

        try:
            importances_df.to_csv(filename, index=False)
            print(f"Combined feature importances successfully saved to '{filename}'")
        except Exception as e:
            print(f"An error occurred while writing to the file: {e}")

    def calculate_ensemble_importances(self,
                                       pre_model,
                                       y_train_original,
                                       y_pred_pre_model_train,
                                       pilot_feature_names: list[str],
                                       pre_model_feature_names: list[str],
                                       categorical_feature_names: list[str],
                                       filename: str = "ensemble_importances.csv"):
         """
        Calculates and saves the combined feature importances for a two-stage model.

        This method first computes the importance of the `pre_model` (e.g., Ridge)
        by attributing its total RSS reduction to its features, proportional to their
        coefficient magnitudes. It then combines these scores with the importances
        from the PILOT tree and saves the final, normalized result.

        Args:
            pre_model (object): The fitted pre-model (e.g., Ridge, Lasso).
            y_train_original (np.ndarray): The original target values.
            y_pred_pre_model_train (np.ndarray): Predictions from the pre-model.
            pilot_feature_names (list): Names of features used by PILOT.
            pre_model_feature_names (list): Names of features used by the pre-model.
            categorical_feature_names (list): Base names of categorical features.
            filename (str): The path for the output CSV file.
        """
        # --- 1. Calculate the total importance gain from the pre-model ---
        rss_con = np.sum((y_train_original - np.mean(y_train_original)) ** 2)
        rss_pre_model = np.sum((y_train_original - y_pred_pre_model_train) ** 2)
        pre_model_total_gain = rss_con - rss_pre_model

        # --- 2. Calculate and Distribute the Pre-Model's Gain ---
        pre_model_importances = {name: 0.0 for name in pilot_feature_names}
        if pre_model_total_gain > 0 and hasattr(pre_model, 'coef_'):
            pre_model_coeffs = np.abs(pre_model.coef_.flatten())
            total_coeffs = np.sum(pre_model_coeffs)

            if total_coeffs > 0:
                # Distribute the total RSS gain proportionally to the coefficients
                proportions = pre_model_coeffs / total_coeffs
                distributed_gain = pre_model_total_gain * proportions

                # Aggregate for one-hot encoded features
                for i, full_name in enumerate(pre_model_feature_names):
                    original_name = full_name
                    for cat_name in categorical_feature_names:
                        if full_name.startswith(cat_name + '_'):
                            original_name = cat_name
                            break
                    if original_name in pre_model_importances:
                        pre_model_importances[original_name] += distributed_gain[i]

        # --- 3. Call the saving function with the calculated scores ---
        self.save_importances_to_csv(
            feature_names=pilot_feature_names,
            filename=filename,
            previous_importances=pre_model_importances  # Pass the calculated scores
        )

    def print_ensemble_model(self,
                         pre_model,
                         pre_model_name: str,
                         pilot_feature_names: list[str] = None,
                         pre_model_feature_names: list[str] = None,
                         categorical_feature_names: list[str] = None):
        """
        Prints a structured summary of a two-stage ensemble model.

        It displays the linear equation of the pre-model followed by the textual
        representation of the PILOT tree that was fitted on the pre-model's residuals.

        Args:
            pre_model (object): The fitted pre-model (e.g., Ridge, Lasso).
            pre_model_name (str): The name of the pre-model.
            pilot_feature_names (list, optional): Feature names for the PILOT tree.
            pre_model_feature_names (list, optional): Feature names for the pre-model.
            categorical_feature_names (list, optional): Base names of categorical features.
        """
        print("="*60)
        print(f"Ensemble Model Structure")
        print("="*60)
        print(f"Stage 1: {pre_model_name} Model\n")

        # --- 1. Print the Pre-Model Summary (Precise Version) ---
        if not hasattr(pre_model, 'coef_'):
            print("  (Pre-model has no coefficients to display)")
        else:
            intercept = pre_model.intercept_
            coeffs = pre_model.coef_.flatten()

            # Build the equation string term by term
            equation = f"  y = {intercept:.3f}"

            # Iterate through all coefficients from the pre-model
            for i, full_name in enumerate(pre_model_feature_names):
                coef = coeffs[i]

                if abs(coef) > 1e-6:  # Only print non-zero coefficients
                    sign = "+" if coef >= 0 else "-"

                    # Check if the feature is categorical to apply custom formatting
                    display_name = full_name
                    for cat_name in categorical_feature_names:
                        if full_name.startswith(cat_name + '_'):
                            # It's a one-hot encoded feature. Let's reformat the name.
                            # Example: 'color_blue' -> 'color_(blue)'
                            class_name = full_name[len(cat_name)+1:]
                            display_name = f"{cat_name}_({class_name})"
                            break

                    equation += f" {sign} {abs(coef):.3f} * {display_name}"

            print(equation)
            print("\n" + "-"*30 + "\n")

        # --- 2. Print the PILOT Tree ---
        print("Stage 2: PILOT Tree on Residuals\n")
        if self.model_tree:
            # Calling the tree printing function (ensure it's the corrected version)
            # You might need to import it or make it accessible if it's outside the class
            print_tree_multi(self.model_tree, level=0, feature_names=pilot_feature_names)
        else:
            print("  (No PILOT tree was fitted)")

        print("="*60)


def print_tree_inner_function(model_tree: tree, level: int) -> None:
    """
    Helper function to recursively print a standard (univariate) PILOT tree.
    """
    if model_tree is None:
        return

    print_tree_inner_function(model_tree.right, level + 1)

    indent = " " * 8 * level + "--> "

    # --- Determine if the model at this node is ACTIVE ---
    is_active_model = False
    if model_tree.node not in ["con", "lin", "END"]:  # A split node
        is_active_model = True
    elif model_tree.node == "con":  # An active terminal model
        is_active_model = True
    elif model_tree.node == "lin":  # Active only if path continues
        if model_tree.left and model_tree.left.node != "END":
            is_active_model = True

    # --- Print based on whether the model is active ---
    if not is_active_model:
        print(f"{indent}END (RSS: {model_tree.Rt:.2f})")
    else:
        # Model is active, print its details
        if model_tree.node == "lin":
            print(
                f"{indent}{model_tree.node} "
                f"({round(model_tree.pivot[0], 3)}) "
                f"RSS:{round(model_tree.Rt, 3)} "
                f"LM:({round(model_tree.lm_l[0], 3)}, {round(model_tree.lm_l[1], 3)})"
            )
        else:  # For 'con' and all split nodes
            pivot_str = f"({round(model_tree.pivot[0], 3)}, {round(model_tree.pivot[1], 3)})"
            lm_l_str = f"({round(model_tree.lm_l[0], 3)}, {round(model_tree.lm_l[1], 3)})"
            lm_r_str = f"({round(model_tree.lm_r[0], 3)}, {round(model_tree.lm_r[1], 3)})" if model_tree.lm_r is not None else "None"
            print(
                f"{indent}{model_tree.node} {pivot_str} "
                f"RSS:{round(model_tree.Rt, 3)} "
                f"LM_L:{lm_l_str} LM_R:{lm_r_str}"
            )

    print_tree_inner_function(model_tree.left, level + 1)


def tree_summary(model_tree, level, tree_id, parent_id, summary):
    """
    Helper function to recursively gather summary data from a standard PILOT tree.
    """
    if model_tree is not None:
        if model_tree.pivot is not None:
            pivot_idx = model_tree.pivot[0]
            pivot_value = model_tree.pivot[1]
        else:
            pivot_idx = pivot_value = None
        summary.append(
            {
                "tree_id": tree_id,
                "parent_id": parent_id,
                "level": level,
                "node": model_tree.node,
                "pivot_idx": pivot_idx,
                "pivot_value": pivot_value,
                "Rt": model_tree.Rt,
                "lm_l": model_tree.lm_l,
                "lm_r": model_tree.lm_r,
            }
        )
        tree_summary(model_tree.left, level + 1, str(uuid.uuid4()).split("-")[-1], tree_id, summary)
        tree_summary(
            model_tree.right, level + 1, str(uuid.uuid4()).split("-")[-1], tree_id, summary
        )

def print_tree_multi(model_tree: tree_multi, level: int, feature_names: list[str] | None = None) -> None:
    """
    Helper function to recursively print a Multi-PILOT tree.

    This function provides an enhanced textual representation that correctly
    displays information for all node types, including numerical splits,
    categorical splits, and multivariate models.

    Args:
        model_tree (tree_multi): The current node in the tree to print.
        level (int): The current depth for indentation.
        feature_names (list, optional): Names of features for interpretability.
    """
    if model_tree is None:
        return # Base case for recursion

    # Recurse on the right child first to print the tree in a more intuitive way
    print_tree_multi(model_tree.right, level + 1, feature_names)

    # --- Display logic for the current node ---
    indent = " " * 8 * level + "--> "

    # --- Determine if the model at this node is ACTIVE ---
    is_active_model = False
    if model_tree.node not in ["con", "lin", "lasso_no_split", "END"]:
        # Split nodes are always considered active if they exist
        is_active_model = True
    elif model_tree.node == "con":
        # 'con' is an active terminal model
        is_active_model = True
    elif model_tree.node in ["lin", "lasso_no_split"]:
        # Active only if the path continues
        if model_tree.left and model_tree.left.node != "END":
            is_active_model = True

    if not is_active_model:
        # If the model is not active, it's a dead end. Just print END.
        print(f"{indent}END (RSS: {model_tree.Rt:.2f})")
    else:

        # --- NEW: Intelligent Pivot Information Generation ---
        pivot_info = ""
        # Only try to get pivot info for nodes that actually perform a split
        if model_tree.node not in ["con", "lin", "lasso_no_split", "END"]:

            feature_idx = int(model_tree.pivot[0])
            feature_name = feature_names[feature_idx] if feature_names else f"Feat_{feature_idx}"

            # Check if the split rule is CATEGORICAL
            # This handles a 'pconc' node OR a 'lasso_split' that was upgraded from a 'pconc'.
            # Assumes 'best_node_prev' is stored on the node object.
            if model_tree.node == 'pconc' or (model_tree.node == 'lasso_split' and model_tree.best_node_prev == 'pconc'):
                 levels = model_tree.pivot_c
                 # Format the output to be readable
                 if len(levels) > 3:
                     pivot_info = f"{feature_name} in [{levels[0]}, ..., {levels[-1]}]"
                 else:
                     pivot_info = f"{feature_name} in {list(levels)}"

            # Otherwise, the split rule must be NUMERICAL
            else:
                 pivot_value = round(model_tree.pivot[1], 2)
                 pivot_info = f"{feature_name} <= {pivot_value}"

        # --- Node-Specific Printing Logic ---
        if model_tree.node == "END":
            print(f"{indent}END (RSS: {model_tree.Rt:.2f})")

        elif model_tree.node in ["con", "lin"]:
            print(f"{indent}{model_tree.node} (RSS: {model_tree.Rt:.2f})")
            print_tree_multi(model_tree.left, level + 1, feature_names)

        elif model_tree.node == "lasso_no_split":
            num_feats = len(model_tree.multi_model_indices_L) if model_tree.multi_model_indices_L is not None else 0
            print(f"{indent}{model_tree.node} (MV model: {num_feats} feats) (RSS: {model_tree.Rt:.2f})")
            print_tree_multi(model_tree.left, level + 1, feature_names)

        elif model_tree.node == "lasso_split":
            num_feats_L = len(model_tree.multi_model_indices_L) if model_tree.multi_model_indices_L is not None else 0
            num_feats_R = len(model_tree.multi_model_indices_R) if model_tree.multi_model_indices_R is not None else 0
            # Now uses the correctly generated pivot_info
            print(f"{indent}{model_tree.node} on [{pivot_info}] (RSS: {model_tree.Rt:.2f})")
            print(f"{' ' * 8 * (level+1)} L: (MV model: {num_feats_L} feats), R: (MV model: {num_feats_R} feats)")
            print_tree_multi(model_tree.left, level + 1, feature_names)

        else: # Handles pcon, plin, blin, pconc
            # Also uses the correctly generated pivot_info
            print(f"{indent}{model_tree.node} on [{pivot_info}] (RSS: {model_tree.Rt:.2f})")
            print_tree_multi(model_tree.left, level + 1, feature_names)

def tree_summary_multi(model_tree: tree_multi, level: int, tree_id: str, parent_id: str | None, summary: list) -> None:
    """
    Helper function to recursively gather summary data from a Multi-PILOT tree.

    This function extends `tree_summary` to also capture the parameters of
    multivariate models stored in `tree_multi` nodes.

    Args:
        model_tree (tree_multi): The current node to summarize.
        level (int): The current depth of the node.
        tree_id (str): A unique ID for the current node.
        parent_id (str, optional): The ID of the parent node.
        summary (list): The list where node summary dictionaries are appended.
    """
    if model_tree is not None:
        if model_tree.pivot is not None:
            pivot_idx = model_tree.pivot[0]
            pivot_value = model_tree.pivot[1]
        else:
            pivot_idx = pivot_value = None

        # Create a dictionary with all possible fields
        node_summary = {
            "tree_id": tree_id,
            "parent_id": parent_id,
            "level": level,
            "node": model_tree.node,
            "pivot_idx": pivot_idx,
            "pivot_value": pivot_value,
            "Rt": model_tree.Rt,
            "lm_l": model_tree.lm_l,
            "lm_r": model_tree.lm_r,
            # --- ADD NEW MULTIVARIATE ATTRIBUTES ---
            "multi_model_indices_L": model_tree.multi_model_indices_L,
            "multi_model_coeffs_L": model_tree.multi_model_coeffs_L,
            "multi_model_intercept_L": model_tree.multi_model_intercept_L,
            "multi_model_indices_R": model_tree.multi_model_indices_R,
            "multi_model_coeffs_R": model_tree.multi_model_coeffs_R,
            "multi_model_intercept_R": model_tree.multi_model_intercept_R,
        }
        summary.append(node_summary)

        # Recurse 
        if model_tree.left:
            tree_summary_multi(model_tree.left, level + 1, str(uuid.uuid4()).split("-")[-1], tree_id, summary)
        if model_tree.right:
            tree_summary_multi(model_tree.right, level + 1, str(uuid.uuid4()).split("-")[-1], tree_id, summary)