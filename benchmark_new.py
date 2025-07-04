import json
import pathlib
import click
import os
import glob
import psutil
import itertools
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.linear_model._coordinate_descent import _alpha_grid
from sklearn.preprocessing import StandardScaler

from pilot import DEFAULT_DF_SETTINGS
from benchmark_util_new import *

from benchmark_config import NEW_DATASET_NAMES


def print_with_timestamp(message):
    """Prints a message to the console, prefixed with the current timestamp.

    Args:
        message (str): The message to be printed.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

OUTPUTFOLDER = pathlib.Path(__file__).parent / "Output"
DATAFOLDER = pathlib.Path(__file__).parent / "Data"

df_setting_alpha01 = dict(
    zip(
        DEFAULT_DF_SETTINGS.keys(),
        1 + 0.01 * (np.array(list(DEFAULT_DF_SETTINGS.values())) - 1),
    )
)

df_setting_alpha5 = dict(
    zip(
        DEFAULT_DF_SETTINGS.keys(),
        1 + 0.5 * (np.array(list(DEFAULT_DF_SETTINGS.values())) - 1),
    )
)

df_setting_alpha01_no_blin = df_setting_alpha01.copy()
df_setting_alpha01_no_blin["blin"] = -1

df_setting_alpha5_no_blin = df_setting_alpha5.copy()
df_setting_alpha5_no_blin["blin"] = -1

df_setting_no_blin = DEFAULT_DF_SETTINGS.copy()
df_setting_no_blin["blin"] = -1

# Define your model names (make them consistent with your `inner_cv_fold_scores` keys)
ALL_MODEL_NAMES = ["CART", "PILOT", "RF", "RAFFLE", "XGB", "Ridge", "Lasso", 
                  "Ridge_pilot_ensemble", "Lasso_pilot_ensemble",
                  "PILOT_NLFS_prefix", "PILOT_NLFS_LARS",
                  "PILOT_NLFS_prefix_tuning", "PILOT_NLFS_LARS_tuning", 
                  "PILOT_NLFS_fallback", "PILOT_NLFS_fallback_tuning",
                  "PILOT_finalist_S_LARS", "PILOT_finalist_S_prefix",
                  "PILOT_finalist_D_LARS", "PILOT_finalist_D_prefix",
                  "PILOT_per_feature_LARS", "PILOT_per_feature_prefix",
                  "PILOT_full_multi_LARS", "PILOT_full_multi_prefix",
                  "PILOT_finalist_S_LARS_df", "PILOT_finalist_S_prefix_df",
                  "PILOT_finalist_D_LARS_df", "PILOT_finalist_D_prefix_df",
                  "PILOT_per_feature_LARS_df", "PILOT_per_feature_prefix_df",
                  "PILOT_full_multi_LARS_df", "PILOT_full_multi_prefix_df",
                   "PILOT_df"]

@click.command()
@click.option("--experiment_name", "-e", required=True, help="Name of the experiment")
@click.option(
    "--models_to_run",
    "-m",
    default=",".join(ALL_MODEL_NAMES), # Default to all models
    help=(
        "Comma-separated list of model names to run (case-insensitive). "
        f"Available: {', '.join(ALL_MODEL_NAMES)}. "
        "Example: CART,PILOT,RF"
    ),
)
@click.option(
    "--datasets_to_run",
    "-d",
    default=",".join(NEW_DATASET_NAMES), # Default to all models
    help=(
        "Comma-separated list of dataset names to run (case-sensitive). "
        f"Available: {', '.join(NEW_DATASET_NAMES)}. "
        "Example: airfoil,communities,admission"
    ),
)
def run_benchmark(experiment_name, models_to_run, datasets_to_run):
    """
    Executes a comprehensive benchmark for various regression models.

    This function orchestrates the entire experimental process, including:
    1.  **Setup**: Creates output directories for results, visualizations,
        and saved artifacts based on the experiment name.
    2.  **Argument Parsing**: Determines which models and datasets to run based
        on command-line arguments.
    3.  **Data Loading & Preprocessing**: Loads specified datasets, handles
        missing values, and applies robust power transformations to numerical features.
    4.  **Nested Cross-Validation**:
        -   **Outer Loop (5-fold CV)**: Splits the data into 5 train/test folds
            for final model evaluation. This provides a robust estimate of
            performance on unseen data.
        -   **Inner Loop (5-fold CV)**: Within each outer training fold, an inner
            cross-validation is performed to tune the hyperparameters for each
            model. This ensures that hyperparameter selection does not leak
            information from the outer test set.
    5.  **Hyperparameter Tuning**: For each model, it iterates through a predefined
        grid of hyperparameters in the inner loop, calculating the average
        validation MSE. The set of hyperparameters with the lowest average MSE
        is selected as the best.
    6.  **Final Evaluation**: The model is retrained on the *full outer training set*
        using the best hyperparameters found in the inner loop. It is then
        evaluated on the corresponding *outer test set*.
    7.  **Results Aggregation**: Performance metrics (R2, MSE, MAE), timings,
        dataset characteristics, and chosen hyperparameters are collected for
        each model and each fold.
    8.  **Checkpointing**: Results are saved to a CSV file after each dataset is
        processed, allowing the experiment to be resumed if interrupted.

    Args:
        experiment_name (str): The name for the experiment, used to create a
            dedicated output folder.
        models_to_run (str): A comma-separated string of model names to include
            in the benchmark run.
        datasets_to_run (str): A comma-separated string of dataset names to use.
    """
    # Make the folders to save the output in
    experiment_folder = OUTPUTFOLDER / experiment_name
    experiment_folder.mkdir(exist_ok=True)
    
    fold_indices_folder = experiment_folder / "fold_indices"
    fold_indices_folder.mkdir(exist_ok=True)
    
    real_datasets_folder = experiment_folder / "real_datasets"
    real_datasets_folder.mkdir(exist_ok=True)
    
    visualizations_folder = experiment_folder / "visualizations"
    visualizations_folder.mkdir(exist_ok=True)

    importances_folder = experiment_folder / "importances"
    importances_folder.mkdir(exist_ok=True)
    
    experiment_file = experiment_folder / "results.csv"
    print(f"Results will be stored in {experiment_file}")

    # Make the outer folds splits
    np.random.seed(42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # --- Process the models_to_run input ---
    if models_to_run.lower() == "all" or not models_to_run: # Allow "all" or empty to mean all
        selected_models_set = set(m.upper() for m in ALL_MODEL_NAMES)
    else:
        user_selected_models = [m.strip().upper() for m in models_to_run.split(',')]
        selected_models_set = set()
        for model_name in user_selected_models:
            if model_name in [m.upper() for m in ALL_MODEL_NAMES]:
                selected_models_set.add(model_name)
            else:
                print_with_timestamp(
                    f"Warning: Model '{model_name}' not recognized and will be skipped. "
                    f"Available models: {', '.join(ALL_MODEL_NAMES)}"
                )
    print_with_timestamp(f"Models selected for this run: {', '.join(sorted(list(selected_models_set)))}")
    
    # --- Process the datasets_to_run input ---
    if datasets_to_run.lower() == "all" or not datasets_to_run: # Allow "all" or empty to mean all
        selected_datasets = set(d for d in NEW_DATASET_NAMES)
    else:
        user_selected_datasets = [d.strip() for d in datasets_to_run.split(',')]
        selected_datasets = set()
        for dataset_name in user_selected_datasets:
            if dataset_name in [d for d in NEW_DATASET_NAMES]:
                selected_datasets.add(dataset_name)
            else:
                print_with_timestamp(
                    f"Warning: Model '{dataset_name}' not recognized and will be skipped. "
                    f"Available models: {', '.join(NEW_DATASET_NAMES)}"
                )
    print_with_timestamp(f"Datasets selected for this run: {', '.join(sorted(list(selected_datasets)))}")

    # Check if the experiment has already been run (same name of experiment)
    if experiment_file.exists():
        results = pd.read_csv(experiment_file)
        processed_repo_ids = results["id"].unique().astype(str)
        results = results.to_dict("records")
    else:
        results = []
        processed_repo_ids = []

    repo_ids_to_process = [
        pathlib.Path(f).stem
        for f in glob.glob(os.path.join(DATAFOLDER, "*"))
        if pathlib.Path(f).stem not in processed_repo_ids
    ]
    
    # --- Make the grids for all models ---
    cart_param_values = {
        'max_depth': [3, 6, 9, 12, 15, 18],
        'min_samples_split': [2, 10, 20, 40],
        'min_samples_leaf': [1, 5, 10, 20],
        'ccp_alpha': [0.0, 0.0005, 0.001, 0.005, 0.01, 0.02],
    }
    cart_param_names = list(cart_param_values.keys())
    cart_hp_combinations = itertools.product(*(cart_param_values[name] for name in cart_param_names))
    cart_grid_list_of_dicts = [dict(zip(cart_param_names, combo)) for combo in cart_hp_combinations]
            
    pilot_param_values = {
        'max_depth': [3, 6, 9, 12, 15, 18],      
        'min_sample_split': [2, 10, 20, 40],    
        'min_sample_leaf': [1, 5, 10, 20],     
    }
    pilot_param_names = list(pilot_param_values.keys())
    pilot_hp_combinations = itertools.product(*(pilot_param_values[name] for name in pilot_param_names))
    pilot_grid_list_of_dicts = [dict(zip(pilot_param_names, combo)) for combo in pilot_hp_combinations]
            
    rf_param_values = {
        'max_depth': [6, 20, None],
        'max_features': [0.7, 1.0],
        'n_estimators': [100], # Assuming n_estimators is fixed at 100 for tuning
    }
    rf_param_names = list(rf_param_values.keys())
    rf_hp_combinations = itertools.product(*(rf_param_values[name] for name in rf_param_names))
    rf_grid_list_of_dicts = [dict(zip(rf_param_names, combo)) for combo in rf_hp_combinations]
            
    raffle_df_configs = [
        {'df_name': "df alpha = 0.01, no blin", 'alpha_cpf': 0.01, 'df_settings_obj': df_setting_alpha01_no_blin},
        {'df_name': "df no blin",             'alpha_cpf': 1,    'df_settings_obj': df_setting_no_blin},
        {'df_name': "df alpha = 0.5, no blin", 'alpha_cpf': 0.5,  'df_settings_obj': df_setting_alpha5_no_blin},
    ]
    raffle_max_depth_values = [6, 20]
    raffle_max_features_values = [0.7, 1.0] # Corresponds to 'n_features_node' in RandomForestCPilot
    raffle_n_estimators_values = [100]
    value_combinations = itertools.product(
        raffle_df_configs,             # Each element is a dict
        raffle_max_depth_values,
        raffle_max_features_values,
        raffle_n_estimators_values
    )
    raffle_grid_list_of_dicts = []
    for df_config, md_val, mf_val, ne_val in value_combinations:
        param_set = {
            'n_estimators': ne_val,
            'max_depth': md_val,
            'max_features': mf_val,
            'df_settings': df_config['df_settings_obj'], 
            'alpha': df_config['alpha_cpf'],           
            '_description_df_name': df_config['df_name'] 
        }
        raffle_grid_list_of_dicts.append(param_set)

    xgb_param_values = {
        'max_depth': [6, 20],
        'max_features': [0.7, 1.0], # Maps to max_node_features in your fit_xgboost
        'n_estimators': [100],
    }
    xgb_param_names = list(xgb_param_values.keys())
    xgb_hp_combinations = itertools.product(*(xgb_param_values[name] for name in xgb_param_names))
    xgb_grid_list_of_dicts = [dict(zip(xgb_param_names, combo)) for combo in xgb_hp_combinations]
    
    alphagrid = None
    
    lin_pilot_ensemble_grid = None
    
    # Loop for all datasets to run
    for repo_id in repo_ids_to_process:     
        print_with_timestamp(repo_id)
        kind, repo_id = repo_id.split("_")
        dataset = load_data(repo_id=repo_id, kind=kind)
        
        if dataset.n_samples > 2e5:
            print_with_timestamp(f"Skipping large dataset {repo_id}")
            continue
        
        if repo_id not in selected_datasets:
            continue
          
        # Start the loop for the outer folds
        dataset_fold_indices = []
        real_dataset_transformed = []   
        for i, (train, test) in enumerate(cv.split(dataset.X, dataset.y), start=1):
            # Save the indices of the folds
            dataset_fold_indices.append({
                "fold": i, # 1-based fold number
                "train_indices_0based": [int(x) for x in train], # Python 0-based
                "test_indices_0based": [int(x) for x in test]   # Python 0-based
            })

            indices_filename = fold_indices_folder / f"{repo_id}_fold_indices.json"
            with open(indices_filename, 'w') as f_json:
              json.dump(dataset_fold_indices, f_json, indent=4)
            print_with_timestamp(f"\tSaved fold indices for {repo_id} to {indices_filename}")
            
            print_with_timestamp(f"\tFold {i} / 5")
            print("\tRAM Used (GB):", psutil.virtual_memory()[3] / 1000000000)
            train_dataset = dataset.subset(train)
            test_dataset = dataset.subset(test)

            # Fit the transformers (if dataset is 'Real Estate' use a different transformer
            if repo_id == "real":
                transformers = fit_transformers(train_dataset, True)
            else:
                transformers = fit_transformers(train_dataset, False)

            # Apply the transformers
            for col, transformer in transformers.items():
                train_dataset.apply_transformer(col, transformer)
                test_dataset.apply_transformer(col, transformer)

            # --- If the dataset is 'Real Estate' save the data for each fold to import to R for the M5 model ---
            if repo_id == "real":
                train_set_serializable = None
                train_column_names_serializable = None # Initialize
                if hasattr(train_dataset, 'X_label_encoded') and train_dataset.X_label_encoded is not None:
                    if isinstance(train_dataset.X_label_encoded, pd.DataFrame):
                        train_set_serializable = train_dataset.X_label_encoded.values.tolist()
                        train_column_names_serializable = train_dataset.X_label_encoded.columns.tolist() # Convert Index to list
                    elif isinstance(train_dataset.X_label_encoded, np.ndarray):
                        train_set_serializable = train_dataset.X_label_encoded.tolist()
                        # NumPy arrays don't have .columns, so train_column_names_serializable remains None
                        # or you might have a predefined list of names if it's from a NumPy array
                    else:
                        print(f"Warning: train_dataset.X_label_encoded is of unexpected type: {type(train_dataset.X_label_encoded)}")
            
                test_set_serializable = None
                test_column_names_serializable = None # Initialize
                if hasattr(test_dataset, 'X_label_encoded') and test_dataset.X_label_encoded is not None:
                    if isinstance(test_dataset.X_label_encoded, pd.DataFrame):
                        test_set_serializable = test_dataset.X_label_encoded.values.tolist()
                        test_column_names_serializable = test_dataset.X_label_encoded.columns.tolist() # Convert Index to list
                    elif isinstance(test_dataset.X_label_encoded, np.ndarray):
                        test_set_serializable = test_dataset.X_label_encoded.tolist()
                        # NumPy arrays don't have .columns
                    else:
                        print(f"Warning: test_dataset.X_label_encoded is of unexpected type: {type(test_dataset.X_label_encoded)}")
            
                categorical_names_serializable = None # Initialize
                if hasattr(dataset, 'cat_names') and dataset.cat_names is not None:
                    if isinstance(dataset.cat_names, pd.Index): # Check if it's a pandas Index
                        categorical_names_serializable = dataset.cat_names.tolist() # Convert Index to list
                    elif isinstance(dataset.cat_names, list): # If it's already a list
                        categorical_names_serializable = dataset.cat_names
                    else:
                        print(f"Warning: dataset.cat_names is of unexpected type: {type(dataset.cat_names)}. Attempting to convert.")
                        try:
                            categorical_names_serializable = list(dataset.cat_names) # General attempt to convert to list
                        except TypeError:
                            print(f"Error: Could not convert dataset.cat_names to list.")
            
                current_fold_data = {
                    "fold": i,
                    "train_set_data": train_set_serializable,
                    "train_column_names": train_column_names_serializable, # Use the serializable version
                    "test_set_data": test_set_serializable,
                    "test_column_names": test_column_names_serializable,   # Use the serializable version
                    "categorical_names": categorical_names_serializable    # Use the serializable version
                }
            
                real_dataset_transformed.append(current_fold_data)
            
                real_dataset_filename = real_datasets_folder / f"{repo_id}_label_encoded_datasets.json"
                try:
                    with open(real_dataset_filename, 'w') as f_json:
                        json.dump(real_dataset_transformed, f_json, indent=4)
                    print_with_timestamp(f"\tSaved label-encoded datasets for {repo_id} (up to fold {i}) to {real_dataset_filename}")
                except TypeError as e:
                    print_with_timestamp(f"ERROR: Could not serialize data to JSON for {repo_id}. {e}")
                    for k, v_item in current_fold_data.items():
                        try:
                            json.dumps(v_item)
                        except TypeError:
                            print_with_timestamp(f"Problematic key during re-check: {k}, type: {type(v_item)}")
                    
            # Make the grid of alpha values for Ridge and Lasso
            alphagrid = _alpha_grid(
              train_dataset.X_oh_encoded.values,
              train_dataset.y.values,
              l1_ratio=1,
              fit_intercept=True,
              eps=1e-3,
              n_alphas=100,
              copy_X=False,
            )
            
            # Make a smaller grid of alpha values for the Ridge and Lasso ensemble models
            smaller_alphagrid = _alpha_grid(
              train_dataset.X_oh_encoded.values,
              train_dataset.y.values,
              l1_ratio=1,
              fit_intercept=True,
              eps=1e-3,
              n_alphas=10,
              copy_X=False,
            )
            
            # Combine the smaller alpha grid with PILOT grid for the combined grid for the ensemble models
            lin_pilot_ensemble_grid_list_of_dicts = []
            if pilot_grid_list_of_dicts and smaller_alphagrid is not None and len(smaller_alphagrid) > 0:
                 for pilot_hp_dict_item in pilot_grid_list_of_dicts: 
                     for alpha_val in smaller_alphagrid:
                         combined_hp = pilot_hp_dict_item.copy()
                         combined_hp['alpha'] = alpha_val
                         lin_pilot_ensemble_grid_list_of_dicts.append(combined_hp)
                         
            # Make the inner folds splits
            inner_cv_seed = 42 + i
            cv_tuning = KFold(n_splits=5, shuffle=True, random_state= inner_cv_seed)

            inner_cv_fold_scores = {} 
            time_per_fold = {}

            # --- Initialize only for selected models ---
            for model_name_upper in selected_models_set:
                if model_name_upper == "CART":
                    inner_cv_fold_scores["CART"] = {str(hp_dict): [] for hp_dict in cart_grid_list_of_dicts}
                    time_per_fold["CART"] = 0.0
                elif model_name_upper == "PILOT":
                    inner_cv_fold_scores["PILOT"] = {str(hp_dict): [] for hp_dict in pilot_grid_list_of_dicts}
                    time_per_fold["PILOT"] = 0.0
                elif model_name_upper == "RF":
                    inner_cv_fold_scores["RF"] = {str(hp_dict): [] for hp_dict in rf_grid_list_of_dicts}
                    time_per_fold["RF"] = 0.0
                elif model_name_upper == "RAFFLE":
                    inner_cv_fold_scores["RAFFLE"] = {str(hp_dict): [] for hp_dict in raffle_grid_list_of_dicts}
                    time_per_fold["RAFFLE"] = 0.0
                elif model_name_upper == "XGB":
                    inner_cv_fold_scores["XGB"] = {str(hp_dict): [] for hp_dict in xgb_grid_list_of_dicts}
                    time_per_fold["XGB"] = 0.0
                elif model_name_upper == "RIDGE": # Note: your key is "Ridge", I used "RIDGE" for set consistency
                    inner_cv_fold_scores["Ridge"] = {alpha_val: [] for alpha_val in alphagrid}
                    time_per_fold["Ridge"] = 0.0
                elif model_name_upper == "LASSO":
                    inner_cv_fold_scores["Lasso"] = {alpha_val: [] for alpha_val in alphagrid}
                    time_per_fold["Lasso"] = 0.0
                elif model_name_upper == "RIDGE_PILOT_ENSEMBLE":
                    inner_cv_fold_scores["Ridge_pilot_ensemble"] = {str(hp_dict): [] for hp_dict in lin_pilot_ensemble_grid_list_of_dicts}
                    time_per_fold["Ridge_pilot_ensemble"] = 0.0
                elif model_name_upper == "LASSO_PILOT_ENSEMBLE":
                    inner_cv_fold_scores["Lasso_pilot_ensemble"] = {str(hp_dict): [] for hp_dict in lin_pilot_ensemble_grid_list_of_dicts}
                    time_per_fold["Lasso_pilot_ensemble"] = 0.0
                elif model_name_upper == "PILOT_NLFS_PREFIX":
                    inner_cv_fold_scores["PILOT_NLFS_prefix"] = {str(hp_dict): [] for hp_dict in pilot_grid_list_of_dicts}
                    time_per_fold["PILOT_NLFS_prefix"] = 0.0
                elif model_name_upper == "PILOT_NLFS_LARS":
                    inner_cv_fold_scores["PILOT_NLFS_LARS"] = {str(hp_dict): [] for hp_dict in pilot_grid_list_of_dicts}
                    time_per_fold["PILOT_NLFS_LARS"] = 0.0
                elif model_name_upper == "PILOT_NLFS_PREFIX_TUNING":
                    inner_cv_fold_scores["PILOT_NLFS_prefix_tuning"] = {str(hp_dict): [] for hp_dict in pilot_grid_list_of_dicts}
                    time_per_fold["PILOT_NLFS_prefix_tuning"] = 0.0
                elif model_name_upper == "PILOT_NLFS_LARS_TUNING":
                    inner_cv_fold_scores["PILOT_NLFS_LARS_tuning"] = {str(hp_dict): [] for hp_dict in pilot_grid_list_of_dicts}
                    time_per_fold["PILOT_NLFS_LARS_tuning"] = 0.0
                elif model_name_upper == "PILOT_NLFS_FALLBACK":
                    inner_cv_fold_scores["PILOT_NLFS_fallback"] = {str(hp_dict): [] for hp_dict in pilot_grid_list_of_dicts}
                    time_per_fold["PILOT_NLFS_fallback"] = 0.0
                elif model_name_upper == "PILOT_NLFS_FALLBACK_TUNING":
                    inner_cv_fold_scores["PILOT_NLFS_fallback_tuning"] = {str(hp_dict): [] for hp_dict in pilot_grid_list_of_dicts}
                    time_per_fold["PILOT_NLFS_fallback_tuning"] = 0.0
                elif model_name_upper == "PILOT_FINALIST_S_LARS":
                    inner_cv_fold_scores["PILOT_finalist_S_LARS"] = {str(hp_dict): [] for hp_dict in pilot_grid_list_of_dicts}
                    time_per_fold["PILOT_finalist_S_LARS"] = 0.0
                elif model_name_upper == "PILOT_FINALIST_S_PREFIX":
                    inner_cv_fold_scores["PILOT_finalist_S_prefix"] = {str(hp_dict): [] for hp_dict in pilot_grid_list_of_dicts}
                    time_per_fold["PILOT_finalist_S_prefix"] = 0.0
                elif model_name_upper == "PILOT_FINALIST_D_LARS":
                    inner_cv_fold_scores["PILOT_finalist_D_LARS"] = {str(hp_dict): [] for hp_dict in pilot_grid_list_of_dicts}
                    time_per_fold["PILOT_finalist_D_LARS"] = 0.0
                elif model_name_upper == "PILOT_FINALIST_D_PREFIX":
                    inner_cv_fold_scores["PILOT_finalist_D_prefix"] = {str(hp_dict): [] for hp_dict in pilot_grid_list_of_dicts}
                    time_per_fold["PILOT_finalist_D_prefix"] = 0.0
                elif model_name_upper == "PILOT_PER_FEATURE_LARS":
                    inner_cv_fold_scores["PILOT_per_feature_LARS"] = {str(hp_dict): [] for hp_dict in pilot_grid_list_of_dicts}
                    time_per_fold["PILOT_per_feature_LARS"] = 0.0
                elif model_name_upper == "PILOT_PER_FEATURE_PREFIX":
                    inner_cv_fold_scores["PILOT_per_feature_prefix"] = {str(hp_dict): [] for hp_dict in pilot_grid_list_of_dicts}
                    time_per_fold["PILOT_per_feature_prefix"] = 0.0
                elif model_name_upper == "PILOT_FULL_MULTI_LARS":
                    inner_cv_fold_scores["PILOT_full_multi_LARS"] = {str(hp_dict): [] for hp_dict in pilot_grid_list_of_dicts}
                    time_per_fold["PILOT_full_multi_LARS"] = 0.0
                elif model_name_upper == "PILOT_FULL_MULTI_PREFIX":
                    inner_cv_fold_scores["PILOT_full_multi_prefix"] = {str(hp_dict): [] for hp_dict in pilot_grid_list_of_dicts}
                    time_per_fold["PILOT_full_multi_prefix"] = 0.0
                elif model_name_upper == "PILOT_FINALIST_S_LARS_DF":
                    time_per_fold["PILOT_finalist_S_LARS_df"] = 0.0
                elif model_name_upper == "PILOT_FINALIST_S_PREFIX_DF":
                    time_per_fold["PILOT_finalist_S_prefix_df"] = 0.0
                elif model_name_upper == "PILOT_FINALIST_D_LARS_DF":
                    time_per_fold["PILOT_finalist_D_LARS_df"] = 0.0
                elif model_name_upper == "PILOT_FINALIST_D_PREFIX_DF":
                    time_per_fold["PILOT_finalist_D_prefix_df"] = 0.0
                elif model_name_upper == "PILOT_PER_FEATURE_LARS_DF":
                    time_per_fold["PILOT_per_feature_LARS_df"] = 0.0
                elif model_name_upper == "PILOT_PER_FEATURE_PREFIX_DF":
                    time_per_fold["PILOT_per_feature_prefix_df"] = 0.0
                elif model_name_upper == "PILOT_FULL_MULTI_LARS_DF":
                    time_per_fold["PILOT_full_multi_LARS_df"] = 0.0
                elif model_name_upper == "PILOT_FULL_MULTI_PREFIX_DF":
                    time_per_fold["PILOT_full_multi_prefix_df"] = 0.0
                elif model_name_upper == "PILOT_DF":
                    time_per_fold["PILOT_df"] = 0.0

            # Start the loop for the inner folds
            for k, (inner_train, inner_val) in enumerate(cv_tuning.split(train_dataset.X, train_dataset.y), start=1):
                inner_train_dataset = train_dataset.subset(inner_train)
                inner_val_dataset = train_dataset.subset(inner_val)
                
                # --- CART (tuning) ---
                if "CART" in selected_models_set: 
                    print_with_timestamp(f"\t\tCART\t{k}")
                    t_begin = time.time()
                    for cart_hp_dict in cart_grid_list_of_dicts:
                       
                        r = fit_cart(train_dataset=inner_train_dataset, test_dataset=inner_val_dataset, **cart_hp_dict)
                        inner_cv_fold_scores["CART"][str(cart_hp_dict)].append(r.mse)
                    
                    t_end = time.time()
                    time_per_fold["CART"] += t_end - t_begin
                

                # --- PILOT (tuning) ---
                if "PILOT" in selected_models_set: 
                    print_with_timestamp(f"\t\tCPILOT\t{k}")
                    t_begin = time.time()
                    for pilot_hp_dict in pilot_grid_list_of_dicts:
                        
                        md = pilot_hp_dict["max_depth"]
                        ms = pilot_hp_dict["min_sample_split"]
                        ml = pilot_hp_dict["min_sample_leaf"]
                        
                        
                        r = fit_cpilot(
                            train_dataset=inner_train_dataset,
                            test_dataset=inner_val_dataset,
                            max_depth=md,
                            min_sample_split=ms,
                            min_sample_leaf=ml,
                        )
                              
                        inner_cv_fold_scores["PILOT"][str(pilot_hp_dict)].append(r.mse)
                        
                    t_end = time.time()
                    time_per_fold["PILOT"] += t_end - t_begin
    
                # --- RF (tuning) ---
                if "RF" in selected_models_set: 
                    print_with_timestamp(f"\t\tRF\t{k}")
                    t_begin = time.time()
                    for rf_hp_dict in rf_grid_list_of_dicts:
                        
                        nt = rf_hp_dict["n_estimators"]
                        md = rf_hp_dict["max_depth"]
                        mf = rf_hp_dict["max_features"]
                        
                        r = fit_random_forest(
                            train_dataset=inner_train_dataset,
                            test_dataset=inner_val_dataset,
                            n_estimators=nt,
                            max_depth=md,
                            max_features=mf,
                        )
                        
                        inner_cv_fold_scores["RF"][str(rf_hp_dict)].append(r.mse)
              
                    t_end = time.time()
                    time_per_fold["RF"] += t_end - t_begin
    
                # --- RAFFLE (tuning) ---
                if "RAFFLE" in selected_models_set: 
                    print_with_timestamp(f"\t\tRAFFLE\t{k}")
                    t_begin = time.time()
                    for raffle_hp_dict in raffle_grid_list_of_dicts:
                        
                        nt = raffle_hp_dict["n_estimators"]
                        md = raffle_hp_dict["max_depth"]
                        mf = raffle_hp_dict["max_features"]
                        df = raffle_hp_dict["df_settings"]
    
                        r = fit_cpilot_forest(
                            train_dataset=inner_train_dataset,
                            test_dataset=inner_val_dataset,
                            n_estimators=nt,
                            max_depth=md,
                            n_features_node=mf,
                            min_sample_leaf=1,
                            min_sample_alpha=2,
                            min_sample_fit=2,
                            df_settings=df,
                            max_pivot=10000,
                        )
                        
                        inner_cv_fold_scores["RAFFLE"][str(raffle_hp_dict)].append(r.mse)
                        
                    t_end = time.time()
                    time_per_fold["RAFFLE"] += t_end - t_begin
    
                # --- XGB (tuning) ---
                if "XGB" in selected_models_set: 
                    print_with_timestamp(f"\t\tXGB\t{k}")
                    t_begin = time.time()
                    for xgb_hp_dict in xgb_grid_list_of_dicts:
                        
                        md = xgb_hp_dict["max_depth"]
                        mf = xgb_hp_dict["max_features"]
                        nt = xgb_hp_dict["n_estimators"]
        
                        r = fit_xgboost(
                            train_dataset=inner_train_dataset,
                            test_dataset=inner_val_dataset,
                            max_depth=md,
                            max_node_features=mf,
                            n_estimators=nt,
                        )
                        
                        inner_cv_fold_scores["XGB"][str(xgb_hp_dict)].append(r.mse)
                        
                    t_end = time.time()
                    time_per_fold["XGB"] += t_end - t_begin
                
                # --- Linear models (tuning) ---
                if "RIDGE" in selected_models_set or "LASSO" in selected_models_set:
                    print_with_timestamp(f"\t\tLin\t{k}")
                    for alpha in alphagrid:

                        # Ridge
                        if "RIDGE" in selected_models_set:
                            t_begin = time.time()
                            r_ridge = fit_ridge(
                                train_dataset=inner_train_dataset,
                                test_dataset=inner_val_dataset,
                                alpha=alpha,
                            )
                            inner_cv_fold_scores["Ridge"][alpha].append(r_ridge.mse)
                            t_end = time.time()
                            time_per_fold["Ridge"] += t_end - t_begin

                        # Lasso
                        if "LASSO" in selected_models_set:
                            t_begin = time.time()
                            r_lasso = fit_lasso(
                                train_dataset=inner_train_dataset,
                                test_dataset=inner_val_dataset,
                                alpha=alpha,
                            )
                            inner_cv_fold_scores["Lasso"][alpha].append(r_lasso.mse)
                            t_end = time.time()
                            time_per_fold["Lasso"] += t_end - t_begin
                    
                # --- Linear_pilot_ensemble models (tuning) ---
                if "RIDGE_PILOT_ENSEMBLE" in selected_models_set or "LASSO_PILOT_ENSEMBLE" in selected_models_set:
                    print_with_timestamp(f"\t\tLin_ens\t{k}")
                    for lin_pilot_ensemble_hp_dict in lin_pilot_ensemble_grid_list_of_dicts:
                        md = lin_pilot_ensemble_hp_dict["max_depth"]
                        ms = lin_pilot_ensemble_hp_dict["min_sample_split"]
                        ml = lin_pilot_ensemble_hp_dict["min_sample_leaf"]
                        alpha = lin_pilot_ensemble_hp_dict["alpha"]

                        # Ridge ensemble
                        if "RIDGE_PILOT_ENSEMBLE" in selected_models_set:
                            t_begin = time.time()
                            r_ridge_ens = fit_ridge_pilot_ensemble(
                                train_dataset=inner_train_dataset,
                                test_dataset=inner_val_dataset,
                                max_depth=md,
                                min_sample_split=ms,
                                min_sample_leaf=ml,
                                alpha=alpha,
                            )
                            
                            inner_cv_fold_scores["Ridge_pilot_ensemble"][str(lin_pilot_ensemble_hp_dict)].append(r_ridge_ens.mse)
                            t_end = time.time()
                            time_per_fold["Ridge_pilot_ensemble"] += t_end - t_begin

                        # Lasso ensemble
                        if "LASSO_PILOT_ENSEMBLE" in selected_models_set:
                            t_begin = time.time()
                            r_lasso_ens = fit_lasso_pilot_ensemble(
                                train_dataset=inner_train_dataset,
                                test_dataset=inner_val_dataset,
                                max_depth=md,
                                min_sample_split=ms,
                                min_sample_leaf=ml,
                                alpha=alpha,
                            )
                            
                            inner_cv_fold_scores["Lasso_pilot_ensemble"][str(lin_pilot_ensemble_hp_dict)].append(r_lasso_ens.mse)
                            t_end = time.time()
                            time_per_fold["Lasso_pilot_ensemble"] += t_end - t_begin
                            
                # --- PILOT_NLFS (tuning) ---
                if "PILOT_NLFS_PREFIX" in selected_models_set or "PILOT_NLFS_LARS" in selected_models_set or \
                "PILOT_NLFS_PREFIX_TUNING" in selected_models_set or "PILOT_NLFS_LARS_TUNING" in selected_models_set or \
                "PILOT_NLFS_FALLBACK" in selected_models_set or "PILOT_NLFS_FALLBACK_TUNING" in selected_models_set:
                    print_with_timestamp(f"\t\tNLFS\t{k}")
                    for pilot_hp_dict in pilot_grid_list_of_dicts:
                            
                        md = pilot_hp_dict["max_depth"]
                        ms = pilot_hp_dict["min_sample_split"]
                        ml = pilot_hp_dict["min_sample_leaf"]

                        # PILOT_NLFS_PREFIX
                        if "PILOT_NLFS_PREFIX" in selected_models_set or "PILOT_NLFS_PREFIX_TUNING" in selected_models_set:
                            t_begin = time.time()
                            r_prefix = fit_pilot_nlfs(
                                train_dataset=inner_train_dataset,
                                test_dataset=inner_val_dataset,
                                max_depth=md,
                                min_sample_split=ms,
                                min_sample_leaf=ml,
                                alpha=0.005, 
                                nlfs_lars=False
                            )
                            
                            if "PILOT_NLFS_PREFIX" in selected_models_set:    
                                inner_cv_fold_scores["PILOT_NLFS_prefix"][str(pilot_hp_dict)].append(r_prefix.mse)
                            elif  "PILOT_NLFS_PREFIX_TUNING" in selected_models_set:
                                inner_cv_fold_scores["PILOT_NLFS_prefix_tuning"][str(pilot_hp_dict)].append(r_prefix.mse)
                                
                            t_end = time.time()
                            
                            if "PILOT_NLFS_PREFIX" in selected_models_set:
                                time_per_fold["PILOT_NLFS_prefix"] += t_end - t_begin
                            if "PILOT_NLFS_PREFIX_TUNING" in selected_models_set:
                                time_per_fold["PILOT_NLFS_prefix_tuning"] += t_end - t_begin

                        # PILOT_NLFS_LARS
                        if "PILOT_NLFS_LARS" in selected_models_set or "PILOT_NLFS_LARS_TUNING" in selected_models_set:
                            t_begin = time.time()
                            r_lars = fit_pilot_nlfs(
                                train_dataset=inner_train_dataset,
                                test_dataset=inner_val_dataset,
                                max_depth=md,
                                min_sample_split=ms,
                                min_sample_leaf=ml,
                                alpha=0.05, 
                                nlfs_lars=True
                            )
                            
                            if "PILOT_NLFS_LARS" in selected_models_set:
                                inner_cv_fold_scores["PILOT_NLFS_LARS"][str(pilot_hp_dict)].append(r_lars.mse)
                            elif "PILOT_NLFS_LARS_TUNING" in selected_models_set:
                                inner_cv_fold_scores["PILOT_NLFS_LARS_tuning"][str(pilot_hp_dict)].append(r_lars.mse)
                                
                            t_end = time.time()
                            
                            if "PILOT_NLFS_LARS" in selected_models_set:
                                time_per_fold["PILOT_NLFS_LARS"] += t_end - t_begin
                            
                            if "PILOT_NLFS_LARS_TUNING" in selected_models_set:
                                time_per_fold["PILOT_NLFS_LARS_tuning"] += t_end - t_begin

                        # PILOT_NLFS_FALLBACK
                        if "PILOT_NLFS_FALLBACK" in selected_models_set or "PILOT_NLFS_FALLBACK_TUNING" in selected_models_set:
                            t_begin = time.time()
                            r_fallback = fit_pilot_nlfs(
                                train_dataset=inner_train_dataset,
                                test_dataset=inner_val_dataset,
                                max_depth=md,
                                min_sample_split=ms,
                                min_sample_leaf=ml,
                                alpha=0.05, 
                                nlfs_lars=False,
                                only_fallback=True
                            )
                            
                            if "PILOT_NLFS_FALLBACK" in selected_models_set:
                                inner_cv_fold_scores["PILOT_NLFS_fallback"][str(pilot_hp_dict)].append(r_fallback.mse)
                            elif "PILOT_NLFS_FALLBACK_TUNING" in selected_models_set:
                                inner_cv_fold_scores["PILOT_NLFS_fallback_tuning"][str(pilot_hp_dict)].append(r_fallback.mse)
                                
                            t_end = time.time()
                            
                            if "PILOT_NLFS_FALLBACK" in selected_models_set:
                                time_per_fold["PILOT_NLFS_fallback"] += t_end - t_begin
                            
                            if "PILOT_NLFS_FALLBACK_TUNING" in selected_models_set:
                                time_per_fold["PILOT_NLFS_fallback_tuning"] += t_end - t_begin
                
                # --- PILOT_FINALIST_S (tuning) ---
                if "PILOT_FINALIST_S_LARS" in selected_models_set or "PILOT_FINALIST_S_PREFIX" in selected_models_set:
                    print_with_timestamp(f"\t\Finalist_S\t{k}")
                    for pilot_hp_dict in pilot_grid_list_of_dicts:
                        
                        md = pilot_hp_dict["max_depth"]
                        ms = pilot_hp_dict["min_sample_split"]
                        ml = pilot_hp_dict["min_sample_leaf"]

                        # PILOT_FINALIST_S_LARS
                        if "PILOT_FINALIST_S_LARS" in selected_models_set: 
                            t_begin = time.time()
                            r = fit_pilot_multi(
                                train_dataset=inner_train_dataset,
                                test_dataset=inner_val_dataset,
                                max_depth=md,
                                min_sample_split=ms,
                                min_sample_leaf=ml,
                                alpha=0.05,
                                multi_lars=True,
                                finalist_s=True,
                                finalist_d=False,
                                per_feature=False,
                                full_multi=False
                            )
                              
                            inner_cv_fold_scores["PILOT_finalist_S_LARS"][str(pilot_hp_dict)].append(r.mse)
                            t_end = time.time()
                            time_per_fold["PILOT_finalist_S_LARS"] += t_end - t_begin 
                        # PILOT_FINALIST_S_PREFIX
                        if "PILOT_FINALIST_S_PREFIX" in selected_models_set:
                            t_begin = time.time()
                            r = fit_pilot_multi(
                                train_dataset=inner_train_dataset,
                                test_dataset=inner_val_dataset,
                                max_depth=md,
                                min_sample_split=ms,
                                min_sample_leaf=ml,
                                alpha=0.01,
                                multi_lars=False,
                                finalist_s=True,
                                finalist_d=False,
                                per_feature=False,
                                full_multi=False
                            )
                              
                            inner_cv_fold_scores["PILOT_finalist_S_prefix"][str(pilot_hp_dict)].append(r.mse)
                            t_end = time.time()
                            time_per_fold["PILOT_finalist_S_prefix"] += t_end - t_begin
                
                # --- PILOT_FINALIST_D (tuning) ---
                if "PILOT_FINALIST_D_LARS" in selected_models_set or "PILOT_FINALIST_D_PREFIX" in selected_models_set:
                    print_with_timestamp(f"\t\Finalist_D\t{k}")
                    for pilot_hp_dict in pilot_grid_list_of_dicts:
                        
                        md = pilot_hp_dict["max_depth"]
                        ms = pilot_hp_dict["min_sample_split"]
                        ml = pilot_hp_dict["min_sample_leaf"]

                        # PILOT_FINALIST_D_LARS
                        if "PILOT_FINALIST_D_LARS" in selected_models_set: 
                            t_begin = time.time()
                            r = fit_pilot_multi(
                                train_dataset=inner_train_dataset,
                                test_dataset=inner_val_dataset,
                                max_depth=md,
                                min_sample_split=ms,
                                min_sample_leaf=ml,
                                alpha=0.05,
                                multi_lars=True,
                                finalist_s=False,
                                finalist_d=True,
                                per_feature=False,
                                full_multi=False
                            )
                              
                            inner_cv_fold_scores["PILOT_finalist_D_LARS"][str(pilot_hp_dict)].append(r.mse)
                            t_end = time.time()
                            time_per_fold["PILOT_finalist_D_LARS"] += t_end - t_begin 

                        # PILOT_FINALIST_D_PREFIX
                        if "PILOT_FINALIST_D_PREFIX" in selected_models_set:
                            t_begin = time.time()
                            
                            r = fit_pilot_multi(
                                train_dataset=inner_train_dataset,
                                test_dataset=inner_val_dataset,
                                max_depth=md,
                                min_sample_split=ms,
                                min_sample_leaf=ml,
                                alpha=0.01,
                                multi_lars=False,
                                finalist_s=False,
                                finalist_d=True,
                                per_feature=False,
                                full_multi=False
                            )
                              
                            inner_cv_fold_scores["PILOT_finalist_D_prefix"][str(pilot_hp_dict)].append(r.mse)
                            t_end = time.time()
                            time_per_fold["PILOT_finalist_D_prefix"] += t_end - t_begin                     
                            
                # --- PILOT_per_feature (tuning) ---
                if "PILOT_PER_FEATURE_LARS" in selected_models_set or "PILOT_PER_FEATURE_PREFIX" in selected_models_set:
                    print_with_timestamp(f"\t\Per_feature\t{k}")
                    for pilot_hp_dict in pilot_grid_list_of_dicts:
                        
                        md = pilot_hp_dict["max_depth"]
                        ms = pilot_hp_dict["min_sample_split"]
                        ml = pilot_hp_dict["min_sample_leaf"]

                        # PILOT_PER_FEATURE_LARS
                        if "PILOT_PER_FEATURE_LARS" in selected_models_set:
                            t_begin = time.time()
                            r = fit_pilot_multi(
                                train_dataset=inner_train_dataset,
                                test_dataset=inner_val_dataset,
                                max_depth=md,
                                min_sample_split=ms,
                                min_sample_leaf=ml,
                                alpha=0.05,
                                multi_lars=True,
                                finalist_s=False,
                                finalist_d=False,
                                per_feature=True,
                                full_multi=False
                            )
                              
                            inner_cv_fold_scores["PILOT_per_feature_LARS"][str(pilot_hp_dict)].append(r.mse)
                            t_end = time.time()
                            time_per_fold["PILOT_per_feature_LARS"] += t_end - t_begin 

                        # PILOT_PER_FEATURE_PREFIX
                        if "PILOT_PER_FEATURE_PREFIX" in selected_models_set:
                            t_begin = time.time()
                            r = fit_pilot_multi(
                                train_dataset=inner_train_dataset,
                                test_dataset=inner_val_dataset,
                                max_depth=md,
                                min_sample_split=ms,
                                min_sample_leaf=ml,
                                alpha=0.01,
                                multi_lars=False,
                                finalist_s=False,
                                finalist_d=False,
                                per_feature=True,
                                full_multi=False
                            )
                              
                            inner_cv_fold_scores["PILOT_per_feature_prefix"][str(pilot_hp_dict)].append(r.mse)
                            t_end = time.time()
                            time_per_fold["PILOT_per_feature_prefix"] += t_end - t_begin 
                            
                # --- PILOT_full_multi (tuning) ---
                if "PILOT_FULL_MULTI_LARS" in selected_models_set or "PILOT_FULL_MULTI_PREFIX" in selected_models_set:
                    print_with_timestamp(f"\t\Full_multi\t{k}")
                    for pilot_hp_dict in pilot_grid_list_of_dicts:
                        
                        md = pilot_hp_dict["max_depth"]
                        ms = pilot_hp_dict["min_sample_split"]
                        ml = pilot_hp_dict["min_sample_leaf"]

                        # PILOT_FULL_MULTI_LARS
                        if "PILOT_FULL_MULTI_LARS" in selected_models_set: 
                            t_begin = time.time()
                            r = fit_pilot_multi(
                                train_dataset=inner_train_dataset,
                                test_dataset=inner_val_dataset,
                                max_depth=md,
                                min_sample_split=ms,
                                min_sample_leaf=ml,
                                alpha=0.05,
                                multi_lars=True,
                                finalist_s=False,
                                finalist_d=False,
                                per_feature=False,
                                full_multi=True
                            )
                              
                            inner_cv_fold_scores["PILOT_full_multi_LARS"][str(pilot_hp_dict)].append(r.mse)
                            t_end = time.time()
                            time_per_fold["PILOT_full_multi_LARS"] += t_end - t_begin 

                        # PILOT_FULL_MULTI_PREFIX
                        if "PILOT_FULL_MULTI_PREFIX" in selected_models_set:
                            t_begin = time.time()
                            r = fit_pilot_multi(
                                train_dataset=inner_train_dataset,
                                test_dataset=inner_val_dataset,
                                max_depth=md,
                                min_sample_split=ms,
                                min_sample_leaf=ml,
                                alpha=0.01,
                                multi_lars=False,
                                finalist_s=False,
                                finalist_d=False,
                                per_feature=False,
                                full_multi=True
                            )
                              
                            inner_cv_fold_scores["PILOT_full_multi_prefix"][str(pilot_hp_dict)].append(r.mse)
                            t_end = time.time()
                            time_per_fold["PILOT_full_multi_prefix"] += t_end - t_begin 

            # ----- Retrain with best hyperparam + prediction -----
            print_with_timestamp("Start with retraining + predictions")
            
            # --- CART (prediction) ---
            if "CART" in selected_models_set:
                t_begin = time.time()
                hp_scores_for_model = inner_cv_fold_scores.get("CART")

                # Get the best hyperparameters
                avg_scores_for_hps = {}
                for hp_key_str, scores_from_inner_folds_list in hp_scores_for_model.items():
                    valid_scores = [s for s in scores_from_inner_folds_list if np.isfinite(s)] # Filter out inf
                    if valid_scores:
                        avg_scores_for_hps[hp_key_str] = np.mean(valid_scores)
                    else:
                        avg_scores_for_hps[hp_key_str] = float('inf')
                        
                best_cart_hp_final_dict = None # Will store the actual best HP dictionary
                min_avg_mse_cart = float('inf')
                
                if avg_scores_for_hps and any(np.isfinite(s) for s in avg_scores_for_hps.values()):
                    best_hp_key_str = min(avg_scores_for_hps, key=avg_scores_for_hps.get)
                    min_avg_mse_cart = avg_scores_for_hps[best_hp_key_str]
                    
                    for original_hp_dict in cart_grid_list_of_dicts: 
                        if str(original_hp_dict) == best_hp_key_str:
                            best_cart_hp_final_dict = original_hp_dict
                            break
                
                else:
                    error_message = (
                        f"Outer Fold {i} for dataset {repo_id}: "
                        f"No valid hyperparameter combination found for CART"
                        f"after inner CV. All hyperparameter combinations resulted in non-finite "
                        f"average scores."
                    )
                    print_with_timestamp(error_message) # Log it first
                    raise ValueError(error_message)

                # Make the final prediction
                r_pred = fit_cart(train_dataset=train_dataset, test_dataset=test_dataset, **best_cart_hp_final_dict)

                t_end = time.time()
                time_per_fold["CART"] += t_end - t_begin
                
                results.append(
                    dict(**dataset.summary(), fold=i, model="CART", **r_pred.asdict(), time_per_fold=time_per_fold["CART"])
                )
            
            # --- PILOT (prediction) ---
            if "PILOT" in selected_models_set:
                t_begin = time.time()
                hp_scores_for_model = inner_cv_fold_scores.get("PILOT")

                # Get the best hyperparameters
                avg_scores_for_hps = {}
                for hp_key_str, scores_from_inner_folds_list in hp_scores_for_model.items():
                    valid_scores = [s for s in scores_from_inner_folds_list if np.isfinite(s)] # Filter out inf
                    if valid_scores:
                        avg_scores_for_hps[hp_key_str] = np.mean(valid_scores)
                    else:
                        avg_scores_for_hps[hp_key_str] = float('inf')
                        
                best_pilot_hp_final_dict = None # Will store the actual best HP dictionary
                min_avg_mse_pilot = float('inf')
                
                if avg_scores_for_hps and any(np.isfinite(s) for s in avg_scores_for_hps.values()):
                    best_hp_key_str = min(avg_scores_for_hps, key=avg_scores_for_hps.get)
                    min_avg_mse_pilot = avg_scores_for_hps[best_hp_key_str]
                    
                    for original_hp_dict in pilot_grid_list_of_dicts: 
                        if str(original_hp_dict) == best_hp_key_str:
                            best_pilot_hp_final_dict = original_hp_dict
                            break
                
                else:
                    error_message = (
                        f"Outer Fold {i} for dataset {repo_id}: "
                        f"No valid hyperparameter combination found for PILOT"
                        f"after inner CV. All hyperparameter combinations resulted in non-finite "
                        f"average scores."
                    )
                    print_with_timestamp(error_message) # Log it first
                    raise ValueError(error_message)

                # Chosen hyperparameters after tuning
                best_md = best_pilot_hp_final_dict["max_depth"]
                best_ms = best_pilot_hp_final_dict["min_sample_split"]
                best_ml = best_pilot_hp_final_dict["min_sample_leaf"]

                # Construct the unique filename
                model_name_for_file = "PILOT"
                vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                # Make the final prediction
                r_pred = fit_cpilot(
                            train_dataset=train_dataset,
                            test_dataset=test_dataset,
                            max_depth=best_md,
                            min_sample_split=best_ms,
                            min_sample_leaf=best_ml,
                            compute_importances=True,
                            print_tree=True,
                            visualize_tree=True,
                            figsize= (20, 12),
                            feature_names=list(train_dataset.X_label_encoded.columns),
                            file_name_importance=str(imp_filename),
                            filename=str(vis_filename)
                )
                
                t_end = time.time()
                time_per_fold["PILOT"] += t_end - t_begin
                
                results.append(
                    dict(**dataset.summary(), 
                    fold=i, 
                    model="PILOT", 
                    **r_pred.asdict(), 
                    max_depth=best_md, min_sample_split=best_ms, min_sample_leaf=best_ml, 
                    time_per_fold=time_per_fold["PILOT"])
                )
            
            # --- RF (prediction) ---
            if "RF" in selected_models_set:
                t_begin = time.time()
                hp_scores_for_model = inner_cv_fold_scores.get("RF")

                # Get the best hyperparameters
                avg_scores_for_hps = {}
                for hp_key_str, scores_from_inner_folds_list in hp_scores_for_model.items():
                    valid_scores = [s for s in scores_from_inner_folds_list if np.isfinite(s)] # Filter out inf
                    if valid_scores:
                        avg_scores_for_hps[hp_key_str] = np.mean(valid_scores)
                    else:
                        avg_scores_for_hps[hp_key_str] = float('inf')
                        
                best_rf_hp_final_dict = None # Will store the actual best HP dictionary
                min_avg_mse_rf = float('inf')
                
                if avg_scores_for_hps and any(np.isfinite(s) for s in avg_scores_for_hps.values()):
                    best_hp_key_str = min(avg_scores_for_hps, key=avg_scores_for_hps.get)
                    min_avg_mse_rf = avg_scores_for_hps[best_hp_key_str]
                    
                    for original_hp_dict in rf_grid_list_of_dicts: 
                        if str(original_hp_dict) == best_hp_key_str:
                            best_rf_hp_final_dict = original_hp_dict
                            break
                
                else:
                    error_message = (
                        f"Outer Fold {i} for dataset {repo_id}: "
                        f"No valid hyperparameter combination found for RF"
                        f"after inner CV. All hyperparameter combinations resulted in non-finite "
                        f"average scores."
                    )
                    print_with_timestamp(error_message) # Log it first
                    raise ValueError(error_message)

                # Chosen hyperparameters after tuning
                best_nt = best_rf_hp_final_dict["n_estimators"]
                best_md = best_rf_hp_final_dict["max_depth"]
                best_mf = best_rf_hp_final_dict["max_features"]

                # Make the final prediction
                r_pred = fit_random_forest(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    n_estimators=best_nt,
                    max_depth=best_md,
                    max_features=best_mf,
                )
                        
                t_end = time.time()
                time_per_fold["RF"] += t_end - t_begin
                
                results.append(
                    dict(**dataset.summary(), 
                    fold=i, 
                    model="RF", 
                    **r_pred.asdict(), 
                    max_depth=best_md, max_features=best_mf, n_estimators=best_nt,  
                    time_per_fold=time_per_fold["RF"])
                )

            # --- RAFFLE (prediction) ---
            if "RAFFLE" in selected_models_set:
                t_begin = time.time()
                hp_scores_for_model = inner_cv_fold_scores.get("RAFFLE")

                # Get the best hyperparameters
                avg_scores_for_hps = {}
                for hp_key_str, scores_from_inner_folds_list in hp_scores_for_model.items():
                    valid_scores = [s for s in scores_from_inner_folds_list if np.isfinite(s)] # Filter out inf
                    if valid_scores:
                        avg_scores_for_hps[hp_key_str] = np.mean(valid_scores)
                    else:
                        avg_scores_for_hps[hp_key_str] = float('inf')
                        
                best_raffle_hp_final_dict = None # Will store the actual best HP dictionary
                min_avg_mse_raffle = float('inf')
                
                if avg_scores_for_hps and any(np.isfinite(s) for s in avg_scores_for_hps.values()):
                    best_hp_key_str = min(avg_scores_for_hps, key=avg_scores_for_hps.get)
                    min_avg_mse_raffle = avg_scores_for_hps[best_hp_key_str]
                    
                    for original_hp_dict in raffle_grid_list_of_dicts: 
                        if str(original_hp_dict) == best_hp_key_str:
                            best_raffle_hp_final_dict = original_hp_dict
                            break
                
                else:
                    error_message = (
                        f"Outer Fold {i} for dataset {repo_id}: "
                        f"No valid hyperparameter combination found for RAFFLE"
                        f"after inner CV. All hyperparameter combinations resulted in non-finite "
                        f"average scores."
                    )
                    print_with_timestamp(error_message) # Log it first
                    raise ValueError(error_message)

                # Chosen hyperparameters after tuning
                best_nt = best_raffle_hp_final_dict["n_estimators"]
                best_md = best_raffle_hp_final_dict["max_depth"]
                best_mf = best_raffle_hp_final_dict["max_features"]
                best_df = best_raffle_hp_final_dict["df_settings"]
                best_alpha = best_raffle_hp_final_dict["alpha"]

                # Make the final prediction
                r_pred = fit_cpilot_forest(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    n_estimators=best_nt,
                    max_depth=best_md,
                    n_features_node=best_mf,
                    min_sample_leaf=1,
                    min_sample_alpha=2,
                    min_sample_fit=2,
                    df_settings=best_df,
                    
                    max_pivot=10000,
                )
                        
                t_end = time.time()
                time_per_fold["RAFFLE"] += t_end - t_begin
                
                results.append(
                    dict(**dataset.summary(), 
                    fold=i, 
                    model="RAFFLE", 
                    **r_pred.asdict(), 
                    max_depth=best_md, max_features=best_mf, n_estimators=best_nt, df_settings=best_df, alpha_cpf=best_alpha, 
                    time_per_fold=time_per_fold["RAFFLE"])
                )

            # --- XGB (prediction) ---
            if "XGB" in selected_models_set:
                t_begin = time.time()
                hp_scores_for_model = inner_cv_fold_scores.get("XGB")

                # Get the best hyperparameters
                avg_scores_for_hps = {}
                for hp_key_str, scores_from_inner_folds_list in hp_scores_for_model.items():
                    valid_scores = [s for s in scores_from_inner_folds_list if np.isfinite(s)] # Filter out inf
                    if valid_scores:
                        avg_scores_for_hps[hp_key_str] = np.mean(valid_scores)
                    else:
                        avg_scores_for_hps[hp_key_str] = float('inf')
                        
                best_xgb_hp_final_dict = None # Will store the actual best HP dictionary
                min_avg_mse_xgb = float('inf')
                
                if avg_scores_for_hps and any(np.isfinite(s) for s in avg_scores_for_hps.values()):
                    best_hp_key_str = min(avg_scores_for_hps, key=avg_scores_for_hps.get)
                    min_avg_mse_xgb = avg_scores_for_hps[best_hp_key_str]
                    
                    for original_hp_dict in xgb_grid_list_of_dicts: 
                        if str(original_hp_dict) == best_hp_key_str:
                            best_xgb_hp_final_dict = original_hp_dict
                            break
                
                else:
                    error_message = (
                        f"Outer Fold {i} for dataset {repo_id}: "
                        f"No valid hyperparameter combination found for XGB"
                        f"after inner CV. All hyperparameter combinations resulted in non-finite "
                        f"average scores."
                    )
                    print_with_timestamp(error_message) # Log it first
                    raise ValueError(error_message)

                # Chosen hyperparameters after tuning
                best_md = best_xgb_hp_final_dict["max_depth"]
                best_mf = best_xgb_hp_final_dict["max_features"]
                best_nt = best_xgb_hp_final_dict["n_estimators"]

                # Make the final prediction
                r_pred = fit_xgboost(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    max_depth=best_md,
                    max_node_features=best_mf,
                    n_estimators=best_nt,
                )
                        
                t_end = time.time()
                time_per_fold["XGB"] += t_end - t_begin
                
                results.append(
                    dict(**dataset.summary(), 
                    fold=i, 
                    model="XGB", 
                    **r_pred.asdict(), 
                    max_depth=best_md, max_features=best_mf, n_estimators=best_nt, 
                    time_per_fold=time_per_fold["XGB"])
                )

            # --- Ridge (prediction) ---
            if "RIDGE" in selected_models_set:
                t_begin = time.time()
                hp_scores_for_model = inner_cv_fold_scores.get("Ridge")

                # Get the best hyperparameters
                avg_mse_per_alpha_ridge = {}
                for alpha_value, scores_list in hp_scores_for_model.items():
                    valid_scores = [s for s in scores_list if np.isfinite(s)]
                    if valid_scores:
                        avg_mse_per_alpha_ridge[alpha_value] = np.mean(valid_scores)
                    else:
                        avg_mse_per_alpha_ridge[alpha_value] = float('inf')
                        
                best_ridge_alpha = None
                min_avg_mse_ridge = float('inf')
                
                if avg_mse_per_alpha_ridge and any(np.isfinite(s) for s in avg_mse_per_alpha_ridge.values()):
                   
                    best_ridge_alpha = min(avg_mse_per_alpha_ridge, key=avg_mse_per_alpha_ridge.get)
                    min_avg_mse_ridge = avg_mse_per_alpha_ridge[best_ridge_alpha]
                else:
                    error_message = (
                        f"Outer Fold {i} for dataset {repo_id}: "
                        f"No valid hyperparameter combination found for Ridge"
                        f"after inner CV. All hyperparameter combinations resulted in non-finite "
                        f"average scores."
                    )
                    print_with_timestamp(error_message) # Log it first
                    raise ValueError(error_message)

                # Make the final prediction
                r_pred = fit_ridge(
                    train_dataset=train_dataset, 
                    test_dataset=test_dataset,   
                    alpha=best_ridge_alpha
                )
                            
                t_end = time.time()
                time_per_fold["Ridge"] += t_end - t_begin
                
                results.append( 
                        dict(**dataset.summary(), 
                             fold=i, 
                             model="Ridge", 
                             **r_pred.asdict(),
                             alpha=best_ridge_alpha,
                             time_per_fold=time_per_fold["Ridge"]
                            )
                )

            # --- Lasso (prediction) ---
            if "LASSO" in selected_models_set:
                t_begin = time.time()
                hp_scores_for_model = inner_cv_fold_scores.get("Lasso")

                # Get the best hyperparameters
                avg_mse_per_alpha_lasso = {}
                for alpha_value, scores_list in hp_scores_for_model.items():
                    valid_scores = [s for s in scores_list if np.isfinite(s)]
                    if valid_scores:
                        avg_mse_per_alpha_lasso[alpha_value] = np.mean(valid_scores)
                    else:
                        avg_mse_per_alpha_lasso[alpha_value] = float('inf')
                        
                best_lasso_alpha = None
                min_avg_mse_lasso = float('inf')
                
                if avg_mse_per_alpha_lasso and any(np.isfinite(s) for s in avg_mse_per_alpha_lasso.values()):
                   
                    best_lasso_alpha = min(avg_mse_per_alpha_lasso, key=avg_mse_per_alpha_lasso.get)
                    min_avg_mse_lasso = avg_mse_per_alpha_lasso[best_lasso_alpha]
                else:
                    error_message = (
                        f"Outer Fold {i} for dataset {repo_id}: "
                        f"No valid hyperparameter combination found for Lasso"
                        f"after inner CV. All hyperparameter combinations resulted in non-finite "
                        f"average scores."
                    )
                    print_with_timestamp(error_message) # Log it first
                    raise ValueError(error_message)

                # Make the final prediction
                r_pred = fit_lasso(
                    train_dataset=train_dataset, 
                    test_dataset=test_dataset,   
                    alpha=best_lasso_alpha
                )
                            
    
                t_end = time.time()
                time_per_fold["Lasso"] += t_end - t_begin
                
                results.append( 
                        dict(**dataset.summary(), 
                             fold=i, 
                             model="Lasso",
                             **r_pred.asdict(), 
                             alpha=best_lasso_alpha,
                             time_per_fold=time_per_fold["Lasso"]
                            )
                )
                
            # --- Ridge_pilot_ensemble (prediction) ---
            if "RIDGE_PILOT_ENSEMBLE" in selected_models_set:
                t_begin = time.time()
                hp_scores_for_model = inner_cv_fold_scores.get("Ridge_pilot_ensemble")

                # Get the best hyperparameters
                avg_scores_for_hps = {}
                for hp_key_str, scores_from_inner_folds_list in hp_scores_for_model.items():
                    valid_scores = [s for s in scores_from_inner_folds_list if np.isfinite(s)] # Filter out inf
                    if valid_scores:
                        avg_scores_for_hps[hp_key_str] = np.mean(valid_scores)
                    else:
                        avg_scores_for_hps[hp_key_str] = float('inf')
                        
                best_ridge_pilot_ensemble_hp_final_dict = None # Will store the actual best HP dictionary
                min_avg_mse_ridge_pilot_ensemble = float('inf')
                
                if avg_scores_for_hps and any(np.isfinite(s) for s in avg_scores_for_hps.values()):
                    best_hp_key_str = min(avg_scores_for_hps, key=avg_scores_for_hps.get)
                    min_avg_mse_ridge_pilot_ensemble = avg_scores_for_hps[best_hp_key_str]
                    
                    for original_hp_dict in lin_pilot_ensemble_grid_list_of_dicts: 
                        if str(original_hp_dict) == best_hp_key_str:
                            best_ridge_pilot_ensemble_hp_final_dict = original_hp_dict
                            break
                
                else:
                    error_message = (
                        f"Outer Fold {i} for dataset {repo_id}: "
                        f"No valid hyperparameter combination found for Ridge_pilot_ensemble"
                        f"after inner CV. All hyperparameter combinations resulted in non-finite "
                        f"average scores."
                    )
                    print_with_timestamp(error_message) # Log it first
                    raise ValueError(error_message)

                # Chosen hyperparameters after tuning
                best_md = best_ridge_pilot_ensemble_hp_final_dict["max_depth"]
                best_ms = best_ridge_pilot_ensemble_hp_final_dict["min_sample_split"]
                best_ml = best_ridge_pilot_ensemble_hp_final_dict["min_sample_leaf"]
                best_alpha = best_ridge_pilot_ensemble_hp_final_dict["alpha"]

                # Construct the unique filename
                model_name_for_file = "Ridge_PILOT_ensemble"
                vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                # Make the final prediction
                r_pred = fit_ridge_pilot_ensemble(
                            train_dataset=train_dataset,
                            test_dataset=test_dataset,
                            max_depth=best_md,
                            min_sample_split=best_ms,
                            min_sample_leaf=best_ml,
                            alpha=best_alpha,
                            compute_importances=False,
                            print_tree=False,
                            visualize_tree=False,
                            figsize=(25, 15),
                            feature_names=list(train_dataset.X_label_encoded.columns),
                            file_name_importance=str(imp_filename),
                            filename=str(vis_filename)
                )
                
                t_end = time.time()
                time_per_fold["Ridge_pilot_ensemble"] += t_end - t_begin
                
                results.append(
                    dict(**dataset.summary(), 
                    fold=i, 
                    model="Ridge_pilot_ensemble", 
                    **r_pred.asdict(), 
                    max_depth=best_md, min_sample_split=best_ms, min_sample_leaf=best_ml, alpha=best_alpha,
                    time_per_fold=time_per_fold["Ridge_pilot_ensemble"])
                )
            
            # --- Lasso_pilot_ensemble (prediction) ---
            if "LASSO_PILOT_ENSEMBLE" in selected_models_set:
                t_begin = time.time()
                hp_scores_for_model = inner_cv_fold_scores.get("Lasso_pilot_ensemble")

                # Get the best hyperparameters
                avg_scores_for_hps = {}
                for hp_key_str, scores_from_inner_folds_list in hp_scores_for_model.items():
                    valid_scores = [s for s in scores_from_inner_folds_list if np.isfinite(s)] # Filter out inf
                    if valid_scores:
                        avg_scores_for_hps[hp_key_str] = np.mean(valid_scores)
                    else:
                        avg_scores_for_hps[hp_key_str] = float('inf')
                        
                best_lasso_pilot_ensemble_hp_final_dict = None # Will store the actual best HP dictionary
                min_avg_mse_lasso_pilot_ensemble = float('inf')
                
                if avg_scores_for_hps and any(np.isfinite(s) for s in avg_scores_for_hps.values()):
                    best_hp_key_str = min(avg_scores_for_hps, key=avg_scores_for_hps.get)
                    min_avg_mse_lasso_pilot_ensemble = avg_scores_for_hps[best_hp_key_str]
                    
                    for original_hp_dict in lin_pilot_ensemble_grid_list_of_dicts: 
                        if str(original_hp_dict) == best_hp_key_str:
                            best_lasso_pilot_ensemble_hp_final_dict = original_hp_dict
                            break
                
                else:
                    error_message = (
                        f"Outer Fold {i} for dataset {repo_id}: "
                        f"No valid hyperparameter combination found for Lasso_pilot_ensemble"
                        f"after inner CV. All hyperparameter combinations resulted in non-finite "
                        f"average scores."
                    )
                    print_with_timestamp(error_message) # Log it first
                    raise ValueError(error_message)

                # Chosen hyperparameters after tuning
                best_md = best_lasso_pilot_ensemble_hp_final_dict["max_depth"]
                best_ms = best_lasso_pilot_ensemble_hp_final_dict["min_sample_split"]
                best_ml = best_lasso_pilot_ensemble_hp_final_dict["min_sample_leaf"]
                best_alpha = best_lasso_pilot_ensemble_hp_final_dict["alpha"]

                # Construct the unique filename
                model_name_for_file = "Lasso_PILOT_ensemble"
                vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                # Make the final prediction
                r_pred = fit_lasso_pilot_ensemble(
                            train_dataset=train_dataset,
                            test_dataset=test_dataset,
                            max_depth=best_md,
                            min_sample_split=best_ms,
                            min_sample_leaf=best_ml,
                            alpha=best_alpha,
                            compute_importances=True,
                            print_tree=True,
                            visualize_tree=True,
                            figsize=(25, 15),
                            feature_names=list(train_dataset.X_label_encoded.columns),
                            file_name_importance=str(imp_filename),
                            filename=str(vis_filename)
                )
                
                t_end = time.time()
                time_per_fold["Lasso_pilot_ensemble"] += t_end - t_begin
                
                results.append(
                    dict(**dataset.summary(), 
                    fold=i, 
                    model="Lasso_pilot_ensemble", 
                    **r_pred.asdict(), 
                    max_depth=best_md, min_sample_split=best_ms, min_sample_leaf=best_ml, alpha=best_alpha,
                    time_per_fold=time_per_fold["Lasso_pilot_ensemble"])
                )
            
            # --- PILOT_NLFS_prefix (prediction) ---
            if "PILOT_NLFS_PREFIX" in selected_models_set or "PILOT_NLFS_PREFIX_TUNING" in selected_models_set:
                t_begin = time.time()
                
                if "PILOT_NLFS_PREFIX" in selected_models_set:
                    hp_scores_for_model = inner_cv_fold_scores.get("PILOT_NLFS_prefix")
                elif "PILOT_NLFS_PREFIX_TUNING" in selected_models_set:
                    hp_scores_for_model = inner_cv_fold_scores.get("PILOT_NLFS_prefix_tuning")

                # Get the best hyperparameters
                avg_scores_for_hps = {}
                for hp_key_str, scores_from_inner_folds_list in hp_scores_for_model.items():
                    valid_scores = [s for s in scores_from_inner_folds_list if np.isfinite(s)] # Filter out inf
                    if valid_scores:
                        avg_scores_for_hps[hp_key_str] = np.mean(valid_scores)
                    else:
                        avg_scores_for_hps[hp_key_str] = float('inf')
                        
                best_nlfs_prefix_hp_final_dict = None # Will store the actual best HP dictionary
                min_avg_mse_nlfs_prefix = float('inf')
                
                if avg_scores_for_hps and any(np.isfinite(s) for s in avg_scores_for_hps.values()):
                    best_hp_key_str = min(avg_scores_for_hps, key=avg_scores_for_hps.get)
                    min_avg_mse_nlfs_prefix = avg_scores_for_hps[best_hp_key_str]
                    
                    for original_hp_dict in pilot_grid_list_of_dicts: 
                        if str(original_hp_dict) == best_hp_key_str:
                            best_nlfs_prefix_hp_final_dict = original_hp_dict
                            break
                
                else:
                    error_message = (
                        f"Outer Fold {i} for dataset {repo_id}: "
                        f"No valid hyperparameter combination found for PILOT_NLFS_prefix"
                        f"after inner CV. All hyperparameter combinations resulted in non-finite "
                        f"average scores."
                    )
                    print_with_timestamp(error_message) # Log it first
                    raise ValueError(error_message)

                # Chosen hyperparameters after tuning
                best_md = best_nlfs_prefix_hp_final_dict["max_depth"]
                best_ms = best_nlfs_prefix_hp_final_dict["min_sample_split"]
                best_ml = best_nlfs_prefix_hp_final_dict["min_sample_leaf"]
                
                t_middle_1 = time.time()
                        
                if "PILOT_NLFS_PREFIX" in selected_models_set:
                    # Construct the unique filename
                    model_name_for_file = "PILOT_NLFS_PREFIX"
                    vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                    imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                    # Make the final prediction
                    r_pred_prefix = fit_pilot_nlfs(
                                    train_dataset=train_dataset,
                                    test_dataset=test_dataset,
                                    max_depth=best_md,
                                    min_sample_split=best_ms,
                                    min_sample_leaf=best_ml,
                                    alpha=0.005,
                                    nlfs_lars=False,
                                    compute_importances=False,
                                    print_tree=False,
                                    visualize_tree=False,
                                    figsize=(40, 25),
                                    feature_names=list(train_dataset.X_label_encoded.columns),
                                    file_name_importance=str(imp_filename),
                                    filename=str(vis_filename)
                    )
                
                    t_end = time.time()
                    time_per_fold["PILOT_NLFS_prefix"] += t_end - t_begin
                
                    results.append(
                        dict(**dataset.summary(), 
                        fold=i, 
                        model="PILOT_NLFS_prefix", 
                        **r_pred_prefix.asdict(), 
                        max_depth=best_md, min_sample_split=best_ms, min_sample_leaf=best_ml, alpha=0.005,
                        time_per_fold=time_per_fold["PILOT_NLFS_prefix"])
                    )
                
                t_middle_2 = time.time()
                
                if "PILOT_NLFS_PREFIX_TUNING" in selected_models_set:
                    # Construct the unique filename
                    model_name_for_file = "PILOT_NLFS_PREFIX_TUNING"
                    vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                    imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                    # Make the final prediction
                    r_pred_prefix = fit_cpilot(
                                    train_dataset=train_dataset,
                                    test_dataset=test_dataset,
                                    max_depth=best_md,
                                    min_sample_split=best_ms,
                                    min_sample_leaf=best_ml,
                                    compute_importances=False,
                                    print_tree=False,
                                    visualize_tree=False,
                                    figsize=(40, 25),
                                    feature_names=list(train_dataset.X_label_encoded.columns),
                                    file_name_importance=str(imp_filename),
                                    filename=str(vis_filename)
                    )
                
                    t_end = time.time()
                    time_per_fold["PILOT_NLFS_prefix_tuning"] += (t_end - t_begin) - (t_middle_2 - t_middle_1)
                
                    results.append(
                        dict(**dataset.summary(), 
                        fold=i, 
                        model="PILOT_NLFS_prefix_tuning", 
                        **r_pred_prefix.asdict(), 
                        max_depth=best_md, min_sample_split=best_ms, min_sample_leaf=best_ml, alpha=0.005,
                        time_per_fold=time_per_fold["PILOT_NLFS_prefix_tuning"])
                    )
                           
            # --- PILOT_NLFS_LARS (prediction) ---
            if "PILOT_NLFS_LARS" in selected_models_set or "PILOT_NLFS_LARS_TUNING" in selected_models_set:
                t_begin = time.time()
                
                if "PILOT_NLFS_LARS" in selected_models_set:
                    hp_scores_for_model = inner_cv_fold_scores.get("PILOT_NLFS_LARS")
                elif "PILOT_NLFS_LARS_TUNING" in selected_models_set:
                    hp_scores_for_model = inner_cv_fold_scores.get("PILOT_NLFS_LARS_tuning")

                # Get the best hyperparameters
                avg_scores_for_hps = {}
                for hp_key_str, scores_from_inner_folds_list in hp_scores_for_model.items():
                    valid_scores = [s for s in scores_from_inner_folds_list if np.isfinite(s)] # Filter out inf
                    if valid_scores:
                        avg_scores_for_hps[hp_key_str] = np.mean(valid_scores)
                    else:
                        avg_scores_for_hps[hp_key_str] = float('inf')
                        
                best_nlfs_lars_hp_final_dict = None # Will store the actual best HP dictionary
                min_avg_mse_nlfs_lars = float('inf')
                
                if avg_scores_for_hps and any(np.isfinite(s) for s in avg_scores_for_hps.values()):
                    best_hp_key_str = min(avg_scores_for_hps, key=avg_scores_for_hps.get)
                    min_avg_mse_nlfs_lars = avg_scores_for_hps[best_hp_key_str]
                    
                    for original_hp_dict in pilot_grid_list_of_dicts: 
                        if str(original_hp_dict) == best_hp_key_str:
                            best_nlfs_lars_hp_final_dict = original_hp_dict
                            break
                
                else:
                    error_message = (
                        f"Outer Fold {i} for dataset {repo_id}: "
                        f"No valid hyperparameter combination found for PILOT_NLFS_LARS"
                        f"after inner CV. All hyperparameter combinations resulted in non-finite "
                        f"average scores."
                    )
                    print_with_timestamp(error_message) # Log it first
                    raise ValueError(error_message)

                # Chosen hyperparameters after tuning
                best_md = best_nlfs_lars_hp_final_dict["max_depth"]
                best_ms = best_nlfs_lars_hp_final_dict["min_sample_split"]
                best_ml = best_nlfs_lars_hp_final_dict["min_sample_leaf"]
                
                t_middle_1 = time.time()
                        
                if "PILOT_NLFS_LARS" in selected_models_set:
                    # Construct the unique filename
                    model_name_for_file = "PILOT_NLFS_LARS"
                    vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                    imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                    # Make the final prediction
                    r_pred_lars = fit_pilot_nlfs(
                                    train_dataset=train_dataset,
                                    test_dataset=test_dataset,
                                    max_depth=best_md,
                                    min_sample_split=best_ms,
                                    min_sample_leaf=best_ml,
                                    alpha=0.05,
                                    nlfs_lars=True,
                                    compute_importances=True,
                                    print_tree=True,
                                    visualize_tree=True,
                                    figsize=(25,15),
                                    feature_names=list(train_dataset.X_label_encoded.columns),
                                    file_name_importance=str(imp_filename),
                                    filename=str(vis_filename)
                    )
                    
                    t_end = time.time()
                    time_per_fold["PILOT_NLFS_LARS"] += t_end - t_begin
                    
                    results.append(
                        dict(**dataset.summary(), 
                        fold=i, 
                        model="PILOT_NLFS_LARS", 
                        **r_pred_lars.asdict(), 
                        max_depth=best_md, min_sample_split=best_ms, min_sample_leaf=best_ml,
                        time_per_fold=time_per_fold["PILOT_NLFS_LARS"])
                    )
                
                t_middle_2 = time.time()
                
                if "PILOT_NLFS_LARS_TUNING" in selected_models_set:
                    # Construct the unique filename
                    model_name_for_file = "PILOT_NLFS_LARS_TUNING"
                    vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                    imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                    # Make the final prediction
                    r_pred_lars = fit_cpilot(
                                    train_dataset=train_dataset,
                                    test_dataset=test_dataset,
                                    max_depth=best_md,
                                    min_sample_split=best_ms,
                                    min_sample_leaf=best_ml,
                                    compute_importances=False,
                                    print_tree=False,
                                    visualize_tree=False,
                                    figsize=(40, 25),
                                    feature_names=list(train_dataset.X_label_encoded.columns),
                                    file_name_importance=str(imp_filename),
                                    filename=str(vis_filename)
                    )
                    
                    t_end = time.time()
                    time_per_fold["PILOT_NLFS_LARS_tuning"] += (t_end - t_begin) - (t_middle_2 - t_middle_1)
                    
                    results.append(
                        dict(**dataset.summary(), 
                        fold=i, 
                        model="PILOT_NLFS_LARS_tuning", 
                        **r_pred_lars.asdict(), 
                        max_depth=best_md, min_sample_split=best_ms, min_sample_leaf=best_ml,
                        time_per_fold=time_per_fold["PILOT_NLFS_LARS_tuning"])
                    )
                    
            # --- PILOT_NLFS_fallback (prediction) ---
            if "PILOT_NLFS_FALLBACK" in selected_models_set or "PILOT_NLFS_FALLBACK_TUNING" in selected_models_set:
                t_begin = time.time()
                
                if "PILOT_NLFS_FALLBACK" in selected_models_set:
                    hp_scores_for_model = inner_cv_fold_scores.get("PILOT_NLFS_fallback")
                elif "PILOT_NLFS_FALLBACK_TUNING" in selected_models_set:
                    hp_scores_for_model = inner_cv_fold_scores.get("PILOT_NLFS_fallback_tuning")

                # Get the best hyperparameters
                avg_scores_for_hps = {}
                for hp_key_str, scores_from_inner_folds_list in hp_scores_for_model.items():
                    valid_scores = [s for s in scores_from_inner_folds_list if np.isfinite(s)] # Filter out inf
                    if valid_scores:
                        avg_scores_for_hps[hp_key_str] = np.mean(valid_scores)
                    else:
                        avg_scores_for_hps[hp_key_str] = float('inf')
                        
                best_nlfs_fallback_hp_final_dict = None # Will store the actual best HP dictionary
                min_avg_mse_nlfs_fallback = float('inf')
                
                if avg_scores_for_hps and any(np.isfinite(s) for s in avg_scores_for_hps.values()):
                    best_hp_key_str = min(avg_scores_for_hps, key=avg_scores_for_hps.get)
                    min_avg_mse_nlfs_fallback = avg_scores_for_hps[best_hp_key_str]
                    
                    for original_hp_dict in pilot_grid_list_of_dicts: 
                        if str(original_hp_dict) == best_hp_key_str:
                            best_nlfs_fallback_hp_final_dict = original_hp_dict
                            break
                
                else:
                    error_message = (
                        f"Outer Fold {i} for dataset {repo_id}: "
                        f"No valid hyperparameter combination found for PILOT_NLFS_FALLBACK"
                        f"after inner CV. All hyperparameter combinations resulted in non-finite "
                        f"average scores."
                    )
                    print_with_timestamp(error_message) # Log it first
                    raise ValueError(error_message)

                # Chosen hyperparameters after tuning
                best_md = best_nlfs_fallback_hp_final_dict["max_depth"]
                best_ms = best_nlfs_fallback_hp_final_dict["min_sample_split"]
                best_ml = best_nlfs_fallback_hp_final_dict["min_sample_leaf"]
                
                t_middle_1 = time.time()
                        
                if "PILOT_NLFS_FALLBACK" in selected_models_set:
                    # Construct the unique filename
                    model_name_for_file = "PILOT_NLFS_FALLBACK"
                    vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                    imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                    # Make the final prediction
                    r_pred_fallback = fit_pilot_nlfs(
                                    train_dataset=train_dataset,
                                    test_dataset=test_dataset,
                                    max_depth=best_md,
                                    min_sample_split=best_ms,
                                    min_sample_leaf=best_ml,
                                    alpha=0.05,
                                    nlfs_lars=False,
                                    only_fallback=True,
                                    compute_importances=False,
                                    print_tree=False,
                                    visualize_tree=False,
                                    figsize=(40, 25),
                                    feature_names=list(train_dataset.X_label_encoded.columns),
                                    file_name_importance=str(imp_filename),
                                    filename=str(vis_filename)
                    )
                    
                    t_end = time.time()
                    time_per_fold["PILOT_NLFS_fallback"] += t_end - t_begin
                    
                    results.append(
                        dict(**dataset.summary(), 
                        fold=i, 
                        model="PILOT_NLFS_fallback", 
                        **r_pred_fallback.asdict(), 
                        max_depth=best_md, min_sample_split=best_ms, min_sample_leaf=best_ml,
                        time_per_fold=time_per_fold["PILOT_NLFS_fallback"])
                    )
                
                t_middle_2 = time.time()
                
                if "PILOT_NLFS_FALLBACK_TUNING" in selected_models_set:
                    # Construct the unique filename
                    model_name_for_file = "PILOT_NLFS_FALLBACK_TUNING"
                    vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                    imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                    # Make the final prediction
                    r_pred_fallback = fit_cpilot(
                                    train_dataset=train_dataset,
                                    test_dataset=test_dataset,
                                    max_depth=best_md,
                                    min_sample_split=best_ms,
                                    min_sample_leaf=best_ml,
                                    compute_importances=False,
                                    print_tree=False,
                                    visualize_tree=False,
                                    figsize=(40, 25),
                                    feature_names=list(train_dataset.X_label_encoded.columns),
                                    file_name_importance=str(imp_filename),
                                    filename=str(vis_filename)
                    )
                    
                    t_end = time.time()
                    time_per_fold["PILOT_NLFS_fallback_tuning"] += (t_end - t_begin) - (t_middle_2 - t_middle_1)
                    
                    results.append(
                        dict(**dataset.summary(), 
                        fold=i, 
                        model="PILOT_NLFS_fallback_tuning", 
                        **r_pred_fallback.asdict(), 
                        max_depth=best_md, min_sample_split=best_ms, min_sample_leaf=best_ml,
                        time_per_fold=time_per_fold["PILOT_NLFS_fallback_tuning"])
                    )
                    
            # --- PILOT_finalist_S_LARS (prediction) ---
            if "PILOT_FINALIST_S_LARS" in selected_models_set:
                t_begin = time.time()
                hp_scores_for_model = inner_cv_fold_scores.get("PILOT_finalist_S_LARS")

                # Get the best hyperparameters
                avg_scores_for_hps = {}
                for hp_key_str, scores_from_inner_folds_list in hp_scores_for_model.items():
                    valid_scores = [s for s in scores_from_inner_folds_list if np.isfinite(s)] # Filter out inf
                    if valid_scores:
                        avg_scores_for_hps[hp_key_str] = np.mean(valid_scores)
                    else:
                        avg_scores_for_hps[hp_key_str] = float('inf')
                        
                best_s_lars_hp_final_dict = None # Will store the actual best HP dictionary
                min_avg_mse_s_lars = float('inf')
                
                if avg_scores_for_hps and any(np.isfinite(s) for s in avg_scores_for_hps.values()):
                    best_hp_key_str = min(avg_scores_for_hps, key=avg_scores_for_hps.get)
                    min_avg_mse_s_lars = avg_scores_for_hps[best_hp_key_str]
                    
                    for original_hp_dict in pilot_grid_list_of_dicts: 
                        if str(original_hp_dict) == best_hp_key_str:
                            best_s_lars_hp_final_dict = original_hp_dict
                            break
                
                else:
                    error_message = (
                        f"Outer Fold {i} for dataset {repo_id}: "
                        f"No valid hyperparameter combination found for PILOT_finalist_S_LARS"
                        f"after inner CV. All hyperparameter combinations resulted in non-finite "
                        f"average scores."
                    )
                    print_with_timestamp(error_message) # Log it first
                    raise ValueError(error_message)

                # Chosen hyperparameters after tuning
                best_md = best_s_lars_hp_final_dict["max_depth"]
                best_ms = best_s_lars_hp_final_dict["min_sample_split"]
                best_ml = best_s_lars_hp_final_dict["min_sample_leaf"]

                # Construct the unique filename
                model_name_for_file = "PILOT_finalist_S_lars"
                vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                # Make the final prediction
                r_pred = fit_pilot_multi(
                            train_dataset=train_dataset,
                            test_dataset=test_dataset,
                            max_depth=best_md,
                            min_sample_split=best_ms,
                            min_sample_leaf=best_ml,
                            alpha=0.05,
                            multi_lars=True,
                            finalist_s=True,
                            finalist_d=False,
                            per_feature=False,
                            full_multi=False,
                            compute_importances=False,
                            print_tree=False,
                            visualize_tree=False,
                            figsize=(40, 25),
                            feature_names=list(train_dataset.X_label_encoded.columns),
                            file_name_importance=str(imp_filename),
                            filename=str(vis_filename)
                )
                
                t_end = time.time()
                time_per_fold["PILOT_finalist_S_LARS"] += t_end - t_begin
                
                results.append(
                    dict(**dataset.summary(), 
                    fold=i, 
                    model="PILOT_finalist_S_LARS", 
                    **r_pred.asdict(), 
                    max_depth=best_md, min_sample_split=best_ms, min_sample_leaf=best_ml, 
                    time_per_fold=time_per_fold["PILOT_finalist_S_LARS"])
                )
                     
            # --- PILOT_finalist_S_PREFIX (prediction) ---
            if "PILOT_FINALIST_S_PREFIX" in selected_models_set:
                t_begin = time.time()
                hp_scores_for_model = inner_cv_fold_scores.get("PILOT_finalist_S_prefix")

                # Get the best hyperparameters
                avg_scores_for_hps = {}
                for hp_key_str, scores_from_inner_folds_list in hp_scores_for_model.items():
                    valid_scores = [s for s in scores_from_inner_folds_list if np.isfinite(s)] # Filter out inf
                    if valid_scores:
                        avg_scores_for_hps[hp_key_str] = np.mean(valid_scores)
                    else:
                        avg_scores_for_hps[hp_key_str] = float('inf')
                        
                best_s_prefix_hp_final_dict = None # Will store the actual best HP dictionary
                min_avg_mse_s_prefix = float('inf')
                
                if avg_scores_for_hps and any(np.isfinite(s) for s in avg_scores_for_hps.values()):
                    best_hp_key_str = min(avg_scores_for_hps, key=avg_scores_for_hps.get)
                    min_avg_mse_s_prefix = avg_scores_for_hps[best_hp_key_str]
                    
                    for original_hp_dict in pilot_grid_list_of_dicts: 
                        if str(original_hp_dict) == best_hp_key_str:
                            best_s_prefix_hp_final_dict = original_hp_dict
                            break
                
                else:
                    error_message = (
                        f"Outer Fold {i} for dataset {repo_id}: "
                        f"No valid hyperparameter combination found for PILOT_finalist_S_prefix"
                        f"after inner CV. All hyperparameter combinations resulted in non-finite "
                        f"average scores."
                    )
                    print_with_timestamp(error_message) # Log it first
                    raise ValueError(error_message)

                # Chosen hyperparameters after tuning
                best_md = best_s_prefix_hp_final_dict["max_depth"]
                best_ms = best_s_prefix_hp_final_dict["min_sample_split"]
                best_ml = best_s_prefix_hp_final_dict["min_sample_leaf"]
                
                # Construct the unique filenames
                model_name_for_file = "PILOT_finalist_S_prefix"
                vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                # Make the final prediction
                r_pred = fit_pilot_multi(
                            train_dataset=train_dataset,
                            test_dataset=test_dataset,
                            max_depth=best_md,
                            min_sample_split=best_ms,
                            min_sample_leaf=best_ml,
                            alpha=0.01,
                            multi_lars=False,
                            finalist_s=True,
                            finalist_d=False,
                            per_feature=False,
                            full_multi=False,
                            compute_importances=True,
                            print_tree=True,
                            visualize_tree=True,
                            figsize=(40, 25),
                            feature_names=list(train_dataset.X_label_encoded.columns),
                            file_name_importance=str(imp_filename),
                            filename=str(vis_filename)
                )
                
                t_end = time.time()
                time_per_fold["PILOT_finalist_S_prefix"] += t_end - t_begin
                
                results.append(
                    dict(**dataset.summary(), 
                    fold=i, 
                    model="PILOT_finalist_S_prefix", 
                    **r_pred.asdict(), 
                    max_depth=best_md, min_sample_split=best_ms, min_sample_leaf=best_ml, 
                    time_per_fold=time_per_fold["PILOT_finalist_S_prefix"])
                ) 
                
            # --- PILOT_finalist_D_LARS (prediction) ---
            if "PILOT_FINALIST_D_LARS" in selected_models_set:
                t_begin = time.time()
                hp_scores_for_model = inner_cv_fold_scores.get("PILOT_finalist_D_LARS")

                # Get the best hyperparameters
                avg_scores_for_hps = {}
                for hp_key_str, scores_from_inner_folds_list in hp_scores_for_model.items():
                    valid_scores = [s for s in scores_from_inner_folds_list if np.isfinite(s)] # Filter out inf
                    if valid_scores:
                        avg_scores_for_hps[hp_key_str] = np.mean(valid_scores)
                    else:
                        avg_scores_for_hps[hp_key_str] = float('inf')
                        
                best_d_lars_hp_final_dict = None # Will store the actual best HP dictionary
                min_avg_mse_d_lars = float('inf')
                
                if avg_scores_for_hps and any(np.isfinite(s) for s in avg_scores_for_hps.values()):
                    best_hp_key_str = min(avg_scores_for_hps, key=avg_scores_for_hps.get)
                    min_avg_mse_d_lars = avg_scores_for_hps[best_hp_key_str]
                    
                    for original_hp_dict in pilot_grid_list_of_dicts: 
                        if str(original_hp_dict) == best_hp_key_str:
                            best_d_lars_hp_final_dict = original_hp_dict
                            break
                
                else:
                    error_message = (
                        f"Outer Fold {i} for dataset {repo_id}: "
                        f"No valid hyperparameter combination found for PILOT_finalist_D_LARS"
                        f"after inner CV. All hyperparameter combinations resulted in non-finite "
                        f"average scores."
                    )
                    print_with_timestamp(error_message) # Log it first
                    raise ValueError(error_message)

                # Chosen hyperparameters after tuning
                best_md = best_d_lars_hp_final_dict["max_depth"]
                best_ms = best_d_lars_hp_final_dict["min_sample_split"]
                best_ml = best_d_lars_hp_final_dict["min_sample_leaf"]

                # Construct the unique filename
                model_name_for_file = "PILOT_finalist_D_lars"
                vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                # Make the final prediction
                r_pred = fit_pilot_multi(
                            train_dataset=train_dataset,
                            test_dataset=test_dataset,
                            max_depth=best_md,
                            min_sample_split=best_ms,
                            min_sample_leaf=best_ml,
                            alpha=0.05,
                            multi_lars=True,
                            finalist_s=False,
                            finalist_d=True,
                            per_feature=False,
                            full_multi=False,
                            compute_importances=False,
                            print_tree=False,
                            visualize_tree=False,
                            figsize=(40, 25),
                            feature_names=list(train_dataset.X_label_encoded.columns),
                            file_name_importance=str(imp_filename),
                            filename=str(vis_filename)
                )
                
                t_end = time.time()
                time_per_fold["PILOT_finalist_D_LARS"] += t_end - t_begin
                
                results.append(
                    dict(**dataset.summary(), 
                    fold=i, 
                    model="PILOT_finalist_D_LARS", 
                    **r_pred.asdict(), 
                    max_depth=best_md, min_sample_split=best_ms, min_sample_leaf=best_ml, 
                    time_per_fold=time_per_fold["PILOT_finalist_D_LARS"])
                )
                     
            # --- PILOT_finalist_D_PREFIX (prediction) ---
            if "PILOT_FINALIST_D_PREFIX" in selected_models_set:
                t_begin = time.time()
                hp_scores_for_model = inner_cv_fold_scores.get("PILOT_finalist_D_prefix")

                # Get the best hyperparameters
                avg_scores_for_hps = {}
                for hp_key_str, scores_from_inner_folds_list in hp_scores_for_model.items():
                    valid_scores = [s for s in scores_from_inner_folds_list if np.isfinite(s)] # Filter out inf
                    if valid_scores:
                        avg_scores_for_hps[hp_key_str] = np.mean(valid_scores)
                    else:
                        avg_scores_for_hps[hp_key_str] = float('inf')
                        
                best_d_prefix_hp_final_dict = None # Will store the actual best HP dictionary
                min_avg_mse_d_prefix = float('inf')
                
                if avg_scores_for_hps and any(np.isfinite(s) for s in avg_scores_for_hps.values()):
                    best_hp_key_str = min(avg_scores_for_hps, key=avg_scores_for_hps.get)
                    min_avg_mse_d_prefix = avg_scores_for_hps[best_hp_key_str]
                    
                    for original_hp_dict in pilot_grid_list_of_dicts: 
                        if str(original_hp_dict) == best_hp_key_str:
                            best_d_prefix_hp_final_dict = original_hp_dict
                            break
                
                else:
                    error_message = (
                        f"Outer Fold {i} for dataset {repo_id}: "
                        f"No valid hyperparameter combination found for PILOT_finalist_D_prefix"
                        f"after inner CV. All hyperparameter combinations resulted in non-finite "
                        f"average scores."
                    )
                    print_with_timestamp(error_message) # Log it first
                    raise ValueError(error_message)

                # Chosen hyperparameters after tuning
                best_md = best_d_prefix_hp_final_dict["max_depth"]
                best_ms = best_d_prefix_hp_final_dict["min_sample_split"]
                best_ml = best_d_prefix_hp_final_dict["min_sample_leaf"]

                # Construct the unique filename
                model_name_for_file = "PILOT_finalist_D_prefix"
                vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                # Make the final prediction
                r_pred = fit_pilot_multi(
                            train_dataset=train_dataset,
                            test_dataset=test_dataset,
                            max_depth=best_md,
                            min_sample_split=best_ms,
                            min_sample_leaf=best_ml,
                            alpha=0.01,
                            multi_lars=False,
                            finalist_s=False,
                            finalist_d=True,
                            per_feature=False,
                            full_multi=False,
                            compute_importances=False,
                            print_tree=False,
                            visualize_tree=False,
                            figsize=(40, 25),
                            feature_names=list(train_dataset.X_label_encoded.columns),
                            file_name_importance=str(imp_filename),
                            filename=str(vis_filename)
                )
                
                t_end = time.time()
                time_per_fold["PILOT_finalist_D_prefix"] += t_end - t_begin
                
                results.append(
                    dict(**dataset.summary(), 
                    fold=i, 
                    model="PILOT_finalist_D_prefix", 
                    **r_pred.asdict(), 
                    max_depth=best_md, min_sample_split=best_ms, min_sample_leaf=best_ml, 
                    time_per_fold=time_per_fold["PILOT_finalist_D_prefix"])
                )
            # --- PILOT_per_feature_LARS (prediction) ---
            if "PILOT_PER_FEATURE_LARS" in selected_models_set:
                t_begin = time.time()
                hp_scores_for_model = inner_cv_fold_scores.get("PILOT_per_feature_LARS")

                # Get the best hyperparameters
                avg_scores_for_hps = {}
                for hp_key_str, scores_from_inner_folds_list in hp_scores_for_model.items():
                    valid_scores = [s for s in scores_from_inner_folds_list if np.isfinite(s)] # Filter out inf
                    if valid_scores:
                        avg_scores_for_hps[hp_key_str] = np.mean(valid_scores)
                    else:
                        avg_scores_for_hps[hp_key_str] = float('inf')
                        
                best_per_feature_lars_hp_final_dict = None # Will store the actual best HP dictionary
                min_avg_mse_per_feature_lars = float('inf')
                
                if avg_scores_for_hps and any(np.isfinite(s) for s in avg_scores_for_hps.values()):
                    best_hp_key_str = min(avg_scores_for_hps, key=avg_scores_for_hps.get)
                    min_avg_mse_per_feature_lars = avg_scores_for_hps[best_hp_key_str]
                    
                    for original_hp_dict in pilot_grid_list_of_dicts: 
                        if str(original_hp_dict) == best_hp_key_str:
                            best_per_feature_lars_hp_final_dict = original_hp_dict
                            break
                
                else:
                    error_message = (
                        f"Outer Fold {i} for dataset {repo_id}: "
                        f"No valid hyperparameter combination found for PILOT_per_feature_LARS"
                        f"after inner CV. All hyperparameter combinations resulted in non-finite "
                        f"average scores."
                    )
                    print_with_timestamp(error_message) # Log it first
                    raise ValueError(error_message)

                # Chosen hyperparameters after tuning
                best_md = best_per_feature_lars_hp_final_dict["max_depth"]
                best_ms = best_per_feature_lars_hp_final_dict["min_sample_split"]
                best_ml = best_per_feature_lars_hp_final_dict["min_sample_leaf"]

                # Construct the unique filename
                model_name_for_file = "PILOT_per_feature_lars"
                vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                # Make the final prediction
                r_pred = fit_pilot_multi(
                            train_dataset=train_dataset,
                            test_dataset=test_dataset,
                            max_depth=best_md,
                            min_sample_split=best_ms,
                            min_sample_leaf=best_ml,
                            alpha=0.05,
                            multi_lars=True,
                            finalist_s=False,
                            finalist_d=False,
                            per_feature=True,
                            full_multi=False,
                            compute_importances=False,
                            print_tree=False,
                            visualize_tree=False,
                            figsize=(40, 25),
                            feature_names=list(train_dataset.X_label_encoded.columns),
                            file_name_importance=str(imp_filename),
                            filename=str(vis_filename)
                )
                
                t_end = time.time()
                time_per_fold["PILOT_per_feature_LARS"] += t_end - t_begin
                
                results.append(
                    dict(**dataset.summary(), 
                    fold=i, 
                    model="PILOT_per_feature_LARS", 
                    **r_pred.asdict(), 
                    max_depth=best_md, min_sample_split=best_ms, min_sample_leaf=best_ml, 
                    time_per_fold=time_per_fold["PILOT_per_feature_LARS"])
                )
                     
            # --- PILOT_per_feature_PREFIX (prediction) ---
            if "PILOT_PER_FEATURE_PREFIX" in selected_models_set:
                t_begin = time.time()
                hp_scores_for_model = inner_cv_fold_scores.get("PILOT_per_feature_prefix")

                # Get the best hyperparameters
                avg_scores_for_hps = {}
                for hp_key_str, scores_from_inner_folds_list in hp_scores_for_model.items():
                    valid_scores = [s for s in scores_from_inner_folds_list if np.isfinite(s)] # Filter out inf
                    if valid_scores:
                        avg_scores_for_hps[hp_key_str] = np.mean(valid_scores)
                    else:
                        avg_scores_for_hps[hp_key_str] = float('inf')
                        
                best_per_feature_prefix_hp_final_dict = None # Will store the actual best HP dictionary
                min_avg_mse_per_feature_prefix = float('inf')
                
                if avg_scores_for_hps and any(np.isfinite(s) for s in avg_scores_for_hps.values()):
                    best_hp_key_str = min(avg_scores_for_hps, key=avg_scores_for_hps.get)
                    min_avg_mse_per_feature_prefix = avg_scores_for_hps[best_hp_key_str]
                    
                    for original_hp_dict in pilot_grid_list_of_dicts: 
                        if str(original_hp_dict) == best_hp_key_str:
                            best_per_feature_prefix_hp_final_dict = original_hp_dict
                            break
                
                else:
                    error_message = (
                        f"Outer Fold {i} for dataset {repo_id}: "
                        f"No valid hyperparameter combination found for PILOT_per_feature_prefix"
                        f"after inner CV. All hyperparameter combinations resulted in non-finite "
                        f"average scores."
                    )
                    print_with_timestamp(error_message) # Log it first
                    raise ValueError(error_message)

                # Chosen hyperparameters after tuning
                best_md = best_per_feature_prefix_hp_final_dict["max_depth"]
                best_ms = best_per_feature_prefix_hp_final_dict["min_sample_split"]
                best_ml = best_per_feature_prefix_hp_final_dict["min_sample_leaf"]

                # Construct the unique filename
                model_name_for_file = "PILOT_per_feature_prefix"
                vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                # Make the final prediction
                r_pred = fit_pilot_multi(
                            train_dataset=train_dataset,
                            test_dataset=test_dataset,
                            max_depth=best_md,
                            min_sample_split=best_ms,
                            min_sample_leaf=best_ml,
                            alpha=0.01,
                            multi_lars=False,
                            finalist_s=False,
                            finalist_d=False,
                            per_feature=True,
                            full_multi=False,
                            compute_importances=False,
                            print_tree=False,
                            visualize_tree=False,
                            figsize=(40, 25),
                            feature_names=list(train_dataset.X_label_encoded.columns),
                            file_name_importance=str(imp_filename),
                            filename=str(vis_filename)
                )
                
                t_end = time.time()
                time_per_fold["PILOT_per_feature_prefix"] += t_end - t_begin
                
                results.append(
                    dict(**dataset.summary(), 
                    fold=i, 
                    model="PILOT_per_feature_prefix", 
                    **r_pred.asdict(), 
                    max_depth=best_md, min_sample_split=best_ms, min_sample_leaf=best_ml, 
                    time_per_fold=time_per_fold["PILOT_per_feature_prefix"])
                )

            # --- PILOT_full_multi_LARS (prediction) ---
            if "PILOT_FULL_MULTI_LARS" in selected_models_set:
                t_begin = time.time()
                hp_scores_for_model = inner_cv_fold_scores.get("PILOT_full_multi_LARS")

                # Get the best hyperparameters
                avg_scores_for_hps = {}
                for hp_key_str, scores_from_inner_folds_list in hp_scores_for_model.items():
                    valid_scores = [s for s in scores_from_inner_folds_list if np.isfinite(s)] # Filter out inf
                    if valid_scores:
                        avg_scores_for_hps[hp_key_str] = np.mean(valid_scores)
                    else:
                        avg_scores_for_hps[hp_key_str] = float('inf')
                        
                best_full_multi_lars_hp_final_dict = None # Will store the actual best HP dictionary
                min_avg_mse_full_multi_lars = float('inf')
                
                if avg_scores_for_hps and any(np.isfinite(s) for s in avg_scores_for_hps.values()):
                    best_hp_key_str = min(avg_scores_for_hps, key=avg_scores_for_hps.get)
                    min_avg_mse_full_multi_lars = avg_scores_for_hps[best_hp_key_str]
                    
                    for original_hp_dict in pilot_grid_list_of_dicts: 
                        if str(original_hp_dict) == best_hp_key_str:
                            best_full_multi_lars_hp_final_dict = original_hp_dict
                            break
                
                else:
                    error_message = (
                        f"Outer Fold {i} for dataset {repo_id}: "
                        f"No valid hyperparameter combination found for PILOT_full_multi_LARS"
                        f"after inner CV. All hyperparameter combinations resulted in non-finite "
                        f"average scores."
                    )
                    print_with_timestamp(error_message) # Log it first
                    raise ValueError(error_message)
                # Chosen hyperparameters after tuning
                best_md = best_full_multi_lars_hp_final_dict["max_depth"]
                best_ms = best_full_multi_lars_hp_final_dict["min_sample_split"]
                best_ml = best_full_multi_lars_hp_final_dict["min_sample_leaf"]

                # Construct the unique filename
                model_name_for_file = "PILOT_full_multi_lars"
                vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                # Make the final prediction
                r_pred = fit_pilot_multi(
                            train_dataset=train_dataset,
                            test_dataset=test_dataset,
                            max_depth=best_md,
                            min_sample_split=best_ms,
                            min_sample_leaf=best_ml,
                            alpha=0.05,
                            multi_lars=True,
                            finalist_s=False,
                            finalist_d=False,
                            per_feature=False,
                            full_multi=True,
                            compute_importances=True,
                            print_tree=True,
                            visualize_tree=True,
                            figsize=(25,15),
                            feature_names=list(train_dataset.X_label_encoded.columns),
                            file_name_importance=str(imp_filename),
                            filename=str(vis_filename)
                )
                
                t_end = time.time()
                time_per_fold["PILOT_full_multi_LARS"] += t_end - t_begin
                
                results.append(
                    dict(**dataset.summary(), 
                    fold=i, 
                    model="PILOT_full_multi_LARS", 
                    **r_pred.asdict(), 
                    max_depth=best_md, min_sample_split=best_ms, min_sample_leaf=best_ml, 
                    time_per_fold=time_per_fold["PILOT_full_multi_LARS"])
                )
                     
            # --- PILOT_full_multi_PREFIX (prediction) ---
            if "PILOT_FULL_MULTI_PREFIX" in selected_models_set:
                t_begin = time.time()
                hp_scores_for_model = inner_cv_fold_scores.get("PILOT_full_multi_prefix")

                # Get the best hyperparameters
                avg_scores_for_hps = {}
                for hp_key_str, scores_from_inner_folds_list in hp_scores_for_model.items():
                    valid_scores = [s for s in scores_from_inner_folds_list if np.isfinite(s)] # Filter out inf
                    if valid_scores:
                        avg_scores_for_hps[hp_key_str] = np.mean(valid_scores)
                    else:
                        avg_scores_for_hps[hp_key_str] = float('inf')
                        
                best_full_multi_prefix_hp_final_dict = None # Will store the actual best HP dictionary
                min_avg_mse_full_multi_prefix = float('inf')
                
                if avg_scores_for_hps and any(np.isfinite(s) for s in avg_scores_for_hps.values()):
                    best_hp_key_str = min(avg_scores_for_hps, key=avg_scores_for_hps.get)
                    min_avg_mse_full_multi_prefix = avg_scores_for_hps[best_hp_key_str]
                    
                    for original_hp_dict in pilot_grid_list_of_dicts: 
                        if str(original_hp_dict) == best_hp_key_str:
                            best_full_multi_prefix_hp_final_dict = original_hp_dict
                            break
                
                else:
                    error_message = (
                        f"Outer Fold {i} for dataset {repo_id}: "
                        f"No valid hyperparameter combination found for PILOT_full_multi_prefix"
                        f"after inner CV. All hyperparameter combinations resulted in non-finite "
                        f"average scores."
                    )
                    print_with_timestamp(error_message) # Log it first
                    raise ValueError(error_message)

                # Chosen hyperparameters after tuning
                best_md = best_full_multi_prefix_hp_final_dict["max_depth"]
                best_ms = best_full_multi_prefix_hp_final_dict["min_sample_split"]
                best_ml = best_full_multi_prefix_hp_final_dict["min_sample_leaf"]

                # Construct the unique filename
                model_name_for_file = "PILOT_full_multi_prefix"
                vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                # Make the final prediction
                r_pred = fit_pilot_multi(
                            train_dataset=train_dataset,
                            test_dataset=test_dataset,
                            max_depth=best_md,
                            min_sample_split=best_ms,
                            min_sample_leaf=best_ml,
                            alpha=0.01,
                            multi_lars=False,
                            finalist_s=False,
                            finalist_d=False,
                            per_feature=False,
                            full_multi=True,
                            compute_importances=False,
                            print_tree=False,
                            visualize_tree=False,
                            figsize=(40, 25),
                            feature_names=list(train_dataset.X_label_encoded.columns),
                            file_name_importance=str(imp_filename),
                            filename=str(vis_filename)
                )
                
                t_end = time.time()
                time_per_fold["PILOT_full_multi_prefix"] += t_end - t_begin
                
                results.append(
                    dict(**dataset.summary(), 
                    fold=i, 
                    model="PILOT_full_multi_prefix", 
                    **r_pred.asdict(), 
                    max_depth=best_md, min_sample_split=best_ms, min_sample_leaf=best_ml, 
                    time_per_fold=time_per_fold["PILOT_full_multi_prefix"])
                )

            # --- PILOT_finalist_S_LARS_df (prediction) ---
            if "PILOT_FINALIST_S_LARS_DF" in selected_models_set:
                t_begin = time.time()

                # Construct the unique filename
                model_name_for_file = "PILOT_finalist_S_LARS_df"
                vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                # Make the final prediction
                r_pred = fit_pilot_multi(
                        train_dataset=train_dataset,
                        test_dataset=test_dataset,
                        alpha=0.01,
                        multi_lars=True,
                        finalist_s=True,
                        finalist_d=False,
                        per_feature=False,
                        full_multi=False,
                        compute_importances=False,
                        print_tree=False,
                        visualize_tree=False,
                        figsize=(40, 25),
                        feature_names=list(train_dataset.X_label_encoded.columns),
                        file_name_importance=str(imp_filename),
                        filename=str(vis_filename)
                )

                t_end = time.time()
                time_per_fold["PILOT_finalist_S_LARS_df"] += t_end - t_begin

                results.append(
                    dict(**dataset.summary(),
                         fold=i,
                         model="PILOT_finalist_S_LARS_df",
                         **r_pred.asdict(),
                         time_per_fold=time_per_fold["PILOT_finalist_S_LARS_df"])
                )

            # --- PILOT_finalist_S_PREFIX_DF  (prediction) ---
            if "PILOT_FINALIST_S_PREFIX_DF" in selected_models_set:
                t_begin = time.time()

                # Construct the unique filename
                model_name_for_file = "PILOT_finalist_S_prefix_df"
                vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                # Make the final prediction
                r_pred = fit_pilot_multi(
                        train_dataset=train_dataset,
                        test_dataset=test_dataset,
                        alpha=0.01,
                        multi_lars=False,
                        finalist_s=True,
                        finalist_d=False,
                        per_feature=False,
                        full_multi=False,
                        compute_importances=True,
                        print_tree=True,
                        visualize_tree=True,
                        figsize=(40, 25),
                        feature_names=list(train_dataset.X_label_encoded.columns),
                        file_name_importance=str(imp_filename),
                        filename=str(vis_filename)
                )

                t_end = time.time()
                time_per_fold["PILOT_finalist_S_prefix_df"] += t_end - t_begin

                results.append(
                    dict(**dataset.summary(),
                         fold=i,
                         model="PILOT_finalist_S_prefix_df",
                         **r_pred.asdict(),
                         time_per_fold=time_per_fold["PILOT_finalist_S_prefix_df"])
                )

            # --- PILOT_finalist_D_LARS_df  (prediction) ---
            if "PILOT_FINALIST_D_LARS_DF" in selected_models_set:
                t_begin = time.time()

                # Construct the unique filename
                model_name_for_file = "PILOT_finalist_D_LARS_df"
                vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                # Make the final prediction
                r_pred = fit_pilot_multi(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    alpha=0.05,
                    multi_lars=True,
                    finalist_s=False,
                    finalist_d=True,
                    per_feature=False,
                    full_multi=False,
                    compute_importances=False,
                    print_tree=False,
                    visualize_tree=False,
                    figsize=(40, 25),
                    feature_names=list(train_dataset.X_label_encoded.columns),
                    file_name_importance=str(imp_filename),
                    filename=str(vis_filename)
                )

                t_end = time.time()
                time_per_fold["PILOT_finalist_D_LARS_df"] += t_end - t_begin

                results.append(
                    dict(**dataset.summary(),
                         fold=i,
                         model="PILOT_finalist_D_LARS_df",
                         **r_pred.asdict(),
                         time_per_fold=time_per_fold["PILOT_finalist_D_LARS_df"])
                )

            # --- PILOT_finalist_D_PREFIX_df (prediction) ---
            if "PILOT_FINALIST_D_PREFIX_DF" in selected_models_set:
                t_begin = time.time()

                # Construct the unique filename
                model_name_for_file = "PILOT_finalist_D_prefix_df"
                vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                # Make the final prediction
                r_pred = fit_pilot_multi(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    alpha=0.01,
                    multi_lars=False,
                    finalist_s=False,
                    finalist_d=True,
                    per_feature=False,
                    full_multi=False,
                    compute_importances=False,
                    print_tree=False,
                    visualize_tree=False,
                    figsize=(40, 25),
                    feature_names=list(train_dataset.X_label_encoded.columns),
                    file_name_importance=str(imp_filename),
                    filename=str(vis_filename)
                )

                t_end = time.time()
                time_per_fold["PILOT_finalist_D_prefix_df"] += t_end - t_begin

                results.append(
                    dict(**dataset.summary(),
                         fold=i,
                         model="PILOT_finalist_D_prefix_df",
                         **r_pred.asdict(),
                         time_per_fold=time_per_fold["PILOT_finalist_D_prefix_df"])
                )
            # --- PILOT_per_feature_LARS_df (prediction) ---
            if "PILOT_PER_FEATURE_LARS_DF" in selected_models_set:
                t_begin = time.time()

                # Construct the unique filename
                model_name_for_file = "PILOT_per_feature_LARS_df"
                vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                # Make the final prediction
                r_pred = fit_pilot_multi(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    alpha=0.05,
                    multi_lars=True,
                    finalist_s=False,
                    finalist_d=False,
                    per_feature=True,
                    full_multi=False,
                    compute_importances=False,
                    print_tree=False,
                    visualize_tree=False,
                    figsize=(40, 25),
                    feature_names=list(train_dataset.X_label_encoded.columns),
                    file_name_importance=str(imp_filename),
                    filename=str(vis_filename)
                )

                t_end = time.time()
                time_per_fold["PILOT_per_feature_LARS_df"] += t_end - t_begin

                results.append(
                    dict(**dataset.summary(),
                         fold=i,
                         model="PILOT_per_feature_LARS_df",
                         **r_pred.asdict(),
                         time_per_fold=time_per_fold["PILOT_per_feature_LARS_df"])
                )

            # --- PILOT_per_feature_PREFIX_df (prediction) ---
            if "PILOT_PER_FEATURE_PREFIX_DF" in selected_models_set:
                t_begin = time.time()

                # Construct the unique filename
                model_name_for_file = "PILOT_per_feature_prefix_df"
                vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                # Make the final prediction
                r_pred = fit_pilot_multi(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    alpha=0.01,
                    multi_lars=False,
                    finalist_s=False,
                    finalist_d=False,
                    per_feature=True,
                    full_multi=False,
                    compute_importances=False,
                    print_tree=False,
                    visualize_tree=False,
                    figsize=(40, 25),
                    feature_names=list(train_dataset.X_label_encoded.columns),
                    file_name_importance=str(imp_filename),
                    filename=str(vis_filename)
                )

                t_end = time.time()
                time_per_fold["PILOT_per_feature_prefix_df"] += t_end - t_begin

                results.append(
                    dict(**dataset.summary(),
                         fold=i,
                         model="PILOT_per_feature_prefix_df",
                         **r_pred.asdict(),
                         time_per_fold=time_per_fold["PILOT_per_feature_prefix_df"])
                )

            # --- PILOT_full_multi_LARS_df (prediction) ---
            if "PILOT_FULL_MULTI_LARS_DF" in selected_models_set:
                t_begin = time.time()

                # Construct the unique filename
                model_name_for_file = "PILOT_full_multi_LARS_df"
                vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                # Make the final prediction
                r_pred = fit_pilot_multi(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    alpha=0.05,
                    multi_lars=True,
                    finalist_s=False,
                    finalist_d=False,
                    per_feature=False,
                    full_multi=True,
                    compute_importances=False,
                    print_tree=False,
                    visualize_tree=False,
                    figsize=(40, 25),
                    feature_names=list(train_dataset.X_label_encoded.columns),
                    file_name_importance=str(imp_filename),
                    filename=str(vis_filename)
                )

                t_end = time.time()
                time_per_fold["PILOT_full_multi_LARS_df"] += t_end - t_begin

                results.append(
                    dict(**dataset.summary(),
                         fold=i,
                         model="PILOT_full_multi_LARS_df",
                         **r_pred.asdict(),
                         time_per_fold=time_per_fold["PILOT_full_multi_LARS_df"])
                )

            # --- PILOT_full_multi_PREFIX_df (prediction) ---
            if "PILOT_FULL_MULTI_PREFIX_DF" in selected_models_set:
                t_begin = time.time()

                # Construct the unique filename
                model_name_for_file = "PILOT_full_multi_prefix_df"
                vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                # Make the final prediction
                r_pred = fit_pilot_multi(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    alpha=0.01,
                    multi_lars=False,
                    finalist_s=False,
                    finalist_d=False,
                    per_feature=False,
                    full_multi=True,
                    compute_importances=False,
                    print_tree=False,
                    visualize_tree=False,
                    figsize=(40, 25),
                    feature_names=list(train_dataset.X_label_encoded.columns),
                    file_name_importance=str(imp_filename),
                    filename=str(vis_filename)
                )

                t_end = time.time()
                time_per_fold["PILOT_full_multi_prefix_df"] += t_end - t_begin

                results.append(
                    dict(**dataset.summary(),
                         fold=i,
                         model="PILOT_full_multi_prefix_df",
                         **r_pred.asdict(),
                         time_per_fold=time_per_fold["PILOT_full_multi_prefix_df"])
                )

            # --- PILOT_DF (prediction) ---
            if "PILOT_DF" in selected_models_set:
                t_begin = time.time()

                # Construct the unique filename
                model_name_for_file = "PILOT_df"
                vis_filename = visualizations_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.png"
                imp_filename = importances_folder / f"{model_name_for_file}_{repo_id}_fold_{i}.csv"

                # Make the final prediction
                r_pred = fit_cpilot(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    compute_importances=True,
                    print_tree=True,
                    visualize_tree=True,
                    figsize=(40, 25),
                    feature_names=list(train_dataset.X_label_encoded.columns),
                    file_name_importance=str(imp_filename),
                    filename=str(vis_filename)
                )

                t_end = time.time()
                time_per_fold["PILOT_df"] += t_end - t_begin

                results.append(
                    dict(**dataset.summary(),
                         fold=i,
                         model="PILOT_df",
                         **r_pred.asdict(),
                         time_per_fold=time_per_fold["PILOT_df"])
                )

    pd.DataFrame(results).to_csv(experiment_file, index=False)

if __name__ == "__main__":
    run_benchmark()
