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

from pilot import DEFAULT_DF_SETTINGS
from benchmark_util import *


def print_with_timestamp(message):
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


@click.command()
@click.option("--experiment_name", "-e", required=True, help="Name of the experiment")
def run_benchmark(experiment_name):
    experiment_folder = OUTPUTFOLDER / experiment_name
    experiment_folder.mkdir(exist_ok=True)
    experiment_file = experiment_folder / "results.csv"
    print(f"Results will be stored in {experiment_file}")
    np.random.seed(42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

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

    for repo_id in repo_ids_to_process:
        print_with_timestamp(repo_id)
        kind, repo_id = repo_id.split("_")
        dataset = load_data(repo_id=repo_id, kind=kind)
        if dataset.n_samples > 2e5:
            print_with_timestamp(f"Skipping large dataset {repo_id}")
            continue
            
        alphagrid = _alpha_grid(
            dataset.X_oh_encoded.values,
            dataset.y.values,
            l1_ratio=1,
            fit_intercept=True,
            eps=1e-3,
            n_alphas=100,
            copy_X=False,
        )
        for i, (train, test) in enumerate(cv.split(dataset.X, dataset.y), start=1):
            print_with_timestamp(f"\tFold {i} / 5")
            print("\tRAM Used (GB):", psutil.virtual_memory()[3] / 1000000000)
            train_dataset = dataset.subset(train)
            test_dataset = dataset.subset(test)

            transformers = fit_transformers(train_dataset)

            for col, transformer in transformers.items():
                train_dataset.apply_transformer(col, transformer)
                test_dataset.apply_transformer(col, transformer)

            # CART
            print_with_timestamp("\t\tCART")
            r = fit_cart(train_dataset=train_dataset, test_dataset=test_dataset)
            results.append(
                dict(**dataset.summary(), fold=i, model="CART", **r.asdict())
            )

            print_with_timestamp("\t\tCPILOT")
            r = fit_cpilot(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                max_depth=20,
            )
            results.append(
                dict(**dataset.summary(), fold=i, model="CPILOT", **r.asdict())
            )

            # RF
            for md, mf, nt in itertools.product([6, 20, None], [0.7, 1.0], [100]):
                model_name = (
                    f"RF - max_depth = {md} - max_features = {mf} - n_estimators = {nt}"
                )
                print_with_timestamp(f"\t\t{model_name}")
                r = fit_random_forest(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    n_estimators=nt,
                    max_depth=md,
                    max_features=mf,
                )
                results.append(
                    dict(
                        **dataset.summary(),
                        fold=i,
                        model=model_name,
                        **r.asdict(),
                        max_depth=md,
                        max_features=mf,
                        n_estimators=nt,
                    )
                )

            for j, (
                (df_name, alpha, df_setting),
                max_depth,
                max_features,
                ntrees,
            ) in enumerate(
                itertools.product(
                    [
                        # ("default df", 1, DEFAULT_DF_SETTINGS),
                        # ("df alpha = 0.01", 0.01, df_setting_alpha01),
                        ("df alpha = 0.01, no blin", 0.01, df_setting_alpha01_no_blin),
                        ("df no blin", 1, df_setting_no_blin),
                        # ("df alpha = 0.5", 0.5, df_setting_alpha5),
                        ("df alpha = 0.5, no blin", 0.5, df_setting_alpha5_no_blin),
                    ],
                    [6, 20],
                    [0.7, 1.0],
                    [100],
                )
            ):
                model_name = f"CPF - {df_name} - max_depth = {max_depth} - max_node_features = {max_features} - n_estimators = {ntrees}"
                print_with_timestamp(f"\t\t{model_name}")
                r = fit_cpilot_forest(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    n_estimators=ntrees,
                    min_sample_leaf=1,
                    min_sample_alpha=2,
                    min_sample_fit=2,
                    max_depth=max_depth,
                    n_features_node=max_features,
                    df_settings=df_setting,
                    max_pivot=10000,
                )
                results.append(
                    dict(
                        **dataset.summary(),
                        fold=i,
                        model=model_name,
                        **r.asdict(),
                        df_setting=df_setting,
                        excl_blin="no_blin" in df_name,
                        alpha=alpha,
                        max_depth=max_depth,
                        max_features=max_features,
                        n_estimators=ntrees,
                    )
                )

            # XGB
            for md, mf, nt in itertools.product([6, 20], [0.7, 1.0], [100]):
                model_name = f"XGB - max_depth = {md} - max_features = {mf} - n_estimators = {nt}"
                print_with_timestamp(f"\t\t{model_name}")
                r = fit_xgboost(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    max_depth=md,
                    max_node_features=mf,
                    n_estimators=nt,
                )
                results.append(
                    dict(
                        **dataset.summary(),
                        fold=i,
                        model=model_name,
                        **r.asdict(),
                        max_depth=md,
                        max_features=mf,
                        n_estimators=nt,
                    )
                )
            # linear models
            for alpha in alphagrid:
                model_name = f"Ridge - alpha = {alpha}"
                print_with_timestamp(f"\t\t{model_name}")
                r = fit_ridge(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    alpha=alpha,
                )
                results.append(
                    dict(
                        **dataset.summary(),
                        fold=i,
                        model=model_name,
                        **r.asdict(),
                        alpha=alpha,
                    )
                )
                model_name = f"Lasso - alpha = {alpha}"
                print_with_timestamp(f"\t\t{model_name}")
                r = fit_lasso(
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    alpha=alpha,
                )
                results.append(
                    dict(
                        **dataset.summary(),
                        fold=i,
                        model=model_name,
                        **r.asdict(),
                        alpha=alpha,
                    )
                )

        pd.DataFrame(results).to_csv(experiment_file, index=False)


if __name__ == "__main__":
    run_benchmark()
