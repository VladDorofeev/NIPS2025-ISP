import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from omegaconf import open_dict

from .image_net_utils import (
    ImageNetDataset,
)
from .cifar_utils import (
    ImageDataset,
    get_image_dataset_params,
)


def prepare_df_for_federated_training(
    cfg: dict,
    directories_key: str,
):
    df = read_dataframe_from_cfg(cfg, directories_key)
    df["client"] = df["client"].apply(lambda x: x - 1)
    if "dirichlet" in cfg.dataset.data_name:
        df = create_dirichlet_df(df, cfg)

    df.reset_index(drop=True, inplace=True)

    num_classes = pd.Series(
        np.concatenate(
            df["target"].apply(lambda x: x if isinstance(x, list) else [x]).values
        )
    ).nunique()

    print(f"Num classes = {num_classes}")

    with open_dict(cfg):
        cfg.training_params.num_classes = num_classes

    print("Preprocess successfull\n")
    return df, cfg


def read_dataframe_from_cfg(
    cfg,
    directories_key="train_directories",
    mode="dataset",
):
    df = pd.DataFrame()
    for directories in cfg[mode]["data_sources"][directories_key]:
        df = pd.concat([df, pd.read_csv(directories, low_memory=False)])
    return df


def get_dataset_loader(
    df: pd.DataFrame,
    cfg,
    drop_last=True,
    mode="train",
    unsupervised=False,
    transforms=None,
):
    if "cifar" in cfg["dataset"]["data_name"]:
        # CIFAR case
        image_size, mean, std = get_image_dataset_params(cfg, df)
        dataset = ImageDataset(df, mode, image_size, mean, std)
    else:
        # ImageNet case
        dataset = ImageNetDataset(df)

    loader = DataLoader(
        dataset,
        batch_size=cfg.training_params.batch_size,
        shuffle=(mode == "train"),
        num_workers=cfg.training_params.num_workers,
        drop_last=drop_last,
    )
    assert (
        len(loader) > 0
    ), f"len(dataloader) is 0, either lower the batch size, or put drop_last=False"
    return loader


def get_stratified_subsample(df, num_samples, random_state):
    """Create a subDataFrame with `num_samples` and stratified label distribution

    Args:
        df (pd.DataFrame): origin DataFrame
        num_samples (_type_): number of samples in subDataFrame

    return: sub_df (pd.DataFrame): sub DataFrame
    """
    sub_df = pd.DataFrame()
    for target in list(df.target.value_counts().keys()):
        tmp = df[df["target"] == target]
        weight = len(tmp) / len(df)
        amount = int(weight * num_samples)
        sub_df = pd.concat(
            [
                sub_df,
                tmp.sample(
                    n=amount,
                    random_state=random_state,
                ),
            ]
        )
    # Remove all rows from df, that are now in sub_df
    df = df[~df["fpath"].isin(list(sub_df["fpath"]))]
    return df, sub_df


def dirichlet_distrubution(
    total_data_points, num_classes, num_clients, alpha, verbose, seed=42
):
    np.random.seed(seed)
    dirichet = np.random.dirichlet(alpha * np.ones(num_clients), num_classes)
    data_distr = (dirichet * total_data_points / num_classes).astype(int)
    data_distr = data_distr.transpose()

    total_assigned = data_distr.sum()
    remaining_data_points = total_data_points - total_assigned
    max_per_class = total_data_points // num_classes

    class_counts = {i: data_distr[:, i].sum() for i in range(num_classes)}

    # Distribute remaining data (because we use .astype(int))
    if remaining_data_points > 0:
        for i in range(remaining_data_points):
            for class_idx in range(num_classes):
                if class_counts[class_idx] < max_per_class:
                    client_idx = np.argmin(data_distr.sum(axis=1))
                    data_distr[client_idx, class_idx] += 1
                    class_counts[class_idx] += 1
                    break

    if verbose:
        print("Total usage data:", data_distr.sum())
        for i, x in enumerate(data_distr):
            x_str = " ".join(f"{num:>4}" for num in x)
            print(f"Client {i:>2} | {x_str} | {sum(x):>5}")

    return data_distr


def create_dirichlet_df(df, cfg):
    n_classes = df["target"].nunique()
    data_distr = dirichlet_distrubution(
        len(df),
        n_classes,
        cfg.federated_params.amount_of_clients,
        cfg.dataset.alpha,
        cfg.dataset.verbose,
        cfg.random_state,
    )

    # Drop 'client' column
    df["client"] = -1

    client_target_count = {
        i: {j: count for j, count in enumerate(row)} for i, row in enumerate(data_distr)
    }

    # Fill 'client' column
    for index, row in df.iterrows():
        target = row["target"]
        for client, counts in client_target_count.items():
            if counts[target] > 0:
                df.at[index, "client"] = client
                client_target_count[client][target] -= 1
                break

    # Check the results
    result = df.groupby(["client", "target"]).size().unstack(fill_value=0)
    for client, row in result.iterrows():
        if client != -1:
            expected = data_distr[client]
            actual = row.tolist()
            assert np.all(
                expected == actual
            ), f"Mismatch for client {client}: Expected {expected}, Actual {actual}"

    print("\nChecking: All clients have the correct distribution of targets.\n")

    return df
