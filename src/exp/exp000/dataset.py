import multiprocessing as mp
import pathlib

import polars as pl
import torch
import torch.utils.data as torch_data
from typing_extensions import TypeAlias

# =============================================================================
# Dataset
# =============================================================================
TrainBatch: TypeAlias = tuple[str, torch.Tensor, torch.Tensor]
ValidBatch: TypeAlias = tuple[str, torch.Tensor, torch.Tensor]


class MyTrainDataset(torch_data.Dataset[TrainBatch]):
    def __init__(self, df: pl.DataFrame) -> None:
        super().__init__()
        self.df = df.to_pandas()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> TrainBatch:
        raise NotImplementedError


class MyValidDataset(torch_data.Dataset[ValidBatch]):
    def __init__(self, df: pl.DataFrame) -> None:
        super().__init__()
        self.df = df.to_pandas()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> ValidBatch:
        raise NotImplementedError


def init_dataloader(
    df_fp: pathlib.Path,
    train_batch_size: int,
    valid_batch_size: int,
    num_workers: int = 16,
    fold: int = 0,
    debug: bool = False,
    fulltrain: bool = False,
    prefetch_factor: int | None = 2,
    persistent_workers: bool = False,
) -> tuple[torch_data.DataLoader, torch_data.DataLoader]:
    if mp.cpu_count() < num_workers:
        num_workers = mp.cpu_count()

    if df_fp.suffix == ".csv":
        df = pl.read_csv(df_fp)
    elif df_fp.suffix == ".parquet":
        df = pl.read_parquet(df_fp)
    else:
        raise ValueError(f"Unknown file type: {df_fp}")

    df_train = df.filter(pl.col("fold") != fold)
    df_valid = df.filter(pl.col("fold") == fold)
    assert len(df_train) > 0, f"df_train is empty: {df_fp=}, {fold=}"
    assert len(df_valid) > 0, f"df_valid is empty: {df_fp=}, {fold=}"
    if debug:
        df_train = df_train.head(100)
        df_valid = df_valid.head(100)

    if fulltrain:
        df_train = df

    # --- Preprocess

    # --- Construct Datasets
    ds_train: torch_data.Dataset[TrainBatch] = MyTrainDataset(df_train)
    ds_valid: torch_data.Dataset[ValidBatch] = MyValidDataset(df_valid)
    # --- Construct DataLoaders
    dl_train = torch_data.DataLoader(
        dataset=ds_train,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    dl_valid = torch_data.DataLoader(
        dataset=ds_valid,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    return dl_train, dl_valid


def _test_dataloaders() -> None:
    from . import config

    cfg = config.Config(is_debug=True)
    dl_train, dl_valid = init_dataloader(
        df_fp=cfg.data_fp_train,
        train_batch_size=cfg.train_batch_size,
        valid_batch_size=cfg.valid_batch_size,
        num_workers=cfg.num_workers,
        fold=0,
        debug=True,
    )
    print("-- Test Train")
    train_batch: TrainBatch
    for i, train_batch in enumerate(dl_train):
        if i > 3:
            break
        _key, x, y = train_batch
        print(f"{_key=}, {x.shape=}, {y.shape=}")

    print("-- Test Valid")
    valid_batch: ValidBatch
    for i, valid_batch in enumerate(dl_valid):
        if i > 3:
            break
        _key, x, y = valid_batch
        print(f"{_key=}, {x.shape=}, {y.shape=}")


if __name__ == "__main__":
    _test_dataloaders()
