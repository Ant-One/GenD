from .. import config as C
from ..config import Config

from ..utils.files import DF40, CDFv2

experiments = {
    "df40-fs-cdf-xception": [
        Config(
            run_dir="runs/test",
            tst_files=DF40.CDF.test_fs,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/xception.pth",
        )
    ],
    "df40-fr-cdf-xception": [
        Config(
            run_dir="runs/test",
            tst_files=DF40.CDF.test_fr,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/xception.pth",
        )
    ],
    "df40-fr-cdf-xception": [
        Config(
            run_dir="runs/test",
            tst_files=DF40.CDF.test_efs,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/xception.pth",
        )
    ],
}