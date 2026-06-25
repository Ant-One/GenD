from ..config import Config, Backbone
from ..utils.files import DF40, CDFv2

experiments = {
    #     "df40-fs-cdf-spsl": [
    #     Config(
    #         run_dir="runs/test",
    #         backbone=Backbone.CLIP_B_16,
    #         tst_files=DF40.CDF.test_fs,
    #         batch_size=128,
    #         mini_batch_size=128,
    #         wandb=False,
    #         devices=[0],
    #         checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/clip.pth",
    #     )
    # ],
    #     "df40-fr-cdf-spsl": [
    #     Config(
    #         run_dir="runs/test",
    #         backbone=Backbone.CLIP_B_16,
    #         tst_files=DF40.CDF.test_fr,
    #         batch_size=128,
    #         mini_batch_size=128,
    #         wandb=False,
    #         devices=[0],
    #         checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/clip.pth",
    #     )
    # ],
    #     "df40-efs-cdf-spsl": [
    #     Config(
    #         run_dir="runs/test",
    #         backbone=Backbone.CLIP_B_16,
    #         tst_files=DF40.CDF.test_efs,
    #         batch_size=128,
    #         mini_batch_size=128,
    #         wandb=False,
    #         devices=[0],
    #         checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/clip.pth",
    #     )
    # ],
}