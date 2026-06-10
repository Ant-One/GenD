from .. import config as C
from ..config import Config

from ..utils.files import DF40, DF40Balanced, CDFv2

experiments = {
    "df40-b-fs-ff-clip-gend": [
        Config(
            backbone=C.Backbone.CLIP_L_14,
            head=C.Head.Linear,
            unfreeze_layers=["pre_layrnorm", "layer_norm1", "layer_norm2", "post_layernorm"],
            loss=C.Loss(ce_labels=1.0),
            run_dir="runs/train",
            trn_files=DF40Balanced.FF.train_fs,
            val_files=DF40Balanced.FF.val_fs,
            tst_files=DF40.CDF.test_fs,
            batch_size=2,
            mini_batch_size=2,
            wandb=True,
            devices=[0],
        )
    ],
    #     "example-training": [
    #     Config(
    #         backbone=C.Backbone.CLIP_L_14,
    #         head=C.Head.Linear,
    #         unfreeze_layers=["pre_layrnorm", "layer_norm1", "layer_norm2", "post_layernorm"],
    #         loss=C.Loss(ce_labels=1.0),
    #         run_dir="runs/example",
    #         trn_files=[
    #             "config/datasets/FF/test/DF.txt",
    #             "config/datasets/FF/test/F2F.txt",
    #             "config/datasets/FF/test/FS.txt",
    #             "config/datasets/FF/test/NT.txt",
    #             "config/datasets/FF/test/real.txt",
    #         ],
    #         val_files=[
    #             "config/datasets/FF/test/DF.txt",
    #             "config/datasets/FF/test/F2F.txt",
    #             "config/datasets/FF/test/FS.txt",
    #             "config/datasets/FF/test/NT.txt",
    #             "config/datasets/FF/test/real.txt",
    #         ],
    #         tst_files=[
    #             "config/datasets/CDFv2/test/Celeb-real.txt",
    #             "config/datasets/CDFv2/test/Celeb-synthesis.txt",
    #             "config/datasets/CDFv2/test/YouTube-real.txt",
    #         ],
    #         batch_size=2,
    #         mini_batch_size=2,
    #         max_epochs=1,
    #         wandb=False,
    #         devices=[0],
    #     )
    # ],

}
