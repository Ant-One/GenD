from .. import config as C
from ..config import Config

from ..utils.files import Files, DF40, DF40Balanced, FSh, UADFV, DFD, DFDC, FFIW

experiments = {

        "own-pe-v1": [
        Config(
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            backbone_args=C.BackboneArgs(),
            head=C.Head.Linear,
            unfreeze_layers=["pre_layrnorm", "layer_norm1", "layer_norm2", "post_layernorm"],
            loss=C.Loss(ce_labels=1.0),
            run_dir="runs/train",
            trn_files=DF40Balanced.FF.train_val_combined_ff,
            val_files=DF40.CDF.val_cdf,
            tst_files=Files(UADFV.test + DFD.test + DFDC.test + FSh.test + FFIW.test),
            batch_size=96,
            mini_batch_size=96,
            wandb=True,
            devices=[0],
            lr_scheduler="cyclic",
            num_epochs_in_cycle=4,
            max_epochs=20,
            warmup_epochs=1,
            #early_stopping_patience=10,
            #monitor_metric="val/mAP_frame",
            #min_delta=0.002,
            #checkpoint="/home/antoine/GenD/runs/train/own-pe-v1/checkpoints/own-pe-best_mAP.ckpt"
        )
    ],

#    "own-bdf40-fs-fr-fsh-clip": [
#         Config(
#             run_dir="runs/test",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=FSh.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="/home/antoine/GenD/runs/train/biggerbatch-cycle-df40-b-fs-fr-ff-clip-gend-non-unique/checkpoints/best_mAP.ckpt",
#         )
#     ],

# "own-bdf40-fs-fr-ffiw-clip": [
#         Config(
#             run_dir="runs/test-from-gend",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=FFIW.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="/home/antoine/GenD/runs/train/biggerbatch-cycle-df40-b-fs-fr-ff-clip-gend-non-unique/checkpoints/best_mAP.ckpt",
#         )
#     ],
# "own-bdf40-fs-fr-uadv-clip": [
#         Config(
#             run_dir="runs/test-from-gend",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=UADFV.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="/home/antoine/GenD/runs/train/biggerbatch-cycle-df40-b-fs-fr-ff-clip-gend-non-unique/checkpoints/best_mAP.ckpt",
#         )
#     ],
# "own-bdf40-fs-fr-dfd-clip": [
#         Config(
#             run_dir="runs/test-from-gend",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=DFD.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="/home/antoine/GenD/runs/train/biggerbatch-cycle-df40-b-fs-fr-ff-clip-gend-non-unique/checkpoints/best_mAP.ckpt",
#         )
#     ],
# "own-bdf40-fs-fr-dfdc-clip": [
#         Config(
#             run_dir="runs/test-from-gend",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=DFDC.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="/home/antoine/GenD/runs/train/biggerbatch-cycle-df40-b-fs-fr-ff-clip-gend-non-unique/checkpoints/best_mAP.ckpt",
#         )
#     ],

#    "own-bdf40-fs-fr-fsh-clip": [
#         Config(
#             run_dir="runs/test",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=FSh.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="/home/antoine/GenD/runs/train/biggerbatch-cycle-df40-b-fs-fr-ff-clip-gend-non-unique/checkpoints/best_mAP.ckpt",
#         )
#     ],

# "gend-test-ffiw-clip": [
#         Config(
#             run_dir="runs/test-from-gend",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=FFIW.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="yermandy/GenD_CLIP_L_14",
#         )
#     ],
# "gend-test-uadfv-clip": [
#         Config(
#             run_dir="runs/test-from-gend",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=UADFV.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="yermandy/GenD_CLIP_L_14",
#         )
#     ],
# "gend-test-dfd-clip": [
#         Config(
#             run_dir="runs/test-from-gend",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=DFD.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="yermandy/GenD_CLIP_L_14",
#         )
#     ],
# "gend-test-dfdc-clip": [
#         Config(
#             run_dir="runs/test-from-gend",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=DFDC.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="yermandy/GenD_CLIP_L_14",
#         )
#     ],
    # "biggerbatch-cycle-df40-b-fs-fr-ff-clip-gend-non-unique": [
    #     Config(
    #         backbone=C.Backbone.CLIP_L_14,
    #         head=C.Head.Linear,
    #         unfreeze_layers=["pre_layrnorm", "layer_norm1", "layer_norm2", "post_layernorm"],
    #         loss=C.Loss(ce_labels=1.0),
    #         run_dir="runs/train",
    #         trn_files=DF40Balanced.FF.train_ff,
    #         val_files=DF40Balanced.FF.val_ff,
    #         tst_files=DF40.CDF.test_fs,
    #         batch_size=96,
    #         mini_batch_size=96,
    #         wandb=True,
    #         devices=[0],
    #         lr_scheduler="cyclic",
    #         num_epochs_in_cycle=4,
    #         max_epochs=20,
    #         warmup_epochs=1,
    #         early_stopping_patience=10,
    #         monitor_metric="val/mAP_frame",
    #         min_delta=0.002
    #     )
    # ],

    # "biggerbatch-cycle-df40-b-fs-ff-clip-gend-non-unique": [
    #     Config(
    #         backbone=C.Backbone.CLIP_L_14,
    #         head=C.Head.Linear,
    #         unfreeze_layers=["pre_layrnorm", "layer_norm1", "layer_norm2", "post_layernorm"],
    #         loss=C.Loss(ce_labels=1.0),
    #         run_dir="runs/train",
    #         trn_files=DF40Balanced.FF.train_fs,
    #         val_files=DF40Balanced.FF.val_fs,
    #         tst_files=DF40.CDF.test_fs,
    #         batch_size=96,
    #         mini_batch_size=96,
    #         wandb=True,
    #         devices=[0],
    #         lr_scheduler="cyclic",
    #         num_epochs_in_cycle=4,
    #         max_epochs=20,
    #         warmup_epochs=1,
    #         early_stopping_patience=10,
    #         monitor_metric="val/mAP_frame"
    #     )
    # ],
    # "own-bdf40-fr-cdf-clip": [
    #     Config(
    #         run_dir="runs/test",
    #         backbone=C.Backbone.CLIP_L_14,
    #         tst_files=DF40.CDF.test_fr,
    #         batch_size=128,
    #         mini_batch_size=128,
    #         wandb=False,
    #         devices=[0],
    #         checkpoint="/home/antoine/GenD/runs/train/biggerbatch-cycle-df40-b-fs-ff-clip-gend-non-unique/checkpoints/best_mAP.ckpt",
    #     )
    # ],
    # "clip-baseline-fs": [
    #     Config(
    #         backbone=C.Backbone.CLIP_L_14,
    #         head=C.Head.Linear,
    #         loss=C.Loss(ce_labels=1.0),
    #         tst_files=DF40.CDF.test_fs,
    #         checkpoint=""
    #     )
    # ],
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
