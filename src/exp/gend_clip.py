from .. import config as C
from ..config import Config

from ..utils.files import Files, DF40, DF40Balanced, FSh, UADFV, DFD, DFDC, FFIW, FF, CDFv3, DeepSpeak_v1_1, DeepSpeak_v2, CDFv2, KoDF, FakeAVCeleb, DFDM, PolyGlotFake, IDForge_v1

experiments = {
    "test-own-clip-v2-uadfv": [
        Config(
            run_dir="runs/test-own/", #todo changer dans "clipvX"
            backbone=C.Backbone.CLIP_L_14,
            tst_files=UADFV.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/clip-from-gend-v2/checkpoints/own-clip-best_val_auroc_frame-v2.ckpt",
            from_exp="clip-from-gend-v2"
        )
    ],
    "test-own-clip-v2-dfd": [
        Config(
            run_dir="runs/test-own",
            backbone=C.Backbone.CLIP_L_14,
            tst_files=DFD.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/clip-from-gend-v2/checkpoints/own-clip-best_val_auroc_frame-v2.ckpt",
            from_exp="clip-from-gend-v2"
        )
    ],
    "test-own-clip-v2-dfdc": [
        Config(
            run_dir="runs/test-own",
            backbone=C.Backbone.CLIP_L_14,
            tst_files=DFDC.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/clip-from-gend-v2/checkpoints/own-clip-best_val_auroc_frame-v2.ckpt",
            from_exp="clip-from-gend-v2"
        )
    ],
    "test-own-clip-v2-fsh": [
        Config(
            run_dir="runs/test-own",
            backbone=C.Backbone.CLIP_L_14,
            tst_files=FSh.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/clip-from-gend-v2/checkpoints/own-clip-best_val_auroc_frame-v2.ckpt",
            from_exp="clip-from-gend-v2"
        )
    ],
    "test-own-clip-v2-cdfv2": [
        Config(
            run_dir="runs/test-own",
            backbone=C.Backbone.CLIP_L_14,
            tst_files=CDFv2.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/clip-from-gend-v2/checkpoints/own-clip-best_val_auroc_frame-v2.ckpt",
            from_exp="clip-from-gend-v2"
        )
    ],

    "test-own-clip-v2-ffiw": [
        Config(
            run_dir="runs/test-own",
            backbone=C.Backbone.CLIP_L_14,
            tst_files=FFIW.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/clip-from-gend-v2/checkpoints/own-clip-best_val_auroc_frame-v2.ckpt",
            from_exp="clip-from-gend-v2"
        )
    ],

    "test-own-clip-v2-kodf-normal": [
        Config(
            run_dir="runs/test-own",
            backbone=C.Backbone.CLIP_L_14,
            tst_files=KoDF.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/clip-from-gend-v2/checkpoints/own-clip-best_val_auroc_frame-v2.ckpt",
            from_exp="clip-from-gend-v2"
        )
    ],
    "test-own-clip-v2-kodf-adv": [
        Config(
            run_dir="runs/test-own",
            backbone=C.Backbone.CLIP_L_14,
            tst_files=KoDF.adversarial,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/clip-from-gend-v2/checkpoints/own-clip-best_val_auroc_frame-v2.ckpt",
            from_exp="clip-from-gend-v2"
        )
    ],
    "test-own-clip-v2-kodf-all": [
        Config(
            run_dir="runs/test-own",
            backbone=C.Backbone.CLIP_L_14,
            tst_files=KoDF.test_all,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/clip-from-gend-v2/checkpoints/own-clip-best_val_auroc_frame-v2.ckpt",
            from_exp="clip-from-gend-v2"
        )
    ],
    "test-own-clip-v2-fakeavceleb": [
        Config(
            run_dir="runs/test-own",
            backbone=C.Backbone.CLIP_L_14,
            tst_files=FakeAVCeleb.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/clip-from-gend-v2/checkpoints/own-clip-best_val_auroc_frame-v2.ckpt",
            from_exp="clip-from-gend-v2"
        )
    ],
    "test-own-clip-v2-dfdm": [
        Config(
            run_dir="runs/test-own",
            backbone=C.Backbone.CLIP_L_14,
            tst_files=DFDM.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/clip-from-gend-v2/checkpoints/own-clip-best_val_auroc_frame-v2.ckpt",
            from_exp="clip-from-gend-v2"
        )
    ],

    "test-own-clip-v2-polyglotfake": [
        Config(
            run_dir="runs/test-own",
            backbone=C.Backbone.CLIP_L_14,
            tst_files=PolyGlotFake.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/clip-from-gend-v2/checkpoints/own-clip-best_val_auroc_frame-v2.ckpt",
            from_exp="clip-from-gend-v2"
        )
    ],
    "test-own-clip-v2-idforge": [
        Config(
            run_dir="runs/test-own",
            backbone=C.Backbone.CLIP_L_14,
            tst_files=IDForge_v1.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/clip-from-gend-v2/checkpoints/own-clip-best_val_auroc_frame-v2.ckpt",
            from_exp="clip-from-gend-v2"
        )
    ],
    "test-own-clip-v2-deepspeakv1": [
        Config(
            run_dir="runs/test-own",
            backbone=C.Backbone.CLIP_L_14,
            tst_files=DeepSpeak_v1_1.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/clip-from-gend-v2/checkpoints/own-clip-best_val_auroc_frame-v2.ckpt",
            from_exp="clip-from-gend-v2"
        )
    ],
    "test-own-clip-v2-deepspeakv2": [
        Config(
            run_dir="runs/test-own",
            backbone=C.Backbone.CLIP_L_14,
            tst_files=DeepSpeak_v2.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/clip-from-gend-v2/checkpoints/own-clip-best_val_auroc_frame-v2.ckpt",
            from_exp="clip-from-gend-v2"
        )
    ],
    "test-own-clip-v2-cdfv3": [
        Config(
            run_dir="runs/test-own",
            backbone=C.Backbone.CLIP_L_14,
            tst_files=CDFv3.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/clip-from-gend-v2/checkpoints/own-clip-best_val_auroc_frame-v2.ckpt",
            from_exp="clip-from-gend-v2"
        )
    ],

    # "test-yermandy-clip-FS-2": [
    #     Config(
    #         run_dir="runs/test",
    #         backbone=C.Backbone.CLIP_L_14,
    #         tst_files=DF40.CDF.val_fs,
    #         batch_size=128,
    #         mini_batch_size=128,
    #         wandb=False,
    #         devices=[0],
    #         checkpoint="yermandy/GenD_CLIP_L_14",
    #         #from_exp="clip-from-gend-v1"
    #     )
    # ],

    #    "test-clip-from-gend-v1-FS": [
    #     Config(
    #         run_dir="runs/test",
    #         backbone=C.Backbone.CLIP_L_14,
    #         tst_files=DF40.CDF.val_fs,
    #         batch_size=128,
    #         mini_batch_size=128,
    #         wandb=False,
    #         devices=[0],
    #         checkpoint="runs/train/clip-from-gend-v1/checkpoints/own-clip-best_val_auroc_frame.ckpt",
    #         from_exp="clip-from-gend-v1"
    #     )
    # ],

        "clip-from-gend-v2": [
        Config(
            backbone=C.Backbone.CLIP_L_14,
            backbone_args=C.BackboneArgs(),
            head=C.Head.Linear,
            unfreeze_layers=["pre_layrnorm", "layer_norm1", "layer_norm2", "post_layernorm"],
            loss=C.Loss(
                ce_labels=1.0, 
                label_smoothing=0.0, 
                uniformity=0.5, 
                alignment_labels=0.1,
                ),
            run_dir="runs/train",
            trn_files=FF.train,
            val_files=Files(CDFv3.val + DeepSpeak_v1_1.val + DeepSpeak_v2.val + FFIW.val),
            tst_files=Files(UADFV.test + DFD.test + DFDC.test + FSh.test + FFIW.test),
            batch_size=96,
            mini_batch_size=96,
            wandb=True,
            devices=[0],
            lr_scheduler="cyclic",
            num_epochs_in_cycle=10,
            max_epochs=100,
            warmup_epochs=1,
            early_stopping_patience=10,
            monitor_metric="val/auroc_frame",
            checkpoint_name="best_val_auroc_frame",
            min_delta=0.002,
            checkpoint="runs/train/clip-from-gend-v2/checkpoints/own-clip-best_val_auroc_frame.ckpt",
            #wandb_id = "ulqep2rd",
            #resume=True,
            #throw_exception_if_run_exists=False,
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
