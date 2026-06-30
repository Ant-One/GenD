from .. import config as C
from ..config import Config

from ..utils.files import Files, DF40, DF40Balanced, FSh, UADFV, DFD, DFDC, FFIW, FF, CDFv3, DeepSpeak_v1_1, DeepSpeak_v1, CDFv1, KoDF, FakeAVCeleb, DFDM, PolyGlotFake, IDForge_v1

experiments = {

        "test-own-pe-v1-uadfv": [
        Config(
            run_dir="runs/test-own/pev1",
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            tst_files=UADFV.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
            from_exp="pe-from-gend-v1"
        )
    ],
    "test-own-pe-v1-dfd": [
        Config(
            run_dir="runs/test-own/pev1",
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            tst_files=DFD.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
            from_exp="pe-from-gend-v1"
        )
    ],
    "test-own-pe-v1-dfdc": [
        Config(
            run_dir="runs/test-own/pev1",
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            tst_files=DFDC.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
            from_exp="pe-from-gend-v1"
        )
    ],
    "test-own-pe-v1-fsh": [
        Config(
            run_dir="runs/test-own/pev1",
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            tst_files=FSh.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
            from_exp="pe-from-gend-v1"
        )
    ],
    "test-own-pe-v1-cdfv1": [
        Config(
            run_dir="runs/test-own/pev1",
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            tst_files=CDFv1.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
            from_exp="pe-from-gend-v1"
        )
    ],

    "test-own-pe-v1-ffiw": [
        Config(
            run_dir="runs/test-own/pev1",
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            tst_files=FFIW.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
            from_exp="pe-from-gend-v1"
        )
    ],

    "test-own-pe-v1-kodf-normal": [
        Config(
            run_dir="runs/test-own/pev1",
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            tst_files=KoDF.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
            from_exp="pe-from-gend-v1"
        )
    ],
    "test-own-pe-v1-kodf-adv": [
        Config(
            run_dir="runs/test-own/pev1",
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            tst_files=KoDF.adversarial,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
            from_exp="pe-from-gend-v1"
        )
    ],
    "test-own-pe-v1-kodf-all": [
        Config(
            run_dir="runs/test-own/pev1",
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            tst_files=KoDF.test_all,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
            from_exp="pe-from-gend-v1"
        )
    ],
    "test-own-pe-v1-fakeavceleb": [
        Config(
            run_dir="runs/test-own/pev1",
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            tst_files=FakeAVCeleb.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
            from_exp="pe-from-gend-v1"
        )
    ],
    "test-own-pe-v1-dfdm": [
        Config(
            run_dir="runs/test-own/pev1",
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            tst_files=DFDM.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
            from_exp="pe-from-gend-v1"
        )
    ],

    "test-own-pe-v1-polyglotfake": [
        Config(
            run_dir="runs/test-own/pev1",
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            tst_files=PolyGlotFake.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
            from_exp="pe-from-gend-v1"
        )
    ],
    "test-own-pe-v1-idforge": [
        Config(
            run_dir="runs/test-own/pev1",
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            tst_files=IDForge_v1.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
            from_exp="pe-from-gend-v1"
        )
    ],
    "test-own-pe-v1-deepspeakv1": [
        Config(
            run_dir="runs/test-own/pev1",
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            tst_files=DeepSpeak_v1_1.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
            from_exp="pe-from-gend-v1"
        )
    ],
    "test-own-pe-v1-deepspeakv1": [
        Config(
            run_dir="runs/test-own/pev1",
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            tst_files=DeepSpeak_v1.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
            from_exp="pe-from-gend-v1"
        )
    ],
    "test-own-pe-v1-cdfv3": [
        Config(
            run_dir="runs/test-own/pev1",
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            tst_files=CDFv3.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
            from_exp="pe-from-gend-v1"
        )
    ],

        "pe-from-gend-v1": [
        Config(
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            backbone_args=C.BackboneArgs(),
            head=C.Head.NLinear,
            unfreeze_layers=["norm1", "norm2", "norm"],
            loss=C.Loss(
                ce_labels=1.0, 
                label_smoothing=0.0, 
                uniformity=0.5, 
                alignment_labels=0.1,
                ),
            run_dir="runs/train",
            trn_files=FF.train,
            val_files=Files(CDFv3.val + DeepSpeak_v1_1.val + DeepSpeak_v1.val + FFIW.val),
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
            checkpoint_name="best_val_roc_frame",
            min_delta=0.002,
            #checkpoint="/home/antoine/GenD/runs/train/pe-from-gend-v1/checkpoints/own-pe-from-gend-best_mAP.ckpt"
        )
    ],
}
