from .. import config as C
from ..config import Config
from ..utils.files import (
    DF40,
    DFD,
    DFDC,
    DFDM,
    FF,
    FFIW,
    NTIRE,
    UADFV,
    CDFv2,
    CDFv3,
    ConfDF,
    Deepfake_vs_Real_60k,
    DeepSpeak_v1_1,
    DeepSpeak_v2,
    DF40Balanced,
    FakeAVCeleb,
    Files,
    FSh,
    IDForge_v1,
    KoDF,
    PolyGlotFake,
)

experiments = {

    "pe-dfd-v1": [
        Config(
            backbone=C.Backbone.PerceptionEncoder_L_p14_336,
            backbone_args=C.BackboneArgs(),
            head=C.Head.NLinear,
            unfreeze_layers=["norm_pre", "norm1", "norm2", "norm"],
            loss=C.Loss(
                ce_labels=1.0,
                #label_smoothing=0.0,
                uniformity=0.5,
                alignment_labels=0.1,
                ),
            run_dir="runs/com-train",
            trn_files=DFD.train,
            val_files=Files(ConfDF.val + Deepfake_vs_Real_60k.val),
            tst_files=Files(CDFv2.test + CDFv3.test + DeepSpeak_v1_1.test + DeepSpeak_v2.test + DFD.test + DFDC.test + DFDM.test + FakeAVCeleb.test + FFIW.test + FSh.test + IDForge_v1.test + KoDF.test + PolyGlotFake.test + UADFV.test),
            batch_size=96,
            mini_batch_size=96,
            wandb=True,
            devices=[0],
            lr_scheduler="cyclic",
            num_epochs_in_cycle=10,
            max_epochs=30,
            warmup_epochs=1,
            early_stopping_patience=-1,
            checkpoint_name = "best_mAP",
            monitor_metric = "val/mAP_video",
            min_delta=0.002,
            #checkpoint="runs/train/clip-from-gend-v2/checkpoints/own-clip-best_val_auroc_frame.ckpt",
            #wandb_id = "ulqep2rd",
            #resume=True,
            #throw_exception_if_run_exists=False,
        )
    ],

         "test-own-pe-v3-ff-ntire": [
            Config(
                run_dir="runs/test-own/pev1",
                backbone=C.Backbone.PerceptionEncoder_L_p14_336,
                backbone_args=C.BackboneArgs(),
                head=C.Head.NLinear,
                tst_files=NTIRE.val,
                batch_size=128,
                mini_batch_size=128,
                wandb=False,
                devices=[0],
                checkpoint="runs/train/pe-from-gend-v3/checkpoints/own-pe-best_val_roc_frame.ckpt",
                from_exp="pe-from-gend-v1"
            )
        ],

    #     "test-own-pe-v1-uadfv": [
    #     Config(
    #         run_dir="runs/test-own/pev1",
    #         backbone=C.Backbone.PerceptionEncoder_L_p14_336,
    #         backbone_args=C.BackboneArgs(),
    #         head=C.Head.NLinear,
    #         tst_files=UADFV.test,
    #         batch_size=128,
    #         mini_batch_size=128,
    #         wandb=False,
    #         devices=[0],
    #         checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
    #         from_exp="pe-from-gend-v1"
    #     )
    # ],
    # "test-own-pe-v1-dfd": [
    #     Config(
    #         run_dir="runs/test-own/pev1",
    #         backbone=C.Backbone.PerceptionEncoder_L_p14_336,
    #         backbone_args=C.BackboneArgs(),
    #         head=C.Head.NLinear,
    #         tst_files=DFD.test,
    #         batch_size=128,
    #         mini_batch_size=128,
    #         wandb=False,
    #         devices=[0],
    #         checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
    #         from_exp="pe-from-gend-v1"
    #     )
    # ],
    # "test-own-pe-v1-dfdc": [
    #     Config(
    #         run_dir="runs/test-own/pev1",
    #         backbone=C.Backbone.PerceptionEncoder_L_p14_336,
    #         backbone_args=C.BackboneArgs(),
    #         head=C.Head.NLinear,
    #         tst_files=DFDC.test,
    #         batch_size=128,
    #         mini_batch_size=128,
    #         wandb=False,
    #         devices=[0],
    #         checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
    #         from_exp="pe-from-gend-v1"
    #     )
    # ],
    # "test-own-pe-v1-fsh": [
    #     Config(
    #         run_dir="runs/test-own/pev1",
    #         backbone=C.Backbone.PerceptionEncoder_L_p14_336,
    #         backbone_args=C.BackboneArgs(),
    #         head=C.Head.NLinear,
    #         tst_files=FSh.test,
    #         batch_size=128,
    #         mini_batch_size=128,
    #         wandb=False,
    #         devices=[0],
    #         checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
    #         from_exp="pe-from-gend-v1"
    #     )
    # ],
    # "test-own-pe-v1-cdfv2": [
    #     Config(
    #         run_dir="runs/test-own/pev1",
    #         backbone=C.Backbone.PerceptionEncoder_L_p14_336,
    #         backbone_args=C.BackboneArgs(),
    #         head=C.Head.NLinear,
    #         tst_files=CDFv2.test,
    #         batch_size=128,
    #         mini_batch_size=128,
    #         wandb=False,
    #         devices=[0],
    #         checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
    #         from_exp="pe-from-gend-v1"
    #     )
    # ],

    # "test-own-pe-v1-ffiw": [
    #     Config(
    #         run_dir="runs/test-own/pev1",
    #         backbone=C.Backbone.PerceptionEncoder_L_p14_336,
    #         backbone_args=C.BackboneArgs(),
    #         head=C.Head.NLinear,
    #         tst_files=FFIW.test,
    #         batch_size=128,
    #         mini_batch_size=128,
    #         wandb=False,
    #         devices=[0],
    #         checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
    #         from_exp="pe-from-gend-v1"
    #     )
    # ],

    # "test-own-pe-v1-kodf-normal": [
    #     Config(
    #         run_dir="runs/test-own/pev1",
    #         backbone=C.Backbone.PerceptionEncoder_L_p14_336,
    #         backbone_args=C.BackboneArgs(),
    #         head=C.Head.NLinear,
    #         tst_files=KoDF.test,
    #         batch_size=128,
    #         mini_batch_size=128,
    #         wandb=False,
    #         devices=[0],
    #         checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
    #         from_exp="pe-from-gend-v1"
    #     )
    # ],
    # "test-own-pe-v1-kodf-adv": [
    #     Config(
    #         run_dir="runs/test-own/pev1",
    #         backbone=C.Backbone.PerceptionEncoder_L_p14_336,
    #         backbone_args=C.BackboneArgs(),
    #         head=C.Head.NLinear,
    #         tst_files=KoDF.adversarial,
    #         batch_size=128,
    #         mini_batch_size=128,
    #         wandb=False,
    #         devices=[0],
    #         checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
    #         from_exp="pe-from-gend-v1"
    #     )
    # ],
    # "test-own-pe-v1-kodf-all": [
    #     Config(
    #         run_dir="runs/test-own/pev1",
    #         backbone=C.Backbone.PerceptionEncoder_L_p14_336,
    #         backbone_args=C.BackboneArgs(),
    #         head=C.Head.NLinear,
    #         tst_files=KoDF.test_all,
    #         batch_size=128,
    #         mini_batch_size=128,
    #         wandb=False,
    #         devices=[0],
    #         checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
    #         from_exp="pe-from-gend-v1"
    #     )
    # ],
    # "test-own-pe-v1-fakeavceleb": [
    #     Config(
    #         run_dir="runs/test-own/pev1",
    #         backbone=C.Backbone.PerceptionEncoder_L_p14_336,
    #         backbone_args=C.BackboneArgs(),
    #         head=C.Head.NLinear,
    #         tst_files=FakeAVCeleb.test,
    #         batch_size=128,
    #         mini_batch_size=128,
    #         wandb=False,
    #         devices=[0],
    #         checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
    #         from_exp="pe-from-gend-v1"
    #     )
    # ],
    # "test-own-pe-v1-dfdm": [
    #     Config(
    #         run_dir="runs/test-own/pev1",
    #         backbone=C.Backbone.PerceptionEncoder_L_p14_336,
    #         backbone_args=C.BackboneArgs(),
    #         head=C.Head.NLinear,
    #         tst_files=DFDM.test,
    #         batch_size=128,
    #         mini_batch_size=128,
    #         wandb=False,
    #         devices=[0],
    #         checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
    #         from_exp="pe-from-gend-v1"
    #     )
    # ],

    # "test-own-pe-v1-polyglotfake": [
    #     Config(
    #         run_dir="runs/test-own/pev1",
    #         backbone=C.Backbone.PerceptionEncoder_L_p14_336,
    #         backbone_args=C.BackboneArgs(),
    #         head=C.Head.NLinear,
    #         tst_files=PolyGlotFake.test,
    #         batch_size=128,
    #         mini_batch_size=128,
    #         wandb=False,
    #         devices=[0],
    #         checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
    #         from_exp="pe-from-gend-v1"
    #     )
    # ],
    # "test-own-pe-v1-idforge": [
    #     Config(
    #         run_dir="runs/test-own/pev1",
    #         backbone=C.Backbone.PerceptionEncoder_L_p14_336,
    #         backbone_args=C.BackboneArgs(),
    #         head=C.Head.NLinear,
    #         tst_files=IDForge_v1.test,
    #         batch_size=128,
    #         mini_batch_size=128,
    #         wandb=False,
    #         devices=[0],
    #         checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
    #         from_exp="pe-from-gend-v1"
    #     )
    # ],
    # "test-own-pe-v1-deepspeakv1": [
    #     Config(
    #         run_dir="runs/test-own/pev1",
    #         backbone=C.Backbone.PerceptionEncoder_L_p14_336,
    #         backbone_args=C.BackboneArgs(),
    #         head=C.Head.NLinear,
    #         tst_files=DeepSpeak_v1_1.test,
    #         batch_size=128,
    #         mini_batch_size=128,
    #         wandb=False,
    #         devices=[0],
    #         checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
    #         from_exp="pe-from-gend-v1"
    #     )
    # ],
    # "test-own-pe-v1-deepspeakv2": [
    #     Config(
    #         run_dir="runs/test-own/pev1",
    #         backbone=C.Backbone.PerceptionEncoder_L_p14_336,
    #         backbone_args=C.BackboneArgs(),
    #         head=C.Head.NLinear,
    #         tst_files=DeepSpeak_v2.test,
    #         batch_size=128,
    #         mini_batch_size=128,
    #         wandb=False,
    #         devices=[0],
    #         checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
    #         from_exp="pe-from-gend-v1"
    #     )
    # ],
    # "test-own-pe-v1-cdfv3": [
    #     Config(
    #         run_dir="runs/test-own/pev1",
    #         backbone=C.Backbone.PerceptionEncoder_L_p14_336,
    #         backbone_args=C.BackboneArgs(),
    #         head=C.Head.NLinear,
    #         tst_files=CDFv3.test,
    #         batch_size=128,
    #         mini_batch_size=128,
    #         wandb=False,
    #         devices=[0],
    #         checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
    #         from_exp="pe-from-gend-v1"
    #     )
    # ],

    #     "pe-from-gend-v1": [
    #     Config(
    #         backbone=C.Backbone.PerceptionEncoder_L_p14_336,
    #         backbone_args=C.BackboneArgs(),
    #         head=C.Head.NLinear,
    #         unfreeze_layers=["norm1", "norm2", "norm"],
    #         loss=C.Loss(
    #             ce_labels=1.0,
    #             label_smoothing=0.0,
    #             uniformity=0.5,
    #             alignment_labels=0.1,
    #             ),
    #         run_dir="runs/train",
    #         trn_files=FF.train,
    #         val_files=Files(CDFv3.val + DeepSpeak_v1_1.val + DeepSpeak_v2.val + FFIW.val),
    #         tst_files=Files(UADFV.test + DFD.test + DFDC.test + FSh.test + FFIW.test),
    #         batch_size=96,
    #         mini_batch_size=96,
    #         wandb=True,
    #         devices=[0],
    #         lr_scheduler="cyclic",
    #         num_epochs_in_cycle=10,
    #         max_epochs=100,
    #         warmup_epochs=1,
    #         early_stopping_patience=10,
    #         monitor_metric="val/auroc_frame",
    #         checkpoint_name="best_val_roc_frame",
    #         min_delta=0.002,
    #         #checkpoint="/home/antoine/GenD/runs/train/pe-from-gend-v1/checkpoints/own-pe-from-gend-best_mAP.ckpt"
    #     )
    # ],
}
