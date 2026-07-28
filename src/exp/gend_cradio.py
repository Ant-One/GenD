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

    "cradio-v2-dfd-sam-labelsmooth": [
        Config(
            backbone=C.Backbone.C_RADIOV4_SO400M,
            backbone_args=C.BackboneArgs(),
            head=C.Head.NLinear,
            unfreeze_layers=["norm1", "norm2"],
            loss=C.Loss(
                ce_labels=1.0,
                uniformity=0.5,
                alignment_labels=0.1,
                label_smoothing=0.1,
            ),
            run_dir="runs/sam",
            trn_files=DFD.train,
            #trn_files=Files(FF.DF.train + FF.FS.train + FF.NT.train),
            #val_files=Files(ConfDF.val + Deepfake_vs_Real_60k.val),
            val_files=Files(CDFv3.val + DeepSpeak_v1_1.val + DeepSpeak_v2.val + FFIW.val),
            tst_files=Files(CDFv2.test + CDFv3.test + DeepSpeak_v1_1.test + DeepSpeak_v2.test + DFD.test + DFDC.test + DFDM.test + FakeAVCeleb.test + FFIW.test + FSh.test + IDForge_v1.test + KoDF.test + PolyGlotFake.test + UADFV.test),
            batch_size=64,
            mini_batch_size=64,
            wandb=True,
            devices=[0],
            lr_scheduler="cyclic",
            num_epochs_in_cycle=10,
            max_epochs=30,
            warmup_epochs=1,
            early_stopping_patience=-1,
            checkpoint_name="best_mAP",
            monitor_metric="val/mAP_video",
            min_delta=0.002,
            optimizer="SAM-AdamW",
            sam_rho=0.05,
            sam_adaptive=True,
            weight_decay=1e-4,
        )
    ],

#     "test-cradio-ff-no-f2f-v1-ntire": [
#         Config(
#             run_dir="runs/com-test/cradio-ff-no-f2f",
#             backbone=C.Backbone.C_RADIOV4_SO400M,
#             tst_files=NTIRE.val,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="runs/com-train/cradio-ff-no-f2f-v1/checkpoints/own-cradio-best_mAP.ckpt",
#             from_exp="cradio-ff-no-f2f-v11"
#         )
#     ],

#     "test-own-cradio-v1-uadfv": [
#         Config(
#             run_dir="runs/com-test/cradio-ff-no-f2f",
#             backbone=C.Backbone.C_RADIOV4_SO400M,
#             tst_files=UADFV.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="runs/com-train/cradio-ff-no-f2f-v1/checkpoints/own-cradio-best_mAP.ckpt",
#             from_exp="cradio-ff-no-f2f-v11"
#         )
#     ],
#     "test-own-cradio-v1-dfd": [
#         Config(
#             run_dir="runs/com-test/cradio-ff-no-f2f",
#             backbone=C.Backbone.C_RADIOV4_SO400M,
#             tst_files=DFD.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="runs/com-train/cradio-ff-no-f2f-v1/checkpoints/own-cradio-best_mAP.ckpt",
#             from_exp="cradio-ff-no-f2f-v11"
#         )
#     ],
#     "test-own-cradio-v1-dfdc": [
#         Config(
#             run_dir="runs/com-test/cradio-ff-no-f2f",
#             backbone=C.Backbone.C_RADIOV4_SO400M,
#             tst_files=DFDC.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="runs/com-train/cradio-ff-no-f2f-v1/checkpoints/own-cradio-best_mAP.ckpt",
#             from_exp="cradio-ff-no-f2f-v11"
#         )
#     ],
#     "test-own-cradio-v1-fsh": [
#         Config(
#             run_dir="runs/com-test/cradio-ff-no-f2f",
#             backbone=C.Backbone.C_RADIOV4_SO400M,
#             tst_files=FSh.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="runs/com-train/cradio-ff-no-f2f-v1/checkpoints/own-cradio-best_mAP.ckpt",
#             from_exp="cradio-ff-no-f2f-v11"
#         )
#     ],
#     "test-own-cradio-v1-cdfv2": [
#         Config(
#             run_dir="runs/com-test/cradio-ff-no-f2f",
#             backbone=C.Backbone.C_RADIOV4_SO400M,
#             tst_files=CDFv2.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="runs/com-train/cradio-ff-no-f2f-v1/checkpoints/own-cradio-best_mAP.ckpt",
#             from_exp="cradio-ff-no-f2f-v11"
#         )
#     ],

#     "test-own-cradio-v1-ffiw": [
#         Config(
#             run_dir="runs/com-test/cradio-ff-no-f2f",
#             backbone=C.Backbone.C_RADIOV4_SO400M,
#             tst_files=FFIW.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="runs/com-train/cradio-ff-no-f2f-v1/checkpoints/own-cradio-best_mAP.ckpt",
#             from_exp="cradio-ff-no-f2f-v11"
#         )
#     ],

#     "test-own-cradio-v1-kodf-normal": [
#         Config(
#             run_dir="runs/com-test/cradio-ff-no-f2f",
#             backbone=C.Backbone.C_RADIOV4_SO400M,
#             tst_files=KoDF.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="runs/com-train/cradio-ff-no-f2f-v1/checkpoints/own-cradio-best_mAP.ckpt",
#             from_exp="cradio-ff-no-f2f-v11"
#         )
#     ],
#     "test-own-cradio-v1-kodf-adv": [
#         Config(
#             run_dir="runs/com-test/cradio-ff-no-f2f",
#             backbone=C.Backbone.C_RADIOV4_SO400M,
#             tst_files=KoDF.adversarial,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="runs/com-train/cradio-ff-no-f2f-v1/checkpoints/own-cradio-best_mAP.ckpt",
#             from_exp="cradio-ff-no-f2f-v11"
#         )
#     ],
#     "test-own-cradio-v1-kodf-all": [
#         Config(
#             run_dir="runs/com-test/cradio-ff-no-f2f",
#             backbone=C.Backbone.C_RADIOV4_SO400M,
#             tst_files=KoDF.test_all,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="runs/com-train/cradio-ff-no-f2f-v1/checkpoints/own-cradio-best_mAP.ckpt",
#             from_exp="cradio-ff-no-f2f-v11"
#         )
#     ],
#     "test-own-cradio-v1-fakeavceleb": [
#         Config(
#             run_dir="runs/com-test/cradio-ff-no-f2f",
#             backbone=C.Backbone.C_RADIOV4_SO400M,
#             tst_files=FakeAVCeleb.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="runs/com-train/cradio-ff-no-f2f-v1/checkpoints/own-cradio-best_mAP.ckpt",
#             from_exp="cradio-ff-no-f2f-v11"
#         )
#     ],
#     "test-own-cradio-v1-dfdm": [
#         Config(
#             run_dir="runs/com-test/cradio-ff-no-f2f",
#             backbone=C.Backbone.C_RADIOV4_SO400M,
#             tst_files=DFDM.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="runs/com-train/cradio-ff-no-f2f-v1/checkpoints/own-cradio-best_mAP.ckpt",
#             from_exp="cradio-ff-no-f2f-v11"
#         )
#     ],

#     "test-own-cradio-v1-polyglotfake": [
#         Config(
#             run_dir="runs/com-test/cradio-ff-no-f2f",
#             backbone=C.Backbone.C_RADIOV4_SO400M,
#             tst_files=PolyGlotFake.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="runs/com-train/cradio-ff-no-f2f-v1/checkpoints/own-cradio-best_mAP.ckpt",
#             from_exp="cradio-ff-no-f2f-v11"
#         )
#     ],
#     "test-own-cradio-v1-idforge": [
#         Config(
#             run_dir="runs/com-test/cradio-ff-no-f2f",
#             backbone=C.Backbone.C_RADIOV4_SO400M,
#             tst_files=IDForge_v1.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="runs/com-train/cradio-ff-no-f2f-v1/checkpoints/own-cradio-best_mAP.ckpt",
#             from_exp="cradio-ff-no-f2f-v11"
#         )
#     ],
#     "test-own-cradio-v1-deepspeakv1": [
#         Config(
#             run_dir="runs/com-test/cradio-ff-no-f2f",
#             backbone=C.Backbone.C_RADIOV4_SO400M,
#             tst_files=DeepSpeak_v1_1.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="runs/com-train/cradio-ff-no-f2f-v1/checkpoints/own-cradio-best_mAP.ckpt",
#             from_exp="cradio-ff-no-f2f-v11"
#         )
#     ],
#     "test-own-cradio-v1-deepspeakv2": [
#         Config(
#             run_dir="runs/com-test/cradio-ff-no-f2f",
#             backbone=C.Backbone.C_RADIOV4_SO400M,
#             tst_files=DeepSpeak_v2.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="runs/com-train/cradio-ff-no-f2f-v1/checkpoints/own-cradio-best_mAP.ckpt",
#             from_exp="cradio-ff-no-f2f-v11"
#         )
#     ],
#     "test-own-cradio-v1-cdfv3": [
#         Config(
#             run_dir="runs/com-test/cradio-ff-no-f2f",
#             backbone=C.Backbone.C_RADIOV4_SO400M,
#             tst_files=CDFv3.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="runs/com-train/cradio-ff-no-f2f-v1/checkpoints/own-cradio-best_mAP.ckpt",
#             from_exp="cradio-ff-no-f2f-v11"
#         )
#     ],
}
