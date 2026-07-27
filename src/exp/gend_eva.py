from .. import config as C
from ..config import Config

from ..utils.files import Files, DF40, DF40Balanced, FSh, UADFV, DFD, DFDC, FFIW, FF, CDFv3, DeepSpeak_v1_1, DeepSpeak_v2, CDFv2, KoDF, FakeAVCeleb, DFDM, PolyGlotFake, IDForge_v1, ConfDF, Deepfake_vs_Real_60k, NTIRE

experiments = {

    "eva-dfd-v1": [
        Config(
            backbone=C.Backbone.EVA02_L_P14_224,
            backbone_args=C.BackboneArgs(img_size=224),
            head=C.Head.NLinear,
            unfreeze_layers=["norm_pre", "norm1", "norm2", "norm"],
            loss=C.Loss(
                ce_labels=1.0, 
                uniformity=0.5, 
                alignment_labels=0.1,
            ),
            run_dir="runs/com-train",
            trn_files=DFD.train,
            val_files=Files(ConfDF.val + Deepfake_vs_Real_60k.val),
            tst_files=Files(CDFv2.test + CDFv3.test + DeepSpeak_v1_1.test + DeepSpeak_v2.test + DFD.test + DFDC.test + DFDM.test + FakeAVCeleb.test + FFIW.test + FSh.test + IDForge_v1.test + KoDF.test + PolyGlotFake.test + UADFV.test),
            batch_size=96,
            mini_batch_size=96,
            wandb=False,
            devices=[0],
            lr_scheduler="cyclic",
            num_epochs_in_cycle=10,
            max_epochs=30,
            warmup_epochs=1,
            early_stopping_patience=-1,
            checkpoint_name="best_mAP",
            monitor_metric="val/mAP_video",
            min_delta=0.002,
        )
    ],

    "test-eva-dfd-v1-ntire": [
        Config(
            run_dir="runs/com-test/eva-v1",
            backbone=C.Backbone.EVA02_L_P14_224,
            tst_files=NTIRE.val,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/com-train/eva-dfd-v1/checkpoints/own-eva-best_mAP.ckpt",
            from_exp="eva-dfd-v1"
        )
    ],
}
