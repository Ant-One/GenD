from .. import config as C
from ..config import Config

from ..utils.files import Files, DF40, DF40Balanced, FSh, UADFV, DFD, DFDC, FFIW, FF, CDFv3, DeepSpeak_v1_1, DeepSpeak_v2, CDFv2, KoDF, FakeAVCeleb, DFDM, PolyGlotFake, IDForge_v1

experiments = {
        "dino-from-gend-v2": [
        Config(
            backbone=C.Backbone.DINOv3_ViT_L,
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
            checkpoint_name="best_val_roc_frame",
            min_delta=0.002,
            #checkpoint="/home/antoine/GenD/runs/train/dino-from-gend-v1/checkpoints/own-dino-from-gend-best_mAP.ckpt"
        )
    ],
}
