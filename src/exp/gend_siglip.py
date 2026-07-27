from .. import config as C
from ..config import Config

from ..utils.files import Files, DF40, DF40Balanced, FSh, UADFV, DFD, DFDC, FFIW, FF, CDFv3, DeepSpeak_v1_1, DeepSpeak_v2, CDFv2, KoDF, FakeAVCeleb, DFDM, PolyGlotFake, IDForge_v1, ConfDF, Deepfake_vs_Real_60k, NTIRE

experiments = {

    "siglip-dfd-v1": [
        Config(
            backbone=C.Backbone.SIGLIP2_SO400M_P14_224,
            backbone_args=C.BackboneArgs(),
            head=C.Head.NLinear,
            #unfreeze_layers=["layer_norm1", "layer_norm2", "post_layernorm"],
            unfreeze_layers=["layer_norm1", "layer_norm2", "post_layernorm", "head.layernorm"],
            loss=C.Loss(
                ce_labels=1.0, 
                uniformity=0.5, 
                alignment_labels=0.1,
            ),
            run_dir="runs/com-train/siglip",
            trn_files=DFD.train,
            #val_files=Files(ConfDF.val + Deepfake_vs_Real_60k.val),
            val_files=Files(CDFv3.val + DeepSpeak_v1_1.val + DeepSpeak_v2.val + FFIW.val),
            tst_files=Files(CDFv2.test + CDFv3.test + DeepSpeak_v1_1.test + DeepSpeak_v2.test + DFD.test + DFDC.test + DFDM.test + FakeAVCeleb.test + FFIW.test + FSh.test + IDForge_v1.test + KoDF.test + PolyGlotFake.test + UADFV.test),
            batch_size=96,
            mini_batch_size=96,
            wandb=True,
            devices=[0],
            lr_scheduler="cosine",
            max_epochs=30,
            warmup_epochs=1,
            early_stopping_patience=-1,
            checkpoint_name="best_mAP",
            monitor_metric="val/mAP_video",
            min_delta=0.002,
        )
    ],

        "test-siglip-dfd-v1-ntire": [
        Config(
            run_dir="runs/com-test/siglip-v1",
            backbone=C.Backbone.SIGLIP2_SO400M_P14_224,
            tst_files=NTIRE.val,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="runs/com-train/siglip/siglip-dfd-v1/checkpoints/own-siglip-best_mAP.ckpt",
            from_exp="siglip-dfd-v1"
        )
    ],

    # "test-siglip-dfd-v1-ff": [
    #     Config(
    #         run_dir="runs/com-test/siglip-v1",
    #         backbone=C.Backbone.SIGLIP2_SO400M_P14_224,
    #         tst_files=FF.train,
    #         batch_size=128,
    #         mini_batch_size=128,
    #         wandb=False,
    #         devices=[0],
    #         checkpoint="runs/com-train/siglip-dfd-v1/checkpoints/own-siglip-best_mAP.ckpt",
    #         from_exp="siglip-dfd-v1"
    #     )
    # ],
}
