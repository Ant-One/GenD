from .. import config as C
from ..config import Config
from ..utils.files import (
    DF40,
    DFD,
    DFDC,
    DFDM,
    FF,
    FFIW,
    UADFV,
    CDFv2,
    CDFv3,
    ConfDF,
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

    # "test-own-clip-v2-confdf": [
    #         Config(
    #             run_dir="runs/test-confdf/",
    #             backbone=C.Backbone.CLIP_L_14,
    #             backbone_args=C.BackboneArgs(),
    #             tst_files=ConfDF.test,
    #             batch_size=128,
    #             mini_batch_size=128,
    #             wandb=False,
    #             devices=[0],
    #             checkpoint="runs/train/clip-from-gend-v2/checkpoints/own-clip-best_val_auroc_frame-v2.ckpt",
    #             #from_exp="clip-from-gend-v3-100-epoch"
    #         )
    #     ],
    #     "test-own-dino-v2-confdf": [
    #         Config(
    #             run_dir="runs/test-confdf/",
    #             backbone=C.Backbone.DINOv3_ViT_L,
    #             backbone_args=C.BackboneArgs(),
    #             tst_files=ConfDF.test,
    #             batch_size=128,
    #             mini_batch_size=128,
    #             wandb=False,
    #             devices=[0],
    #             checkpoint="runs/train/dino-from-gend-v2/checkpoints/own-dino-best_val_roc_frame.ckpt",
    #             #from_exp="clip-from-gend-v3-100-epoch"
    #         )
    #     ],
    #     "test-own-pe-v1-confdf": [
    #         Config(
    #             run_dir="runs/test-confdf/",
    #             backbone=C.Backbone.PerceptionEncoder_L_p14_336,
    #             backbone_args=C.BackboneArgs(),
    #             tst_files=ConfDF.test,
    #             batch_size=128,
    #             mini_batch_size=128,
    #             wandb=False,
    #             devices=[0],
    #             checkpoint="runs/train/pe-from-gend-v1/checkpoints/own-pe-best_val_roc_frame.ckpt",
    #             #from_exp="clip-from-gend-v3-100-epoch"
    #         )
    #     ],
    #     "test-yermandy-clip-confdf": [
    #         Config(
    #             run_dir="runs/test-confdf/",
    #             backbone=C.Backbone.CLIP_L_14,
    #             backbone_args=C.BackboneArgs(),
    #             tst_files=ConfDF.test,
    #             batch_size=128,
    #             mini_batch_size=128,
    #             wandb=False,
    #             devices=[0],
    #             checkpoint="yermandy/GenD_CLIP_L_14",
    #             #from_exp="clip-from-gend-v3-100-epoch"
    #         )
    #     ],
    #     "test-yermandy-dino-confdf": [
    #         Config(
    #             run_dir="runs/test-confdf/",
    #             backbone=C.Backbone.DINOv3_ViT_L,
    #             backbone_args=C.BackboneArgs(),
    #             tst_files=ConfDF.test,
    #             batch_size=128,
    #             mini_batch_size=128,
    #             wandb=False,
    #             devices=[0],
    #             checkpoint="yermandy/GenD_DINOv3_L",
    #             #from_exp="clip-from-gend-v3-100-epoch"
    #         )
    #     ],
    #     "test-yermandy-pe-confdf": [
    #         Config(
    #             run_dir="runs/test-confdf/",
    #             backbone=C.Backbone.PerceptionEncoder_L_p14_336,
    #             backbone_args=C.BackboneArgs(),
    #             tst_files=ConfDF.test,
    #             batch_size=128,
    #             mini_batch_size=128,
    #             wandb=False,
    #             devices=[0],
    #             checkpoint="yermandy/GenD_PE_L",
    #             #from_exp="clip-from-gend-v3-100-epoch"
    #         )
    #     ],

    #      "test-own-clip-v3-confdf": [
    #         Config(
    #             run_dir="runs/test-confdf/",
    #             backbone=C.Backbone.CLIP_L_14,
    #             backbone_args=C.BackboneArgs(),
    #             tst_files=ConfDF.test,
    #             batch_size=128,
    #             mini_batch_size=128,
    #             wandb=False,
    #             devices=[0],
    #             checkpoint="runs/train/clip-from-gend-v3/checkpoints/own-clip-best_val_auroc_frame.ckpt",
    #             #from_exp="clip-from-gend-v3-100-epoch"
    #         )
    #     ],
    #      "test-own-dino-v3-confdf": [
    #         Config(
    #             run_dir="runs/test-confdf/",
    #             backbone=C.Backbone.DINOv3_ViT_L,
    #             backbone_args=C.BackboneArgs(),
    #             tst_files=ConfDF.test,
    #             batch_size=128,
    #             mini_batch_size=128,
    #             wandb=False,
    #             devices=[0],
    #             checkpoint="runs/train/dino-from-gend-v3/checkpoints/own-dino-best_val_roc_frame.ckpt",
    #             #from_exp="clip-from-gend-v3-100-epoch"
    #         )
    #     ],
    #      "test-own-pe-v3-confdf": [
    #         Config(
    #             run_dir="runs/test-confdf/",
    #             backbone=C.Backbone.PerceptionEncoder_L_p14_336,
    #             backbone_args=C.BackboneArgs(),
    #             tst_files=ConfDF.test,
    #             batch_size=128,
    #             mini_batch_size=128,
    #             wandb=False,
    #             devices=[0],
    #             checkpoint="runs/train/pe-from-gend-v3/checkpoints/own-pe-best_val_roc_frame.ckpt",
    #             #from_exp="clip-from-gend-v3-100-epoch"
    #         )
    #     ],
    # "test-own-clip-v3-nof2f": [
    #         Config(
    #             run_dir="runs/test-confdf/",
    #             backbone=C.Backbone.CLIP_L_14,
    #             backbone_args=C.BackboneArgs(),
    #             tst_files=ConfDF.test,
    #             batch_size=128,
    #             mini_batch_size=128,
    #             wandb=False,
    #             devices=[0],
    #             checkpoint="runs/train/clip-from-gend-v3-nof2f/checkpoints/best_mAPvideo.ckpt",
    #             #from_exp="clip-from-gend-v3-100-epoch"
    #         )
    #     ],
        # "test-own-clip-v4-nlin": [
        #     Config(
        #         run_dir="runs/test-confdf/",
        #         backbone=C.Backbone.CLIP_L_14,
        #         backbone_args=C.BackboneArgs(),
        #         tst_files=ConfDF.test,
        #         batch_size=128,
        #         mini_batch_size=128,
        #         wandb=False,
        #         devices=[0],
        #         checkpoint="runs/train/clip-from-gend-v4-nlin/checkpoints/own-clip-v4-best_val_auroc_frame.ckpt",
        #         #from_exp="clip-from-gend-v3-100-epoch"
        #     )
        # ],
}
