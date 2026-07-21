from .. import config as C
from ..config import Config

from ..utils.files import Files, DF40, DF40Balanced, FSh, UADFV, DFD, DFDC, FFIW, FF, CDFv3, DeepSpeak_v1_1, DeepSpeak_v2, CDFv2, KoDF, FakeAVCeleb, DFDM, PolyGlotFake, IDForge_v1


experiments = {
#    "test-pdm-clipL-uadfv": [
#         Config(
#             run_dir="runs/test-pdm/clipL",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=UADFV.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/clip_large.pth",
#             #from_exp="clip-from-gend-v2"
#         )
#     ],

#     "test-pdm-clipL-dfd": [
#         Config(
#             run_dir="runs/test-pdm/clipL",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=DFD.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/clip_large.pth",
#             #from_exp="clip-from-gend-v2"
#         )
#     ],
#     "test-pdm-clipL-dfdc": [
#         Config(
#             run_dir="runs/test-pdm/clipL",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=DFDC.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/clip_large.pth",
#             #from_exp="clip-from-gend-v2"
#         )
#     ],
#     "test-pdm-clipL-fsh": [
#         Config(
#             run_dir="runs/test-pdm/clipL",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=FSh.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/clip_large.pth",
#             #from_exp="clip-from-gend-v2"
#         )
#     ],
#     "test-pdm-clipL-cdfv2": [
#         Config(
#             run_dir="runs/test-pdm/clipL",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=CDFv2.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/clip_large.pth",
#             #from_exp="clip-from-gend-v2"
#         )
#     ],

#     "test-pdm-clipL-ffiw": [
#         Config(
#             run_dir="runs/test-pdm/clipL",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=FFIW.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/clip_large.pth",
#             #from_exp="clip-from-gend-v2"
#         )
#     ],

#     "test-pdm-clipL-kodf-normal": [
#         Config(
#             run_dir="runs/test-pdm/clipL",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=KoDF.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/clip_large.pth",
#             #from_exp="clip-from-gend-v2"
#         )
#     ],
#     "test-pdm-clipL-kodf-adv": [
#         Config(
#             run_dir="runs/test-pdm/clipL",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=KoDF.adversarial,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/clip_large.pth",
#             #from_exp="clip-from-gend-v2"
#         )
#     ],
#     "test-pdm-clipL-kodf-all": [
#         Config(
#             run_dir="runs/test-pdm/clipL",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=KoDF.test_all,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/clip_large.pth",
#             #from_exp="clip-from-gend-v2"
#         )
#     ],
#     "test-pdm-clipL-fakeavceleb": [
#         Config(
#             run_dir="runs/test-pdm/clipL",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=FakeAVCeleb.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/clip_large.pth",
#             #from_exp="clip-from-gend-v2"
#         )
#     ],
#     "test-pdm-clipL-dfdm": [
#         Config(
#             run_dir="runs/test-pdm/clipL",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=DFDM.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/clip_large.pth",
#             #from_exp="clip-from-gend-v2"
#         )
#     ],

#     "test-pdm-clipL-polyglotfake": [
#         Config(
#             run_dir="runs/test-pdm/clipL",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=PolyGlotFake.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/clip_large.pth",
#             #from_exp="clip-from-gend-v2"
#         )
#     ],
#     "test-pdm-clipL-idforge": [
#         Config(
#             run_dir="runs/test-pdm/clipL",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=IDForge_v1.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/clip_large.pth",
#             #from_exp="clip-from-gend-v2"
#         )
#     ],
#     "test-pdm-clipL-deepspeakv1": [
#         Config(
#             run_dir="runs/test-pdm/clipL",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=DeepSpeak_v1_1.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/clip_large.pth",
#             #from_exp="clip-from-gend-v2"
#         )
#     ],
#     "test-pdm-clipL-deepspeakv2": [
#         Config(
#             run_dir="runs/test-pdm/clipL",
#             backbone=C.Backbone.CLIP_L_14,
#             tst_files=DeepSpeak_v2.test,
#             batch_size=128,
#             mini_batch_size=128,
#             wandb=False,
#             devices=[0],
#             checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/clip_large.pth",
#             #from_exp="clip-from-gend-v2"
#         )
#     ],
    # "test-pdm-clipL-cdfv3": [
    #     Config(
    #         run_dir="runs/test-pdm/clipL",
    #         backbone=C.Backbone.CLIP_L_14,
    #         tst_files=CDFv3.test,
    #         batch_size=128,
    #         mini_batch_size=128,
    #         wandb=False,
    #         devices=[0],
    #         checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/clip_large.pth",
    #         #from_exp="clip-from-gend-v2"
    #     )
    #  ],
}