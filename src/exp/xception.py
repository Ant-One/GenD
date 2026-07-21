from .. import config as C
from ..config import Config

from ..utils.files import Files, DF40, DF40Balanced, FSh, UADFV, DFD, DFDC, FFIW, FF, CDFv3, DeepSpeak_v1_1, DeepSpeak_v2, CDFv2, KoDF, FakeAVCeleb, DFDM, PolyGlotFake, IDForge_v1

experiments = {
   "test-pdm-xception-uadfv": [
        Config(
            run_dir="runs/test-pdm/xception",
            tst_files=UADFV.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/xception.pth",
            #from_exp="clip-from-gend-v2"
        )
    ],

    "test-pdm-xception-dfd": [
        Config(
            run_dir="runs/test-pdm/xception",
            tst_files=DFD.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/xception.pth",
            #from_exp="clip-from-gend-v2"
        )
    ],
    "test-pdm-xception-dfdc": [
        Config(
            run_dir="runs/test-pdm/xception",
            tst_files=DFDC.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/xception.pth",
            #from_exp="clip-from-gend-v2"
        )
    ],
    "test-pdm-xception-fsh": [
        Config(
            run_dir="runs/test-pdm/xception",
            tst_files=FSh.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/xception.pth",
            #from_exp="clip-from-gend-v2"
        )
    ],
    "test-pdm-xception-cdfv2": [
        Config(
            run_dir="runs/test-pdm/xception",
            tst_files=CDFv2.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/xception.pth",
            #from_exp="clip-from-gend-v2"
        )
    ],

    "test-pdm-xception-ffiw": [
        Config(
            run_dir="runs/test-pdm/xception",
            tst_files=FFIW.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/xception.pth",
            #from_exp="clip-from-gend-v2"
        )
    ],

    "test-pdm-xception-kodf-normal": [
        Config(
            run_dir="runs/test-pdm/xception",
            tst_files=KoDF.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/xception.pth",
            #from_exp="clip-from-gend-v2"
        )
    ],
    "test-pdm-xception-kodf-adv": [
        Config(
            run_dir="runs/test-pdm/xception",
            tst_files=KoDF.adversarial,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/xception.pth",
            #from_exp="clip-from-gend-v2"
        )
    ],
    "test-pdm-xception-kodf-all": [
        Config(
            run_dir="runs/test-pdm/xception",
            tst_files=KoDF.test_all,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/xception.pth",
            #from_exp="clip-from-gend-v2"
        )
    ],
    "test-pdm-xception-fakeavceleb": [
        Config(
            run_dir="runs/test-pdm/xception",
            tst_files=FakeAVCeleb.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/xception.pth",
            #from_exp="clip-from-gend-v2"
        )
    ],
    "test-pdm-xception-dfdm": [
        Config(
            run_dir="runs/test-pdm/xception",
            tst_files=DFDM.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/xception.pth",
            #from_exp="clip-from-gend-v2"
        )
    ],

    "test-pdm-xception-polyglotfake": [
        Config(
            run_dir="runs/test-pdm/xception",
            tst_files=PolyGlotFake.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/xception.pth",
            #from_exp="clip-from-gend-v2"
        )
    ],
    "test-pdm-xception-idforge": [
        Config(
            run_dir="runs/test-pdm/xception",
            tst_files=IDForge_v1.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/xception.pth",
            #from_exp="clip-from-gend-v2"
        )
    ],
    "test-pdm-xception-deepspeakv1": [
        Config(
            run_dir="runs/test-pdm/xception",
            tst_files=DeepSpeak_v1_1.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/xception.pth",
            #from_exp="clip-from-gend-v2"
        )
    ],
    "test-pdm-xception-deepspeakv2": [
        Config(
            run_dir="runs/test-pdm/xception",
            tst_files=DeepSpeak_v2.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/xception.pth",
            #from_exp="clip-from-gend-v2"
        )
    ],
    "test-pdm-xception-cdfv3": [
        Config(
            run_dir="runs/test-pdm/xception",
            tst_files=CDFv3.test,
            batch_size=128,
            mini_batch_size=128,
            wandb=False,
            devices=[0],
            checkpoint="/home/antoine/df-benchmarks/df-benchmark/weights/train_on_df40-all-ff/xception.pth",
            #from_exp="clip-from-gend-v2"
        )
     ],
}
