from glob import glob


def find_run_dir(run_name: str) -> str:
    runs = list(glob(f"runs/*/{run_name}", recursive=True))
    if len(runs) == 0:
        raise FileNotFoundError(f"Directory for run '{run_name}' is not found")
    if len(runs) > 1:
        raise FileExistsError(f"Multiple directories found for run '{run_name}': {runs}")
    return runs[0]


# Extend list definition with map function
class Files(list):
    def __init__(self, *files):
        # If a single non-string iterable is passed, use it directly; otherwise treat args as items
        if len(files) == 1 and not isinstance(files[0], (str, bytes)):
            super().__init__(files[0])
        else:
            super().__init__(files)

    def map(self, func):
        return Files(map(func, self))

    def unique(self):
        return Files(sorted(set(self)))

    def cat(self, other):
        return Files(self + other)


class FF:
    """https://arxiv.org/abs/1901.08971"""

    class DF:
        train = Files(
            "config/datasets/train-from-gend/FF-1.3x-th0.5/FF-x1.3-th0.5-DF.txt",
            "config/datasets/train-from-gend/FF-1.3x-th0.5/FF-x1.3-th0.5-real.txt",
        )
        test = Files(
            "config/datasets/FF/test/DF.txt",
            "config/datasets/FF/test/real.txt",
        )

    class F2F:
        train = Files(
            "config/datasets/train-from-gend/FF-1.3x-th0.5/FF-x1.3-th0.5-F2F.txt",
            "config/datasets/train-from-gend/FF-1.3x-th0.5/FF-x1.3-th0.5-real.txt",
        )
        test = Files(
            "config/datasets/FF/test/F2F.txt",
            "config/datasets/FF/test/real.txt",
        )

    class FS:
        train = Files(
            "config/datasets/train-from-gend/FF-1.3x-th0.5/FF-x1.3-th0.5-FS.txt",
            "config/datasets/train-from-gend/FF-1.3x-th0.5/FF-x1.3-th0.5-real.txt",
        )
        test = Files(
            "config/datasets/FF/test/FS.txt",
            "config/datasets/FF/test/real.txt",
        )

    class NT:
        train = Files(
            "config/datasets/train-from-gend/FF-1.3x-th0.5/FF-x1.3-th0.5-NT.txt",
            "config/datasets/train-from-gend/FF-1.3x-th0.5/FF-x1.3-th0.5-real.txt",
        )
        test = Files(
            "config/datasets/FF/test/NT.txt",
            "config/datasets/FF/test/real.txt",
        )

    def to_train(f) -> str:
        return f.replace("/test/", "/train/")

    def to_val(f) -> str:
        return f.replace("/test/", "/val/")

    def to_x1_5(f) -> str:
        return f.replace("/FF/", "/FF-x1.5/")

    def to_x2(f) -> str:
        return f.replace("/FF/", "/FF-x2.0/")

    def to_rmbg_x1_5(f) -> str:
        return f.replace("/FF/", "/FF-rmbg-x1.5/")

    test = Files(DF.test + F2F.test + FS.test + NT.test).unique()
    train = Files(DF.train + F2F.train + FS.train + NT.train)
    val = test.map(to_val)


class CDFv2:
    """https://arxiv.org/abs/1909.12962"""

    test = Files(
        "config/datasets/CDFv2/test/Celeb-synthesis.txt",
        "config/datasets/CDFv2/test/YouTube-real.txt",
        "config/datasets/CDFv2/test/Celeb-real.txt",
    )

    # It is not an official validation set but generated from {all}\{test} files
    # using scripts/datasets/create_validation_set.py
    val = Files(
        "config/datasets/CDFv2/val/Celeb-synthesis.txt",
        "config/datasets/CDFv2/val/YouTube-real.txt",
        "config/datasets/CDFv2/val/Celeb-real.txt",
    )

    my_train = Files(
        "config/datasets/CDFv2/my-train/Celeb-synthesis.txt",
        "config/datasets/CDFv2/my-train/YouTube-real.txt",
        "config/datasets/CDFv2/my-train/Celeb-real.txt",
    )


class CDFv3:
    """https://arxiv.org/abs/2507.18015v1"""

    class FS:
        """Face-swap"""

        class CDFv2:
            test = Files(
                "config/datasets/CDFv3/test/Celeb-DF-v2.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class BlendFace:
            test = Files(
                "config/datasets/CDFv3/test/BlendFace.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class GHOST:
            test = Files(
                "config/datasets/CDFv3/test/GHOST.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class HifiFace:
            test = Files(
                "config/datasets/CDFv3/test/HifiFace.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class InSwapper:
            test = Files(
                "config/datasets/CDFv3/test/InSwapper.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class MobileFaceSwap:
            test = Files(
                "config/datasets/CDFv3/test/MobileFaceSwap.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class SimSwap:
            test = Files(
                "config/datasets/CDFv3/test/SimSwap.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class UniFace:
            test = Files(
                "config/datasets/CDFv3/test/UniFace.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        test = Files(
            CDFv2.test
            + BlendFace.test
            + GHOST.test
            + HifiFace.test
            + InSwapper.test
            + MobileFaceSwap.test
            + SimSwap.test
            + UniFace.test
        ).unique()

    class FR:
        """Face Reenectment"""

        class DaGAN:
            test = Files(
                "config/datasets/CDFv3/test/DaGAN.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class FSRT:
            test = Files(
                "config/datasets/CDFv3/test/FSRT.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class HyperReenact:
            test = Files(
                "config/datasets/CDFv3/test/HyperReenact.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class LIA:
            test = Files(
                "config/datasets/CDFv3/test/LIA.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class LivePortrait:
            test = Files(
                "config/datasets/CDFv3/test/LivePortrait.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class MCNET:
            test = Files(
                "config/datasets/CDFv3/test/MCNET.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class TPSMM:
            test = Files(
                "config/datasets/CDFv3/test/TPSMM.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        test = Files(
            DaGAN.test + FSRT.test + HyperReenact.test + LIA.test + LivePortrait.test + MCNET.test + TPSMM.test
        ).unique()

    class TF:
        """Talking Face"""

        class AniTalker:
            test = Files(
                "config/datasets/CDFv3/test/AniTalker.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class EchoMimic:
            test = Files(
                "config/datasets/CDFv3/test/EchoMimic.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class EDTalk:
            test = Files(
                "config/datasets/CDFv3/test/EDTalk.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class FLOAT:
            test = Files(
                "config/datasets/CDFv3/test/FLOAT.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class IP_LAP:
            test = Files(
                "config/datasets/CDFv3/test/IP_LAP.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class Real3DPortrait:
            test = Files(
                "config/datasets/CDFv3/test/Real3DPortrait.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        class SadTalker:
            test = Files(
                "config/datasets/CDFv3/test/SadTalker.txt",
                "config/datasets/CDFv3/test/Celeb-real.txt",
                "config/datasets/CDFv3/test/YouTube-real.txt",
            )

        test = Files(
            AniTalker.test
            + EchoMimic.test
            + EDTalk.test
            + FLOAT.test
            + IP_LAP.test
            + Real3DPortrait.test
            + SadTalker.test
        ).unique()

    def to_train(f) -> str:
        return f.replace("/test/", "/train/")

    def to_my_val(f) -> str:
        return f.replace("/test/", "/my-val/")

    def to_x1_5(f) -> str:
        return f.replace("/CDFv3/", "/CDFv3-x1.5/")

    def to_x2(f) -> str:
        return f.replace("/CDFv3/", "/CDFv3-x2.0/")

    def to_rmbg_x1_5(f) -> str:
        return f.replace("/CDFv3/", "/CDFv3-rmbg-x1.5/")

    def to_x1_3_th0_5_all(f) -> str:
        return f.replace("/CDFv3/", "/CDFv3-x1.3-th0.5-all/")

    test = Files(FS.test + FR.test + TF.test).unique()
    train = test.map(to_train)
    my_val = test.map(to_my_val)

    @classmethod
    def get_test_dict(cls) -> dict[str, list[str]]:
        return {
            "CDFv3": cls.test,
            "CDFv3-FS": cls.FS.test,
            "CDFv3-FR": cls.FR.test,
            "CDFv3-TF": cls.TF.test,
            "CDFv3-FS-CDFv2": cls.FS.CDFv2.test,
            "CDFv3-FS-BlendFace": cls.FS.BlendFace.test,
            "CDFv3-FS-GHOST": cls.FS.GHOST.test,
            "CDFv3-FS-HifiFace": cls.FS.HifiFace.test,
            "CDFv3-FS-InSwapper": cls.FS.InSwapper.test,
            "CDFv3-FS-MobileFaceSwap": cls.FS.MobileFaceSwap.test,
            "CDFv3-FS-SimSwap": cls.FS.SimSwap.test,
            "CDFv3-FS-UniFace": cls.FS.UniFace.test,
            "CDFv3-FR-DaGAN": cls.FR.DaGAN.test,
            "CDFv3-FR-FSRT": cls.FR.FSRT.test,
            "CDFv3-FR-HyperReenact": cls.FR.HyperReenact.test,
            "CDFv3-FR-LIA": cls.FR.LIA.test,
            "CDFv3-FR-LivePortrait": cls.FR.LivePortrait.test,
            "CDFv3-FR-MCNET": cls.FR.MCNET.test,
            "CDFv3-FR-TPSMM": cls.FR.TPSMM.test,
            "CDFv3-TF-AniTalker": cls.TF.AniTalker.test,
            "CDFv3-TF-EchoMimic": cls.TF.EchoMimic.test,
            "CDFv3-TF-EDTalk": cls.TF.EDTalk.test,
            "CDFv3-TF-FLOAT": cls.TF.FLOAT.test,
            "CDFv3-TF-IP_LAP": cls.TF.IP_LAP.test,
            "CDFv3-TF-Real3DPortrait": cls.TF.Real3DPortrait.test,
            "CDFv3-TF-SadTalker": cls.TF.SadTalker.test,
        }


class DFD:
    test = Files(
        "/home/antoine/GenD/config/datasets/test-from-gend/DFD_fake.txt",
        "/home/antoine/GenD/config/datasets/test-from-gend/DFD_real.txt",
    )


class DFDC:
    test = Files(
        "/home/antoine/GenD/config/datasets/test-from-gend/DFDC_fake.txt",
        "/home/antoine/GenD/config/datasets/test-from-gend/DFDC_real.txt",
    )


class FSh:
    """
    FSh: https://github.com/maum-ai/faceshifter
    FF++: https://github.com/ondyari/FaceForensics
    """

    test = Files(
        "/home/antoine/GenD/config/datasets/test-from-gend/FSh_test_real.txt",
        "/home/antoine/GenD/config/datasets/test-from-gend/FSh_test_fake.txt",
    )


class UADFV:
    """https://arxiv.org/abs/1806.02877"""

    test = Files(
        "/home/antoine/GenD/config/datasets/test-from-gend/UADFV_fake.txt",
        "/home/antoine/GenD/config/datasets/test-from-gend/UADFV_real.txt",
    )


class DFDM:
    """https://arxiv.org/abs/2202.12951"""

    test = Files(
        "config/datasets/DFDM/all/dfaker.txt",
        "config/datasets/DFDM/all/dfl.txt",
        "config/datasets/DFDM/all/iae.txt",
        "config/datasets/DFDM/all/lightweight.txt",
        "config/datasets/CDFv2/all/Celeb-real.txt",
    )


class FFIW:
    """https://arxiv.org/abs/2103.16076"""

    test = Files(
        "/home/antoine/GenD/config/datasets/test-from-gend/FFIW_fake.txt",
        "/home/antoine/GenD/config/datasets/test-from-gend/FFIW_real.txt",
    )

    val = Files(
        "config/datasets/FFIW/val-fake.txt",
        "config/datasets/FFIW/val-real.txt",
    )

    train = Files(
        "config/datasets/FFIW/train-fake.txt",
        "config/datasets/FFIW/train-real.txt",
    )

    # My subsets of FFIW
    train_subset_1024 = Files(
        "config/datasets/FFIW/subsets/train-fake-subset-1024.txt",
        "config/datasets/FFIW/subsets/train-real-subset-1024.txt",
    )

    # My subsets of FFIW created using scripts/datasets/FFIW/create_FFIW_subset.py
    train_subset_2048 = Files(
        "config/datasets/FFIW/subset-2048/train-fake.txt",
        "config/datasets/FFIW/subset-2048/train-real.txt",
    )


class DeepSpeak_v1_1:
    """https://arxiv.org/abs/2408.05366"""

    test = Files(
        "config/datasets/DeepSpeak-1.1/test/test-facefusion_gan.txt",
        "config/datasets/DeepSpeak-1.1/test/test-facefusion_live.txt",
        "config/datasets/DeepSpeak-1.1/test/test-facefusion.txt",
        "config/datasets/DeepSpeak-1.1/test/test-real.txt",
        "config/datasets/DeepSpeak-1.1/test/test-retalking.txt",
        "config/datasets/DeepSpeak-1.1/test/test-wav2lip.txt",
    )

    train = test.map(lambda x: x.replace("/test/test-", "/train/train-"))

    # DeepSpeak-1.1 has a train folder. my-val is sampled from train
    my_val = test.map(lambda x: x.replace("/test/test-", "/my-val/val-"))

    # DeepSpeak-1.1 has a train folder. my-val is sampled from train, my-train is train \ my-val
    my_train = test.map(lambda x: x.replace("/test/test-", "/my-train/train-"))


class DeepSpeak_v2:
    """https://arxiv.org/abs/2408.05366"""

    test = Files(
        "config/datasets/DeepSpeak-2.0/test/test-diff2lip.txt",
        "config/datasets/DeepSpeak-2.0/test/test-facefusion.txt",
        "config/datasets/DeepSpeak-2.0/test/test-hellomeme.txt",
        "config/datasets/DeepSpeak-2.0/test/test-latentsync.txt",
        "config/datasets/DeepSpeak-2.0/test/test-liveportrait.txt",
        "config/datasets/DeepSpeak-2.0/test/test-memo.txt",
        "config/datasets/DeepSpeak-2.0/test/test-real.txt",
    )

    train = test.map(lambda x: x.replace("/test/test-", "/train/train-"))

    # DeepSpeak-2.0 has a train folder. my-val is sampled from train
    my_val = test.map(lambda x: x.replace("/test/test-", "/my-val/val-"))

    # DeepSpeak-2.0 has a train folder. my-val is sampled from train, my-train is train \ my-val
    my_train = test.map(lambda x: x.replace("/test/test-", "/my-train/train-"))


class KoDF:
    """https://arxiv.org/abs/2103.10094"""

    test = Files(
        "config/datasets/KoDF/real.txt",
        "config/datasets/KoDF/fake-audio-driven.txt",
        "config/datasets/KoDF/fake-dffs.txt",
        "config/datasets/KoDF/fake-dfl.txt",
        "config/datasets/KoDF/fake-fo.txt",
        "config/datasets/KoDF/fake-fsgan.txt",
    )

    adversarial = Files(
        "config/datasets/KoDF/fake-adv.txt",
        "config/datasets/KoDF/real-adv.txt",
    )


class FaceFusion:
    """Dataset created by VRG group"""

    class FF:
        train = Files(
            "config/datasets/FaceFusion/train/ff_inswapper_128_fp16.txt",
        )

    class CDF:
        test = Files(
            "config/datasets/FaceFusion/test/cdf_hififace_unofficial_256.txt",
            "config/datasets/FaceFusion/test/cdf_inswapper_128_fp16.txt",
            "config/datasets/CDFv2/test/YouTube-real.txt",
            "config/datasets/CDFv2/test/Celeb-real.txt",
        )


class VRG:
    """Dataset created by VRG group"""

    class CSFD:
        files = Files(
            "config/datasets/CSFD/real.txt",
        )


class AVSpeech:
    files = Files(
        "config/datasets/AVSpeech/real.txt",
    )


class FakeAVCeleb:
    """https://arxiv.org/abs/2108.05080"""

    test = Files(
        "config/datasets/FakeAVCeleb/FV-FA-faceswap-wav2lip.txt",
        "config/datasets/FakeAVCeleb/FV-FA-fsgan-wav2lip.txt",
        "config/datasets/FakeAVCeleb/FV-FA-wav2lip.txt",
        "config/datasets/FakeAVCeleb/FV-RA-faceswap.txt",
        "config/datasets/FakeAVCeleb/FV-RA-fsgan.txt",
        "config/datasets/FakeAVCeleb/FV-RA-wav2lip.txt",
        "config/datasets/FakeAVCeleb/RV-RA-real.txt",
    )

    class FV_RA_WL:
        test = Files(
            "config/datasets/FakeAVCeleb/FV-RA-wav2lip.txt",
            "config/datasets/FakeAVCeleb/RV-RA-real.txt",
        )

    class FV_FA_FS:
        test = Files(
            "config/datasets/FakeAVCeleb/FV-FA-faceswap-wav2lip.txt",
            "config/datasets/FakeAVCeleb/RV-RA-real.txt",
        )

    class FV_FA_GAN:
        test = Files(
            "config/datasets/FakeAVCeleb/FV-FA-fsgan-wav2lip.txt",
            "config/datasets/FakeAVCeleb/RV-RA-real.txt",
        )

    class FV_FA_WL:
        test = Files(
            "config/datasets/FakeAVCeleb/FV-FA-wav2lip.txt",
            "config/datasets/FakeAVCeleb/RV-RA-real.txt",
        )


class PolyGlotFake:
    """https://arxiv.org/abs/2405.08838"""

    test = Files(
        "config/datasets/PolyGlotFake/real-ar.txt",
        "config/datasets/PolyGlotFake/real-en.txt",
        "config/datasets/PolyGlotFake/real-es.txt",
        "config/datasets/PolyGlotFake/real-fr.txt",
        "config/datasets/PolyGlotFake/real-ja.txt",
        "config/datasets/PolyGlotFake/real-ru.txt",
        "config/datasets/PolyGlotFake/real-zh.txt",
        "config/datasets/PolyGlotFake/ar2en_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ar2en_video_retalking.txt",
        "config/datasets/PolyGlotFake/ar2es_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ar2es_video_retalking.txt",
        "config/datasets/PolyGlotFake/ar2fr_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ar2fr_video_retalking.txt",
        "config/datasets/PolyGlotFake/ar2ja_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ar2ja_video_retalking.txt",
        "config/datasets/PolyGlotFake/ar2ru_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ar2ru_video_retalking.txt",
        "config/datasets/PolyGlotFake/ar2zh_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ar2zh_video_retalking.txt",
        "config/datasets/PolyGlotFake/en2ar_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/en2ar_video_retalking.txt",
        "config/datasets/PolyGlotFake/en2es_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/en2es_video_retalking.txt",
        "config/datasets/PolyGlotFake/en2fr_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/en2fr_video_retalking.txt",
        "config/datasets/PolyGlotFake/en2ja_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/en2ja_video_retalking.txt",
        "config/datasets/PolyGlotFake/en2ru_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/en2ru_video_retalking.txt",
        "config/datasets/PolyGlotFake/en2zh_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/en2zh_video_retalking.txt",
        "config/datasets/PolyGlotFake/es2ar_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/es2ar_video_retalking.txt",
        "config/datasets/PolyGlotFake/es2en_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/es2en_video_retalking.txt",
        "config/datasets/PolyGlotFake/es2fr_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/es2fr_video_retalking.txt",
        "config/datasets/PolyGlotFake/es2ja_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/es2ja_video_retalking.txt",
        "config/datasets/PolyGlotFake/es2ru_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/es2ru_video_retalking.txt",
        "config/datasets/PolyGlotFake/es2zh_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/es2zh_video_retalking.txt",
        "config/datasets/PolyGlotFake/fr2ar_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/fr2ar_video_retalking.txt",
        "config/datasets/PolyGlotFake/fr2en_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/fr2en_video_retalking.txt",
        "config/datasets/PolyGlotFake/fr2es_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/fr2es_video_retalking.txt",
        "config/datasets/PolyGlotFake/fr2ja_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/fr2ja_video_retalking.txt",
        "config/datasets/PolyGlotFake/fr2ru_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/fr2ru_video_retalking.txt",
        "config/datasets/PolyGlotFake/fr2zh_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/fr2zh_video_retalking.txt",
        "config/datasets/PolyGlotFake/ja2ar_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ja2ar_video_retalking.txt",
        "config/datasets/PolyGlotFake/ja2en_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ja2en_video_retalking.txt",
        "config/datasets/PolyGlotFake/ja2es_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ja2es_video_retalking.txt",
        "config/datasets/PolyGlotFake/ja2fr_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ja2fr_video_retalking.txt",
        "config/datasets/PolyGlotFake/ja2ru_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ja2ru_video_retalking.txt",
        "config/datasets/PolyGlotFake/ja2zh_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ja2zh_video_retalking.txt",
        "config/datasets/PolyGlotFake/ru2ar_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ru2ar_video_retalking.txt",
        "config/datasets/PolyGlotFake/ru2en_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ru2en_video_retalking.txt",
        "config/datasets/PolyGlotFake/ru2es_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ru2es_video_retalking.txt",
        "config/datasets/PolyGlotFake/ru2fr_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ru2fr_video_retalking.txt",
        "config/datasets/PolyGlotFake/ru2ja_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ru2zh_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/ru2zh_video_retalking.txt",
        "config/datasets/PolyGlotFake/zh2ar_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/zh2ar_video_retalking.txt",
        "config/datasets/PolyGlotFake/zh2en_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/zh2en_video_retalking.txt",
        "config/datasets/PolyGlotFake/zh2es_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/zh2es_video_retalking.txt",
        "config/datasets/PolyGlotFake/zh2fr_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/zh2fr_video_retalking.txt",
        "config/datasets/PolyGlotFake/zh2ja_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/zh2ru_Wav2Lip.txt",
        "config/datasets/PolyGlotFake/zh2ru_video_retalking.txt",
    )


class IDForge_v1:
    """https://arxiv.org/abs/2401.11764"""

    train = Files(
        "config/datasets/IDForge-v1/train/train-face_tts_infoswap.txt",
        "config/datasets/IDForge-v1/train/train-face_tts_roop.txt",
        "config/datasets/IDForge-v1/train/train-face_tts_simswap.txt",
        "config/datasets/IDForge-v1/train/train-real.txt",
    )

    val = Files(
        "config/datasets/IDForge-v1/val/val-face_tts_infoswap.txt",
        "config/datasets/IDForge-v1/val/val-face_tts_roop.txt",
        "config/datasets/IDForge-v1/val/val-face_tts_simswap.txt",
        "config/datasets/IDForge-v1/val/val-real.txt",
    )

    test = Files(
        "config/datasets/IDForge-v1/test/test-face_tts_infoswap.txt",
        "config/datasets/IDForge-v1/test/test-face_tts_roop.txt",
        "config/datasets/IDForge-v1/test/test-face_tts_simswap.txt",
        "config/datasets/IDForge-v1/test/test-real.txt",
    )


class DF40:
    """https://arxiv.org/abs/2406.13495"""

    class CDF:
        # FS
        class BlendFace:
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/test/fake/blendface_cdf.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/Celeb-real.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/YouTube-real.txt",
            )
        class E4S:
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/test/fake/e4s_cdf.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/Celeb-real.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/YouTube-real.txt",
            )
        class Facedancer:
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/test/fake/facedancer_cdf.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/Celeb-real.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/YouTube-real.txt",
            )

        class Faceswap:
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/test/fake/faceswap_cdf.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/Celeb-real.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/YouTube-real.txt",
            )
        class FSGan:
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/test/fake/fsgan_cdf.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/Celeb-real.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/YouTube-real.txt",
            )
        class InSwap:
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/test/fake/inswap_cdf.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/Celeb-real.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/YouTube-real.txt",
            )
        class MobileSwap:
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/test/fake/mobileswap_cdf.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/Celeb-real.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/YouTube-real.txt",
            )
        class SimSwap:
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/test/fake/simswap_cdf.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/Celeb-real.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/YouTube-real.txt",
            )
        class Uniface:
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/test/fake/uniface_cdf.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/Celeb-real.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/YouTube-real.txt",
            )

        val_fs = Files(
                BlendFace.val + E4S.val + Facedancer.val + Faceswap.val + FSGan.val + 
                InSwap.val + MobileSwap.val + SimSwap.val + Uniface.val
            ).unique()
        
        # FR

        # class MRAA:
        #     test = Files(
        #         "/home/antoine/balanced_df40_for_gend/df40/test/fake/mraa_cdf.txt",
        #         "/home/antoine/balanced_df40_for_gend/df40/test/real/Celeb-real.txt",
        #         "/home/antoine/balanced_df40_for_gend/df40/test/real/YouTube-real.txt",
        #     )

        class DANET:
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/test/fake/danet_cdf.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/Celeb-real.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/YouTube-real.txt",
            )

        class FaceVid2Vid:
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/test/fake/facevid2vid_cdf.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/Celeb-real.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/YouTube-real.txt",
            )
        class Fomm:
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/test/fake/fomm_cdf.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/Celeb-real.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/YouTube-real.txt",
            )

        class Hyperreenact:
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/test/fake/hyperreenact_cdf.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/Celeb-real.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/YouTube-real.txt",
            )

        class Lia:
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/test/fake/lia_cdf.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/Celeb-real.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/YouTube-real.txt",
            )

        class MCNet:
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/test/fake/mcnet_cdf.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/Celeb-real.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/YouTube-real.txt",
            )

        class OneShotFree:
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/test/fake/oneshotfree_cdf.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/Celeb-real.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/YouTube-real.txt",
            )
        
        class Pirender:
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/test/fake/pirender_cdf.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/Celeb-real.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/YouTube-real.txt",
            )

        class SadTalker:
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/test/fake/sadtalker_cdf.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/Celeb-real.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/YouTube-real.txt",
            )

        class TPSM:
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/test/fake/tpsm_cdf.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/Celeb-real.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/YouTube-real.txt",
            )

        class Wav2Lip:
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/test/fake/wav2lip_cdf.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/Celeb-real.txt",
                "/home/antoine/balanced_df40_for_gend/df40/test/real/YouTube-real.txt",
            )

        val_fr = Files(
                DANET.val + FaceVid2Vid.val + Fomm.val + Hyperreenact.val + Lia.val + MCNet.val + OneShotFree.val + Pirender.val + SadTalker.val + TPSM.val + Wav2Lip.val
            ).unique()
        
     # EFS

        class DiT:
            val = Files(
                "config/datasets/DF40/test/dit_cdf.txt",
                "config/datasets/CDFv2/test/YouTube-real.txt",
                "config/datasets/CDFv2/test/Celeb-real.txt",
            )
        class SiT:
            val = Files(
                "config/datasets/DF40/test/sit_cdf.txt",
                "config/datasets/CDFv2/test/YouTube-real.txt",
                "config/datasets/CDFv2/test/Celeb-real.txt",
            )
        class StyleGAN2:
            val = Files(
                "config/datasets/DF40/test/stylegan2_cdf.txt",
                "config/datasets/CDFv2/test/YouTube-real.txt",
                "config/datasets/CDFv2/test/Celeb-real.txt",
            )
        class StyleGAN3:
            val = Files(
                "config/datasets/DF40/test/stylegan3_cdf.txt",
                "config/datasets/CDFv2/test/YouTube-real.txt",
                "config/datasets/CDFv2/test/Celeb-real.txt",
            )
        class StyleGANXL:
            val = Files(
                "config/datasets/DF40/test/styleganxl_cdf.txt",
                "config/datasets/CDFv2/test/YouTube-real.txt",
                "config/datasets/CDFv2/test/Celeb-real.txt",
            )
        class VQGAN:
            val = Files(
                "config/datasets/DF40/test/vqgan_cdf.txt",
                "config/datasets/CDFv2/test/YouTube-real.txt",
                "config/datasets/CDFv2/test/Celeb-real.txt",
            )
        class DDIM:
            val = Files(
                "config/datasets/DF40/test/ddim_cdf.txt",
                "config/datasets/CDFv2/test/YouTube-real.txt",
                "config/datasets/CDFv2/test/Celeb-real.txt",
            )
        class Pixart:
            val = Files(
                "config/datasets/DF40/test/pixart_cdf.txt",
                "config/datasets/CDFv2/test/YouTube-real.txt",
                "config/datasets/CDFv2/test/Celeb-real.txt",
            )
        class RDDM:
            val = Files(
                "config/datasets/DF40/test/rddm_cdf.txt",
                "config/datasets/CDFv2/test/YouTube-real.txt",
                "config/datasets/CDFv2/test/Celeb-real.txt",
            )
        class SD21:
            val = Files(
                "config/datasets/DF40/test/sd21_cdf.txt",
                "config/datasets/CDFv2/test/YouTube-real.txt",
                "config/datasets/CDFv2/test/Celeb-real.txt",
            )

        val_efs = Files(DiT.val + SiT.val + StyleGAN2.val + StyleGAN3.val + VQGAN.val + DDIM.val + Pixart.val + RDDM.val + SD21.val).unique()

        val_cdf = Files(val_fs + val_fr)

        # @classmethod
        # def get_test_dict(cls) -> dict[str, list[str]]:
        #     return {
        #         "DF40-CDF-BlendFace": cls.BlendFace.test,
        #         "DF40-CDF-E4S": cls.E4S.test,
        #         "DF40-CDF-Facedancer": cls.Facedancer.test,
        #         "DF40-CDF-FSGan": cls.FSGan.test,
        #         "DF40-CDF-InSwap": cls.InSwap.test,
        #         "DF40-CDF-MobileSwap": cls.MobileSwap.test,
        #         "DF40-CDF-SimSwap": cls.SimSwap.test,
        #         "DF40-CDF-Uniface": cls.Uniface.test,
        #         "DF40-CDF-All": cls.test,
        #     }


class DF40Balanced:
    """https://arxiv.org/abs/2406.13495"""

    # Train and val are combined below

    class FF:
        
        # FS
        class BlendFace:
            train = Files(
                "/home/antoine/balanced_df40_for_gend/df40/train/fake/blendface_ff_fake_balanced_train.txt",
                "/home/antoine/balanced_df40_for_gend/df40/train/real/blendface_ff_real_balanced_train.txt",
            )
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/val/fake/blendface_ff_fake_balanced_val.txt",
                "/home/antoine/balanced_df40_for_gend/df40/val/real/blendface_ff_real_balanced_val.txt",
            )

        class E4S:
            train = Files(
                "/home/antoine/balanced_df40_for_gend/df40/train/fake/e4s_ff_fake_balanced_train.txt",
                "/home/antoine/balanced_df40_for_gend/df40/train/real/e4s_ff_real_balanced_train.txt",
            )
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/val/fake/e4s_ff_fake_balanced_val.txt",
                "/home/antoine/balanced_df40_for_gend/df40/val/real/e4s_ff_real_balanced_val.txt",
            )
        class Facedancer:
            train = Files(
                "/home/antoine/balanced_df40_for_gend/df40/train/fake/facedancer_ff_fake_balanced_train.txt",
                "/home/antoine/balanced_df40_for_gend/df40/train/real/facedancer_ff_real_balanced_train.txt",
            )
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/val/fake/facedancer_ff_fake_balanced_val.txt",
                "/home/antoine/balanced_df40_for_gend/df40/val/real/facedancer_ff_real_balanced_val.txt",
            )

        class Faceswap:
            train = Files(
                "/home/antoine/balanced_df40_for_gend/df40/train/fake/faceswap_ff_fake_balanced_train.txt",
                "/home/antoine/balanced_df40_for_gend/df40/train/real/faceswap_ff_real_balanced_train.txt",
            )
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/val/fake/faceswap_ff_fake_balanced_val.txt",
                "/home/antoine/balanced_df40_for_gend/df40/val/real/faceswap_ff_real_balanced_val.txt",
            )
        class FSGan:
            train = Files(
                "/home/antoine/balanced_df40_for_gend/df40/train/fake/fsgan_ff_fake_balanced_train.txt",
                "/home/antoine/balanced_df40_for_gend/df40/train/real/fsgan_ff_real_balanced_train.txt",
            )
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/val/fake/fsgan_ff_fake_balanced_val.txt",
                "/home/antoine/balanced_df40_for_gend/df40/val/real/fsgan_ff_real_balanced_val.txt",
            )
        class InSwap:
            train = Files(
                "/home/antoine/balanced_df40_for_gend/df40/train/fake/inswap_ff_fake_balanced_train.txt",
                "/home/antoine/balanced_df40_for_gend/df40/train/real/inswap_ff_real_balanced_train.txt",
            )
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/val/fake/inswap_ff_fake_balanced_val.txt",
                "/home/antoine/balanced_df40_for_gend/df40/val/real/inswap_ff_real_balanced_val.txt",
            )
        class MobileSwap:
            train = Files(
                "/home/antoine/balanced_df40_for_gend/df40/train/fake/mobileswap_ff_fake_balanced_train.txt",
                "/home/antoine/balanced_df40_for_gend/df40/train/real/mobileswap_ff_real_balanced_train.txt",
            )
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/val/fake/mobileswap_ff_fake_balanced_val.txt",
                "/home/antoine/balanced_df40_for_gend/df40/val/real/mobileswap_ff_real_balanced_val.txt",
            )
        class SimSwap:
            train = Files(
                "/home/antoine/balanced_df40_for_gend/df40/train/fake/simswap_ff_fake_balanced_train.txt",
                "/home/antoine/balanced_df40_for_gend/df40/train/real/simswap_ff_real_balanced_train.txt",
            )
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/val/fake/simswap_ff_fake_balanced_val.txt",
                "/home/antoine/balanced_df40_for_gend/df40/val/real/simswap_ff_real_balanced_val.txt",
            )
        class Uniface:
            train = Files(
                "/home/antoine/balanced_df40_for_gend/df40/train/fake/uniface_ff_fake_balanced_train.txt",
                "/home/antoine/balanced_df40_for_gend/df40/train/real/uniface_ff_real_balanced_train.txt",
            )
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/val/fake/uniface_ff_fake_balanced_val.txt",
                "/home/antoine/balanced_df40_for_gend/df40/val/real/uniface_ff_real_balanced_val.txt",
            )

        train_fs = Files(BlendFace.train + E4S.train + Facedancer.train + Faceswap.train + FSGan.train + InSwap.train + MobileSwap.train + SimSwap.train + Uniface.train)
        val_fs = Files(BlendFace.val + E4S.val + Facedancer.val + Faceswap.val + FSGan.val + InSwap.val + MobileSwap.val + SimSwap.val + Uniface.val)

        #FR

        class DANET:
            train = Files(
                "/home/antoine/balanced_df40_for_gend/df40/train/fake/danet_ff_fake_balanced_train.txt",
                "/home/antoine/balanced_df40_for_gend/df40/train/real/danet_ff_real_balanced_train.txt",
            )
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/val/fake/danet_ff_fake_balanced_val.txt",
                "/home/antoine/balanced_df40_for_gend/df40/val/real/danet_ff_real_balanced_val.txt",
            )

        class FaceVid2Vid:
            train = Files(
                "/home/antoine/balanced_df40_for_gend/df40/train/fake/facevid2vid_ff_fake_balanced_train.txt",
                "/home/antoine/balanced_df40_for_gend/df40/train/real/facevid2vid_ff_real_balanced_train.txt",
            )
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/val/fake/facevid2vid_ff_fake_balanced_val.txt",
                "/home/antoine/balanced_df40_for_gend/df40/val/real/facevid2vid_ff_real_balanced_val.txt",
            )

        class Fomm:
            train = Files(
                "/home/antoine/balanced_df40_for_gend/df40/train/fake/fomm_ff_fake_balanced_train.txt",
                "/home/antoine/balanced_df40_for_gend/df40/train/real/fomm_ff_real_balanced_train.txt",
            )
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/val/fake/fomm_ff_fake_balanced_val.txt",
                "/home/antoine/balanced_df40_for_gend/df40/val/real/fomm_ff_real_balanced_val.txt",
            )

        class Hyperreenact:
            train = Files(
                "/home/antoine/balanced_df40_for_gend/df40/train/fake/hyperreenact_ff_fake_balanced_train.txt",
                "/home/antoine/balanced_df40_for_gend/df40/train/real/hyperreenact_ff_real_balanced_train.txt",
            )
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/val/fake/hyperreenact_ff_fake_balanced_val.txt",
                "/home/antoine/balanced_df40_for_gend/df40/val/real/hyperreenact_ff_real_balanced_val.txt",
            )

        class Lia:
            train = Files(
                "/home/antoine/balanced_df40_for_gend/df40/train/fake/lia_ff_fake_balanced_train.txt",
                "/home/antoine/balanced_df40_for_gend/df40/train/real/lia_ff_real_balanced_train.txt",
            )
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/val/fake/lia_ff_fake_balanced_val.txt",
                "/home/antoine/balanced_df40_for_gend/df40/val/real/lia_ff_real_balanced_val.txt",
            )

        class MCNet:
            train = Files(
                "/home/antoine/balanced_df40_for_gend/df40/train/fake/mcnet_ff_fake_balanced_train.txt",
                "/home/antoine/balanced_df40_for_gend/df40/train/real/mcnet_ff_real_balanced_train.txt",
            )
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/val/fake/mcnet_ff_fake_balanced_val.txt",
                "/home/antoine/balanced_df40_for_gend/df40/val/real/mcnet_ff_real_balanced_val.txt",
            )

        class OneShotFree:
            train = Files(
                "/home/antoine/balanced_df40_for_gend/df40/train/fake/one_shot_free_ff_fake_balanced_train.txt",
                "/home/antoine/balanced_df40_for_gend/df40/train/real/one_shot_free_ff_real_balanced_train.txt",
            )
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/val/fake/one_shot_free_ff_fake_balanced_val.txt",
                "/home/antoine/balanced_df40_for_gend/df40/val/real/one_shot_free_ff_real_balanced_val.txt",
            )

        class Pirender:
            train = Files(
                "/home/antoine/balanced_df40_for_gend/df40/train/fake/pirender_ff_fake_balanced_train.txt",
                "/home/antoine/balanced_df40_for_gend/df40/train/real/pirender_ff_real_balanced_train.txt",
            )
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/val/fake/pirender_ff_fake_balanced_val.txt",
                "/home/antoine/balanced_df40_for_gend/df40/val/real/pirender_ff_real_balanced_val.txt",
            )

        class SadTalker:
            train = Files(
                "/home/antoine/balanced_df40_for_gend/df40/train/fake/sadtalker_ff_fake_balanced_train.txt",
                "/home/antoine/balanced_df40_for_gend/df40/train/real/sadtalker_ff_real_balanced_train.txt",
            )
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/val/fake/sadtalker_ff_fake_balanced_val.txt",
                "/home/antoine/balanced_df40_for_gend/df40/val/real/sadtalker_ff_real_balanced_val.txt",
            )

        class TPSM:
            train = Files(
                "/home/antoine/balanced_df40_for_gend/df40/train/fake/tpsm_ff_fake_balanced_train.txt",
                "/home/antoine/balanced_df40_for_gend/df40/train/real/tpsm_ff_real_balanced_train.txt",
            )
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/val/fake/tpsm_ff_fake_balanced_val.txt",
                "/home/antoine/balanced_df40_for_gend/df40/val/real/tpsm_ff_real_balanced_val.txt",
            )

        class Wav2Lip:
            train = Files(
                "/home/antoine/balanced_df40_for_gend/df40/train/fake/wav2lip_ff_fake_balanced_train.txt",
                "/home/antoine/balanced_df40_for_gend/df40/train/real/wav2lip_ff_real_balanced_train.txt",
            )
            val = Files(
                "/home/antoine/balanced_df40_for_gend/df40/val/fake/wav2lip_ff_fake_balanced_val.txt",
                "/home/antoine/balanced_df40_for_gend/df40/val/real/wav2lip_ff_real_balanced_val.txt",
            )

        train_fr = Files(DANET.train + FaceVid2Vid.train + Fomm.train + Hyperreenact.train + Lia.train + MCNet.train + OneShotFree.train + Pirender.train + SadTalker.train + TPSM.train + Wav2Lip.train)
        val_fr = Files(DANET.val + FaceVid2Vid.val + Fomm.val + Hyperreenact.val + Lia.val + MCNet.val + OneShotFree.val + Pirender.val + SadTalker.val + TPSM.val + Wav2Lip.val)

        train_ff = Files(train_fs + train_fr)
        val_ff = Files(val_fs + val_fr)

        train_val_combined_ff = Files(train_ff + val_ff)
        

class FFv2:
    """
    FaceFusion v2 dataset created by VRG group
    """

    class FF:
        train = Files(
            "config/datasets/FF/train/real.txt",
            "config/datasets/FFv2/train/FF_blendswap_256.txt",
            "config/datasets/FFv2/train/FF_ghost_1_256.txt",
            # "config/datasets/FFv2/train/FF_ghost_2_256.txt",
            # "config/datasets/FFv2/train/FF_ghost_3_256.txt",
            "config/datasets/FFv2/train/FF_hififace_unofficial_256.txt",
            "config/datasets/FFv2/train/FF_hyperswap_1a_256.txt",
            # "config/datasets/FFv2/train/FF_hyperswap_1c_256.txt",
            "config/datasets/FFv2/train/FF_inswapper_128_fp16.txt",
            # "config/datasets/FFv2/train/FF_inswapper_128.txt",
            "config/datasets/FFv2/train/FF_simswap_256.txt",
            # "config/datasets/FFv2/train/FF_simswap_unofficial_512.txt",
            "config/datasets/FFv2/train/FF_uniface_256.txt",
        )

    class SS:
        train = Files(
            "config/datasets/FF/train/real.txt",
            "config/datasets/FFv2/train/SS_blendswap_256.txt",
            "config/datasets/FFv2/train/SS_ghost_1_256.txt",
            # "config/datasets/FFv2/train/SS_ghost_2_256.txt",
            # "config/datasets/FFv2/train/SS_ghost_3_256.txt",
            "config/datasets/FFv2/train/SS_hififace_unofficial_256.txt",
            "config/datasets/FFv2/train/SS_hyperswap_1a_256.txt",
            # "config/datasets/FFv2/train/SS_hyperswap_1c_256.txt",
            "config/datasets/FFv2/train/SS_inswapper_128_fp16.txt",
            # "config/datasets/FFv2/train/SS_inswapper_128.txt",
            "config/datasets/FFv2/train/SS_simswap_256.txt",
            # "config/datasets/FFv2/train/SS_simswap_unofficial_512.txt",
            "config/datasets/FFv2/train/SS_uniface_256.txt",
        )


if __name__ == "__main__":
    import pandas as pd

    def get_video(file_path: str) -> str:
        return file_path.split("/")[-2]

    val_files = [
        # *CDFv3.test.map(CDFv3.to_train).map(CDFv3.to_x1_5),
        # *FF.train.map(FF.to_x1_5),
        # *FF.train.map(FF.to_rmbg_x1_5),
        # *CDFv3.train.map(CDFv3.to_x1_3_th0_5_all),
        # *DeepSpeak_v1_1.train
        # *DeepSpeak_v2.train.cat(DeepSpeak_v1_1.train)
        *FF.train.map(lambda x: x.replace("/FF/", "/FF-x1.3-th0.5-all/subset/1st-frame/")),
        *DeepSpeak_v1_1.train.map(lambda x: x.replace("/DeepSpeak-1.1/", "/DeepSpeak-1.1/subset/1st-frame/")),
        *DeepSpeak_v2.train.map(lambda x: x.replace("/DeepSpeak-2.0/", "/DeepSpeak-2.0/subset/1st-frame/")),
        *FFIW.train.map(lambda x: x.replace("/FFIW/", "/FFIW/subset/1st-frame/")),
    ]

    total_videos = 0
    for file in val_files:
        # read with pandas
        df = pd.read_csv(file, names=["files"])

        df["video"] = df["files"].apply(lambda x: get_video(x))

        # unique values
        unique_videos = df["video"].unique()

        print(f"Unique videos in {file} : {len(unique_videos)}")

        total_videos += len(unique_videos)

    print(f"Total unique videos: {total_videos}")
