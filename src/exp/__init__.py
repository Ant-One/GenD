#from .unused import examples, third_party, wacv_rebuttal_aug_robustness, wacv_rebuttal_paired_unpaired
from . import clip_large, confdf, gend_clip, gend_cradio, gend_dino, gend_eva, gend_pe, gend_siglip, xception

#from .unused import (
#    wacv_rebuttal,
#)

experiments = {
    #**examples.experiments,
    #**third_party.experiments,
    #**wacv_rebuttal.experiments,
    #**wacv_rebuttal_paired_unpaired.experiments,
    #**wacv_rebuttal_aug_robustness.experiments,
    **gend_clip.experiments,
    **gend_dino.experiments,
    **gend_pe.experiments,
    **xception.experiments,
    **clip_large.experiments,
    **confdf.experiments,
    **gend_siglip.experiments,
    **gend_eva.experiments,
    **gend_cradio.experiments,
}
