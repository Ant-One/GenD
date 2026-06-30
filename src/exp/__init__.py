#from .unused import examples, third_party, wacv_rebuttal_aug_robustness, wacv_rebuttal_paired_unpaired
from . import gend_clip, xception, clip

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
    **xception.experiments,
    **clip.experiments,
}
