from .SubNets.FeatureNets import BERTEncoder
from .FusionNets.MULT import MULT

text_backbones_map = {
                    'bert-base-uncased': BERTEncoder
                }

methods_map = {
    'mult': MULT,
}
