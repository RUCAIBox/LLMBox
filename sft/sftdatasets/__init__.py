from collections import OrderedDict
from .alpaca import AlpacaDataset
from .self_instruct import SelfInstructDataset
from .evol_instruct import EvolInstructDataset
from .dolly import DollyDataset
from .lima import LimaDataset
from .sharegpt import ShareGPTDataset
from .belle import BelleDataset
from .openassistant import OpenAssistantDataset
from .flan import FlanDataset

# You can add your own dataset name and corresponding class here
DATASETNAMEMAP = OrderedDict({
    "alpaca": AlpacaDataset,
    "belle": BelleDataset,
    "self_instruct": SelfInstructDataset,
    "evol_instruct": EvolInstructDataset,
    "dolly": DollyDataset,
    "lima": LimaDataset,
    "sharegpt": ShareGPTDataset,
    "openassistant": OpenAssistantDataset,
    "flan": FlanDataset,
})
DATASETNAMEMAPLIST = list(DATASETNAMEMAP.keys())
