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


class Dataset():

    def __new__(self, args):
        datapath = args.data_path
        for datasetname, datasetclass in DATASETNAMEMAP.items():
            # if the datasetname is in the datapath, then we select this dataset
            if datasetname in datapath:
                print(f"Dataset: {datasetname} is selected")
                return datasetclass(args)

        # failed to find the dataset
        raise ValueError(
            f"Your {datapath} should contain names like these: {DATASETNAMEMAP.keys()}, so that it can find our sftdataset class. Or you can add your own dataset class in sftdatasets package."
        )
