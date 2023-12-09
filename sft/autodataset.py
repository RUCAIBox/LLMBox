from datasets import load_dataset
from collections import OrderedDict
from sftdatasets import *

# You can add your own dataset name and corresponding class here
DATASETNAMEMAP = OrderedDict({
    "alpaca": AlpacaDataset,
    "belle": BelleDataset,
    "selfinstruct": SelfInstructDataset,
    "evolinstruct": EvolInstructDataset,
    "dolly": DollyDataset,
    "lima": LimaDataset,
    "sharegpt": ShareGPTDataset,
    "openassistant": OpenAssistantDataset,
})
DATASETNAMEMAPLIST = list(DATASETNAMEMAP.keys())


class AutoDataset():

    def __new__(self, args):
        datapath = args.data_path
        for datasetname, datasetclass in DATASETNAMEMAP.items():
            # if the datasetname is in the datapath, then we select this dataset
            if datasetname in datapath:
                print(f"AutoDataset: {datasetname} is selected")
                return datasetclass(args)

        # failed to find the dataset
        raise ValueError(
            f"Your {datapath} should contain names like these: {DATASETNAMEMAP.keys()}, so that it can find our sftdataset class. Or you can add your own dataset class to {__file__}."
        )
