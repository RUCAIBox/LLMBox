from datasets import load_dataset, load_from_disk

from .translation_dataset import TranslationDataset

class Wmt(TranslationDataset):
    """ The dataset of Wmt dataset.
    
    Wmt dataset contains 2 translation tasks:
        wmt14:

    
    Example:
        hypothesis: Obama welcomes Netanyahu
        reference: Obama receives Netanyahu
    """
    
    def __init__(self, args, model):
        self.name = 'wmt'
        
        # dataset = load_from_disk("../dataset/wmt16")
        dataset = load_dataset('wmt16', args.config)
        self.example_data = list(dataset[args.example_set])
        self.evaluation_data = list(dataset[args.evaluation_set])
        self.instruction = f"Translate from {args.config[:2]} to {args.config[3:5]}"

        super().__init__(args, model)
        
    def format_instance(self, instance):
        return instance['translation']
        # return dict(
        #     source=instance['translation']['en'],
        #     target=instance['translation']['de'],
        # )
        
    @property
    def references(self):
        return [instance['translation'] for instance in self.evaluation_data]