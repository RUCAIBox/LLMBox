from functools import cached_property

from ..metric import WordAccuracy
from .generation_dataset import GenerationDataset

class Fda(GenerationDataset):
    r"""The dataset of FDA.

    FDA (Information Extraction). The task is to extract key-value pairs from a set of PDFs scraped from the FDA website. 

    Example:
        'key': 'purpose for submission',
        'value': 'Clearance of a new device',
        'text': 'STANTIAL EQUIVALENCE DETERMINATION DECISION SUMMARY A. 510(k) Number: K153137 B. Purpose for Submission: Clearance of a new device C. Measurand: Anti-PF4/Heparin Total Antibodies D. Type of Test: Automated, latex enhanced immuno-turbidimetric assay E. Applicant: Instrumentation Laboratory (IL) Co. F. Proprietary and Established Names: HemosIL HIT‐Ab(PF4‐H) HemosIL HIT‐Ab(PF4‐H) Controls G. Regulatory Information: 1. Regulation section: 21 CFR 864.7695, Platelet factor 4 radioimmunoassay 21 CFR 864.5425, Multipurpose system for in vitro coagulation studies 2. Classification: Class II 3. Product code: 2 LCO, Platelet factor 4 radioimmunoassay GGN, Plasma, Coagulation Control 4. Panel: Hematology (81) H. Intended Use: 1. Intended use(s): HemosIL HIT-Ab(PF4-H) is a qualitative, fully automated, latex enhanced immunoassay for the detection of anti-platelet factor 4/heparin (PF4/H) antibodies. The assay is for use in human 3.2% or 3.8% citrated plasma on the ACL TOP® Family of instruments in a laboratory setting. The result provided by the assay should be interpreted as either positive or negative based on the assay cut-off (1.0 U/mL). The positive or negative result aids in determining the risk for heparin induced thrombocytopenia (HIT) when used in conjunction with other laboratory and clinical findings. Anti-PF4/Heparin antibodies are commonly found in patients with HIT. For use in adult population suspected of HIT. (...)

        Purpose for submission:',

    """
    instruction = "{source}"
    evaluation_set = "validation"
    example_set = None
    load_args = ("hazyresearch/based-fda", "default")
    extra_model_args = dict(max_tokens=48, temperature=0, stop=["\n"])
    metrics = [WordAccuracy()]


    def format_instance(self, instance):
        instance["source"] = instance["text"]
        instance["target"] = instance["value"]
        return instance

    def post_processing(self, predictions):
        return [prediction.strip() for prediction in predictions]
    
    @cached_property
    def references(self):
        return [instance["target"] for instance in self.evaluation_data]