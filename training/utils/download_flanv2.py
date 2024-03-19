import os

from datasets import load_dataset

# load from remote url
flan2021_submix_original = load_dataset("DataProvenanceInitiative/flan2021_submix_original")
t0_submix_original = load_dataset("DataProvenanceInitiative/t0_submix_original")
niv2_submix_original = load_dataset("DataProvenanceInitiative/niv2_submix_original")
cot_submix_original = load_dataset("DataProvenanceInitiative/cot_submix_original")
dialog_submix_original = load_dataset("DataProvenanceInitiative/dialog_submix_original")
# Save json to data/raw_train/flanv2/*.json
os.mkdir("data/raw_train/flanv2", exist_ok=True)
flan2021_submix_original.to_json(f"data/raw_train/flanv2/flan2021_submix_original.json")
t0_submix_original.to_json(f"data/raw_train/flanv2/t0_submix_original.json")
niv2_submix_original.to_json(f"data/raw_train/flanv2/niv2_submix_original.json")
cot_submix_original.to_json(f"data/raw_train/flanv2/cot_submix_original.json")
dialog_submix_original.to_json(f"data/raw_train/flanv2/dialog_submix_original.json")
