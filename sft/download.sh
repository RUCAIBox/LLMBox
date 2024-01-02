# custom wget args 
wget_args="-nc" 

# check if there is $HF_TOKEN in the environment variables
if [ -z "$HF_TOKEN" ]
then
    echo "Warning: HuggingFace dataset LIMA requires permissive access."
    echo "Warning: Please request the access at https://huggingface.co/datasets/GAIR/lima and set the HF_TOKEN environment variable before running this script."
    exit 1
fi

echo "Downloading self-instruct data..."
wget -P data/raw_train/self_instruct/ https://raw.githubusercontent.com/yizhongw/self-instruct/main/data/gpt3_generations/batch_221203/all_instances_82K.jsonl  $wget_args


echo "Downloading Stanford alpaca data..."
wget -P data/raw_train/alpaca/ https://github.com/tatsu-lab/stanford_alpaca/raw/main/alpaca_data.json  $wget_args


echo "Downloading the dolly dataset..."
wget -P data/raw_train/dolly/ https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl  $wget_args


echo "Downloading the OpenAssistant data (oasst1)..."
wget -P data/raw_train/openassistant/ https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz  $wget_args
gzip -d data/raw_train/openassistant/2023-04-12_oasst_ready.trees.jsonl.gz


echo "Downloading ShareGPT dataset..."
wget -P data/raw_train/sharegpt/ https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part1_html_cleaned.json  $wget_args
wget -P data/raw_train/sharegpt/ https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part2_html_cleaned.json  $wget_args

echo "Downloading LIMA dataset..."
wget --header="Authorization: Bearer $HF_TOKEN" -P data/raw_train/lima/ https://huggingface.co/datasets/GAIR/lima/raw/main/train.jsonl  $wget_args


echo "Downloading evol-instruct dataset..."
wget -P data/raw_train/evol_instruct/ https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k/resolve/main/WizardLM_evol_instruct_V2_143k.json  $wget_args


echo "Downloading the Belle dataset..."
wget -P data/raw_train/belle https://huggingface.co/datasets/BelleGroup/train_0.5M_CN/resolve/main/Belle_open_source_0.5M.json $wget_args

echo "Downloading the Flanv2 dataset..."
python scripts/download_flanv2.py