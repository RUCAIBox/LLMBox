## Supervised Fine-tuning and Continual Pre-training

### 1. Supervised fine-tuning (instruction tuning)
#### Downloading the SFT dataset
```shell
bash download.sh
```

#### Training a SFT model
Just change the `data_path` in `bash/run_7b_ds3.sh` to the path you want, e.g., `data/alpaca_data.json`. Then run the following script:
```shell
bash bash/run_7b_ds3.sh
```

The default behaviour is to train the LLaMA-2-7b model on the subset of the alpaca dataset.

### 2. Continual pre-training with your own corpora

#### Merging tokenizer

If you want to add new tokens (such as Chinese characters) to your vocabulary and then continual pre-train the model on your corpora, you just need to prepare the corpora under the folder `data/chinese.txt` and run the following script:

```shell
bash bash/run_7b_pt.sh
```

It will first merge the vocabulary of your corpora and the original vocabulary and then tune the parameters of the whole model to adapt to your corpora.

#### User-defined symbols

If you want to add user-defined symbols when merging new tokenizers, you can rewrite the `user_defined_symbols.json`. 

```json
{
    "list": [
    	"symbol1",
    	"symbol2"
    ]
}
```

#### Others
You can also leverage part of the script in `bash/run_7b_pt.sh` to just merge the tokenizer or continual pre-train the model using your corpora with the original tokenizer.