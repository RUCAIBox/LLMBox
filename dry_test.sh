
datasets=("agieval" "alpaca_eval" "anli" "arc" "bbh" "boolq" "cb" "ceval" "cmmlu" "cnn_dailymail" "color_objects" "commonsenseqa" "copa" "coqa" "crows_pairs" "drop" "gaokao" "gsm8k" "halueval" "hellaswag" "humaneval" "ifeval" "lambada" "math" "mbpp" "mmlu" "mt_bench" "nq" "openbookqa" "penguins_in_a_table" "piqa" "quac" "race" "real_toxicity_prompts" "rte" "siqa" "squad" "squad_v2" "story_cloze" "tldr" "triviaqa" "truthfulqa_mc" "vicuna_bench" "webq" "wic" "winogender" "winograd" "winogrande" "wmt16:en-ro" "wsc" "xsum")



function dry_test() {
    echo "Running dry test on all datasets"
    for dataset in "${datasets[@]}"
    do
        if [[ "$dataset" = "mbpp" || "$dataset" = "humaneval" ]]; then
            args="--pass_at_k 1"
        elif [[ "$dataset" = "coqa" ]]; then
            if [[ -n "$COQA_PATH" ]]; then
                args="--dataset_path $COQA_PATH"
            else
                echo "Skipping $dataset - COQA_PATH not set"
                continue
            fi
        elif [[ "$dataset" = "story_cloze" ]]; then
            if [[ -n "$STORY_CLOZE_PATH" ]]; then
                args="--dataset_path $STORY_CLOZE_PATH"
            else
                echo "Skipping $dataset - STORY_CLOZE_PATH not set"
                continue
            fi
        elif [[ "$dataset" = "real_toxicity_prompts" ]]; then
            if [[ -n "$PERSPECTIVE_API_KEY" ]]; then
                args="--perspective_api_key $PERSPECTIVE_API_KEY"
            else
                echo "Skipping $dataset - PERSPECTIVE_API_KEY not set"
                continue
            fi
        elif [[ "$dataset" = "alpaca_eval" || "$dataset" = "mt_bench" || "$dataset" = "vicuna_bench" ]]; then
            if [[ -n "$OPENAI_API_KEY" ]]; then
                args="--openai_api_key $OPENAI_API_KEY"
            else
                echo "Skipping $dataset - OPENAI_API_KEY not set"
                continue
            fi
        else
            args=""
        fi

        echo "Running on $dataset"
        python inference.py -m gpt2 -d "$dataset" --dry_run $args 2>&1 /dev/null
        if [ $? -ne 0 ]; then
            echo " ❎"
        else
            echo " ✅"
        fi
    done
}


function prefix_caching_test() {
    green="\033[32m"
    blue="\033[34m"
    reset="\033[0m"
    echo "Running prefix_caching test on all datasets"
    for dataset in "${datasets[@]}"
    do
        if [[ "$dataset" = "mbpp" || "$dataset" = "humaneval" ]]; then
            args="--pass_at_k 1"
        elif [[ "$dataset" = "coqa" ]]; then
            if [[ -n "$COQA_PATH" ]]; then
                args="--dataset_path $COQA_PATH"
            else
                echo "Skipping $dataset - COQA_PATH not set"
                continue
            fi
        elif [[ "$dataset" = "story_cloze" ]]; then
            if [[ -n "$STORY_CLOZE_PATH" ]]; then
                args="--dataset_path $STORY_CLOZE_PATH"
            else
                echo "Skipping $dataset - STORY_CLOZE_PATH not set"
                continue
            fi
        elif [[ "$dataset" = "real_toxicity_prompts" ]]; then
            if [[ -n "$PERSPECTIVE_API_KEY" ]]; then
                args="--perspective_api_key $PERSPECTIVE_API_KEY"
            else
                echo "Skipping $dataset - PERSPECTIVE_API_KEY not set"
                continue
            fi
        elif [[ "$dataset" = "alpaca_eval" || "$dataset" = "mt_bench" || "$dataset" = "vicuna_bench" ]]; then
            if [[ -n "$OPENAI_API_KEY" ]]; then
                args="--openai_api_key $OPENAI_API_KEY"
            else
                echo "Skipping $dataset - OPENAI_API_KEY not set"
                continue
            fi
        else
            args=""
        fi

        echo -e "${green}Running on $dataset (--prefix_caching True)${reset}"
        python inference.py -m /home/tangtianyi/Llama-2-7b-hf -d "$dataset" --max_evaluation_instances 50 -shots 5 --model_type instruction -b 20:auto $args | tail -n 2 | head -n 1 | echo -e "${blue}>>> $dataset (--prefix_caching True): $(cat)${reset}"

        echo -e "${green}Running on $dataset (--prefix_caching False)${reset}"
        python inference.py -m /home/tangtianyi/Llama-2-7b-hf -d "$dataset" --prefix_caching False --max_evaluation_instances 50 -shots 5 --model_type instruction -b 20:auto $args | tail -n 2 | head -n 1 | echo -e "${blue}>>> $dataset (--prefix_caching False): $(cat)${reset}"
    done
}



if [[ -z $1 ]]; then
    echo "Usage: dry_test.sh <command>"
    echo "Commands:"
    echo "  all: Run test on all datasets"
    echo "  dry_test: Run dry test on all datasets"
    echo "  prefix_caching: Run prefix caching test on all datasets"
    exit 1
fi

if [[ -z $CUDA_VISIBLE_DEVICES ]]; then
    CUDA_VISIBLE_DEVICES=0
fi

if [[ $1 = "all" ]]; then
    dry_test
    prefix_caching_test
elif [[ $1 = "dry_test" ]]; then
    dry_test
elif [[ $1 = "prefix_caching" ]]; then
    prefix_caching_test
else
    echo "Invalid command"
    exit 1
fi

