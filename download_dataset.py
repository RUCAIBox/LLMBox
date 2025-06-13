import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import sleep

os.environ['OPENAI_API_KEY'] = 'FAKE_KEY'

from utilization.load_dataset import import_dataset_classes
from utilization.utils.hfd import huggingface_download
from utilization.utils.logging import list_datasets


def download_dataset(dataset_name):
    try:
        # avoid rate limit
        sleep(1)

        print(f"Downloading dataset: {dataset_name}")
        # get the subset names
        if ":" in dataset_name:
            dataset_name, cmd_subset_names = dataset_name.split(":")
            cmd_subset_names = set(cmd_subset_names.split(","))
        else:
            cmd_subset_names = set()

        dataset_classes = import_dataset_classes(dataset_name)

        for dcls in dataset_classes:
            if len(dcls.load_args) > 0:
                huggingface_download(
                    dcls.load_args[0],
                    args.hfd_cache_path,
                    hf_username=args.hf_username,
                    hf_token=args.hf_token,
                    mirror=args.hf_mirror,
                    evaluation_args=args,
                )
            else:
                huggingface_download(
                    dataset_name,
                    args.hfd_cache_path,
                    hf_username=args.hf_username,
                    hf_token=args.hf_token,
                    mirror=args.hf_mirror,
                    evaluation_args=args,
                )
    except Exception as e:
        print(f"Error downloading dataset: {dataset_name}")
        print(e)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets",
    )
    parser.add_argument(
        "--dataset_names", "--dataset", "-d",
        type=str,
        nargs="+",
        help="Space splitted dataset names. If only one dataset is specified, it can be followed by subset names or category names. Format: 'dataset1 dataset2', 'dataset:subset1,subset2', or 'dataset:[cat1],[cat2]', e.g., 'copa race', 'race:high', 'wmt16:en-ro,en-fr', or 'mmlu:[stem],[humanities]'. Supported datasets: "
        + ", ".join(list_datasets()),
    )
    parser.add_argument(
        "--hf_mirror",
        default=False,
        help="Use hf mirror to download dataset",
    )
    parser.add_argument(
        "--hfd_cache_path",
        default="~/.cache/huggingface/datasets",
        help="Path to cache dataset",
    )
    parser.add_argument(
        "--hf_username",
        default="",
        help="Hugging Face username",
    )
    parser.add_argument(
        "--hf_token",
        default="",
    )
    parser.add_argument(
        "--hfd_exclude_pattern",
        default="",
        help="Exclude pattern for dataset",
    )
    parser.add_argument(
        "--hfd_include_pattern",
        default="",
        help="Include pattern for dataset",
    )
    parser.add_argument(
        "--hfd_skip_check",
        default=False,
        help="Skip check for dataset",
    )
    args, extra_args = parser.parse_known_args()

    assert not args.hfd_skip_check, "You cannot skip downloading dataset in this script"

    if args.all:
        dataset_names = list_datasets()
    else:
        dataset_names = args.dataset_names

    # random.shuffle(dataset_names)
    dataset_names = reversed(dataset_names)

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(download_dataset, dataset_name) for dataset_name in dataset_names]
        for future in as_completed(futures):
            future.result()
