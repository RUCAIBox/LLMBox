# from process_oasst import from_tree_to_conversation
import json
import os

from process_belle import remove_unusual_line_terminators
from process_sg import truncate_filter

data_path = "data"
download_path = "data/raw_train"
data_dir = os.listdir(download_path)
for subdir in data_dir:
    subdir_path = os.path.join(download_path, subdir)  # e.g. data/raw_train/alpaca
    files = os.listdir(subdir_path)

    if subdir == "openassistant":
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(subdir_path, file)
                # from_tree_to_conversation(file_path, os.path.join(data_path, 'openassistant.jsonl')) TODO: Not supported yet

    elif subdir == "sharegpt":
        content1 = json.load(open(os.path.join(subdir_path, files[0]), "r"))
        content2 = json.load(open(os.path.join(subdir_path, files[1]), "r"))
        content = content1 + content2
        json.dump(content, open(os.path.join(data_path, "sharegpt_bug.json"), "w"))
        truncate_filter(os.path.join(data_path, "sharegpt_bug.json"), os.path.join(data_path, "sharegpt.json"))
        os.system("rm " + os.path.join(data_path, "sharegpt_bug.json"))

    elif subdir == "belle":
        file_path = os.path.join(subdir_path, files[0])
        save_path = os.path.join(data_path, "belle.json")
        remove_unusual_line_terminators(file_path, save_path)

    elif subdir == "flanv2":
        os.rename(os.path.join(subdir_path, files[0]), os.path.join(data_path, "flanv2.jsonl"))
    #     # merge all json files into one
    #     content = []
    #     for file in files:
    #         if file.endswith(".json"):
    #             content += json.load(open(os.path.join(subdir_path, file), "r"))

    #     # save the merged json file
    #     save_path = os.path.join(data_path, "flanv2.json")
    #     json.dump(content, open(save_path, "w", encoding="utf-8"))

    else:
        file_path = os.path.join(subdir_path, files[0])
        os.system("cp " + file_path + " " + os.path.join(data_path, subdir + "." + files[0].split(".")[-1]))
