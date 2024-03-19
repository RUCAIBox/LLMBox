import json


def remove_unusual_line_terminators(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile:
        data = infile.read()

    # Remove unusual line terminators
    data = data.replace("\u2028", "").replace("\u2029", "")

    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write(data)


if __name__ == "__main__":
    input_file = "data/raw_train/belle/Belle_open_source_0.5M.json"
    output_file = "data/belle.json"
    remove_unusual_line_terminators(input_file, output_file)
