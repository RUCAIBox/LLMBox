import argparse


def parse_argument():
    r"""Parse arguments from command line. Using `argparse` for predefined ones, and an easy mannal parser for others (saved in `kwargs`).

    Returns:
        Namespace: the parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="curie")
    parser.add_argument("--dataset", type=str, default="copa")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--max_example_tokens", type=int, default=1024)
    parser.add_argument("--example_set", type=str, default="train")
    parser.add_argument("--evaluation_set", type=str, default="validation")
    parser.add_argument("--example_separator_string", type=str, default="\n\n")
    parser.add_argument("--instruction", type=str, default="")
    parser.add_argument("--openai_api_key", type=str, default="")

    args, unparsed = parser.parse_known_args()

    new_unparsed = []
    for arg in unparsed:
        if arg.find('=') >= 0:
            new_unparsed.append(arg.split('='))
        else:
            new_unparsed.append(arg)

    assert len(new_unparsed) % 2 == 0, "Arguments parsing error!"
    kwargs = {}
    for i in range(len(new_unparsed) // 2):
        key, value = new_unparsed[i * 2:i * 2 + 2]
        if key.find('--') != 0:
            raise KeyError
        else:
            key = key[2:]
            try:
                value = eval(value)
            except:
                pass
            setattr(args, key, value)
            kwargs[key] = value
    args.kwargs = kwargs

    return args
