from llmbox import Evaluator, parse_argument


def main():
    r"""The main pipeline for argument parsing, initialization, and evaluation."""
    args = parse_argument()

    evaluator = Evaluator(args)
    evaluator.evaluate()


if __name__ == "__main__":
    import os

    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    main()
