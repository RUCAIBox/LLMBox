from accelerate.utils import set_seed
from llm_box.utils import parse_argument


def main():
    r"""The main pipeline for argument parsing, initialization, and evaluation.
    """
    args = parse_argument()

    set_seed(args.seed)
    # TODO: init logger

    from llm_box.evaluator import Evaluator
    evaluator = Evaluator(args)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
