from utilization import Evaluator, parse_argument


def main():
    r"""The main pipeline for argument parsing, initialization, and evaluation."""
    args = parse_argument()

    evaluator = Evaluator(args)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
