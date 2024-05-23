from utilization import Evaluator, parse_argument


def main():
    r"""The main pipeline for argument parsing, initialization, and evaluation."""
    model_args, dataset_args, evaluation_args = parse_argument(initalize=True)

    evaluator = Evaluator(
        model_args=model_args,
        dataset_args=dataset_args,
        evaluation_args=evaluation_args,
        initalize=False,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()
