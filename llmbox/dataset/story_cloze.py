from .multiple_choice_dataset import MultipleChoiceDataset


class Story_cloze(MultipleChoiceDataset):
    """The dataset of storycloze.

    Story Cloze Test' is a new commonsense reasoning framework for evaluating story understanding, story generation, and script learning.This test requires a system to choose the correct ending to a four-sentence story.

    Example:
        'story_id': 'b929f263-1dcd-4a0b-b267-5d5ff2fe65bb',
        'input_sentence_1': 'My friends all love to go to the club to dance.',
        'input_sentence_2': "They think it's a lot of fun and always invite.",
        'input_sentence_3': 'I finally decided to tag along last Saturday.',
        'input_sentence_4': "I danced terribly and broke a friend's toe.",
        'sentence_quiz1': 'My friends decided to keep inviting me out as I am so much fun.',
        'sentence_quiz2': 'The next weekend, I was asked to please stay home.',
        'answer_right_ending': 2
    """

    instruction = ""
    evaluation_set = "test"
    example_set = "validation"
    """
    Because of the dataset policy, we can't download the dataset automatically, so we need to download it manually and load it from the local disk.

    Please follow the manual download instructions:
    To use Story Cloze you have to download it manually. Please fill this google form (http://goo.gl/forms/aQz39sdDrO). Complete the form. Then you will receive a download link for the dataset.
    Download the file ``cloze_test_val__spring2016 - cloze_test_ALL_val.csv` and `cloze_test_test__spring2016 - cloze_test_ALL_test.csv` into `/path/dataset`.
    Then load it using: `--dataset_path /path/dataset`.
    """
    load_args = ("story_cloze", "2016")

    def format_instance(self, instance):
        source = " ".join([instance[f"input_sentence_{i}"] for i in range(1, 5)])

        label2text = {
            0: " " + instance["sentence_quiz1"],
            1: " " + instance["sentence_quiz2"],
        }

        options = [label2text[option] for option in (0, 1)]
        return dict(
            source=source,
            target_idx=instance["answer_right_ending"] - 1,
            options=options,
        )

    @property
    def references(self):
        return [instance["answer_right_ending"] - 1 for instance in self.evaluation_data]
