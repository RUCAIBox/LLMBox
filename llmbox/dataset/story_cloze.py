from .multiple_choice_dataset import MultipleChoiceDataset


class Story_cloze(MultipleChoiceDataset):
    """The dataset of storycloze.

    Story Cloze Test' is a new commonsense reasoning framework for evaluating story understanding, story generation, and script learning.This test requires a system to choose the correct ending to a four-sentence story.

    Example:
        context: Mary wanted to go to work. She got into her car and drove on the highway. She realized she forgot her lunch and turned around to go back and get it. She went back home and got her lunch. She then drove to work.
        endings: "[\", Mary got into her car and drove to work.\", \", Mary got into her car and drove to the store.\", \", Mary got into her car and drove to the airport.\", \", Mary got into her car and drove to the hospital.\"]",
        label: 0,
    """

    instruction = ""
    evaluation_set = "test"
    example_set = "validation"
    """
     Because of the dataset policy, we can't download the dataset automatically, so we need to download it manually and load it from the local disk.

     Please follow the manual download instructions:
     To use Story Cloze you have to download it manually. Please fill this google form (http://goo.gl/forms/aQz39sdDrO). Complete the form. Then you will receive a download link for the dataset. Load it using: `--dataset_path /path/to/dataset`.
    """
    load_args = ("story_cloze", "2016")

    def format_instance(self, instance):
        instance["answer_right_ending"] = instance["answer_right_ending"] - 1

        source = (
            instance["input_sentence_1"] + " " + instance["input_sentence_2"] + " " + instance["input_sentence_3"] +
            " " + instance["input_sentence_4"]
        )

        label2text = {
            0: " " + instance["sentence_quiz1"][0] + instance["sentence_quiz1"][1:],
            1: " " + instance["sentence_quiz2"][0] + instance["sentence_quiz2"][1:],
        }

        options = [label2text[option] for option in (0, 1)]
        return dict(
            source=source,
            target=label2text[instance["answer_right_ending"]],
            options=options,
        )

    @property
    def references(self):
        return [instance["answer_right_ending"] for instance in self.evaluation_data]
