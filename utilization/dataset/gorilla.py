import re
import signal
import json
from typing import Dict, List

from pandas.core.api import DataFrame as DataFrame

from ..metric import Gorilla_eval
from .generation_dataset import GenerationDataset

class Gorilla(GenerationDataset):
    instruction = "Answer the following question."
    evaluation_set = "test"
    example_set = ""
    load_args = ("gorilla-llm/APIBench",)
    metrics = []
    extra_model_args = dict(temperature=0)

    def __init__(self, dataset_name, args, model, subset_name=None):
        super().__init__(dataset_name, args, model, subset_name)
        self.metrics.append(Gorilla_eval(self.args.api_name))

    def load_raw_dataset(self, dataset_path, subset_name, evaluation_set, example_set):
        self.api_bench = []
        self.evaluation_data = []
        
        api_bench_files = f'./utilization/dataset/gorilla_data/data/apibench/{self.args.api_name}_eval.json'
        question_files = f"./utilization/dataset/gorilla_data/eval/eval-data/questions/{self.args.api_name}/questions_{self.args.api_name}_0_shot.jsonl"

        with open(api_bench_files, 'r') as f:
            for line in f:
                self.api_bench.append(json.loads(line)["api_data"])

        with open(question_files, "r") as ques_file:
            for line in ques_file:
                ques_json = json.loads(line)
                self.evaluation_data.append(ques_json)

    @staticmethod
    def post_processing(preds):
        return [pred.strip() for pred in preds]

    def format_instance(self, instance):
        question = instance["text"]
        if self.args.api_name == "torchhub":
            domains = "1. $DOMAIN is inferred from the task description and should include one of {Classification, Semantic Segmentation, Object Detection, Audio Separation, Video Classification, Text-to-Speech}."
        elif self.args.api_name == "huggingface":
            domains = "1. $DOMAIN should include one of {Multimodal Feature Extraction, Multimodal Text-to-Image, Multimodal Image-to-Text, Multimodal Text-to-Video, \
            Multimodal Visual Question Answering, Multimodal Document Question Answer, Multimodal Graph Machine Learning, Computer Vision Depth Estimation,\
            Computer Vision Image Classification, Computer Vision Object Detection, Computer Vision Image Segmentation, Computer Vision Image-to-Image, \
            Computer Vision Unconditional Image Generation, Computer Vision Video Classification, Computer Vision Zero-Shor Image Classification, \
            Natural Language Processing Text Classification, Natural Language Processing Token Classification, Natural Language Processing Table Question Answering, \
            Natural Language Processing Question Answering, Natural Language Processing Zero-Shot Classification, Natural Language Processing Translation, \
            Natural Language Processing Summarization, Natural Language Processing Conversational, Natural Language Processing Text Generation, Natural Language Processing Fill-Mask,\
            Natural Language Processing Text2Text Generation, Natural Language Processing Sentence Similarity, Audio Text-to-Speech, Audio Automatic Speech Recognition, \
            Audio Audio-to-Audio, Audio Audio Classification, Audio Voice Activity Detection, Tabular Tabular Classification, Tabular Tabular Regression, \
            Reinforcement Learning Reinforcement Learning, Reinforcement Learning Robotics }"
        elif self.args.api_name == "tensorflowhub":
            domains = "1. $DOMAIN is inferred from the task description and should include one of {text-sequence-alignment, text-embedding, text-language-model, text-preprocessing, text-classification, text-generation, text-question-answering, text-retrieval-question-answering, text-segmentation, text-to-mel, image-classification, image-feature-vector, image-object-detection, image-segmentation, image-generator, image-pose-detection, image-rnn-agent, image-augmentation, image-classifier, image-style-transfer, image-aesthetic-quality, image-depth-estimation, image-super-resolution, image-deblurring, image-extrapolation, image-text-recognition, image-dehazing, image-deraining, image-enhancemenmt, image-classification-logits, image-frame-interpolation, image-text-detection, image-denoising, image-others, video-classification, video-feature-extraction, video-generation, video-audio-text, video-text, audio-embedding, audio-event-classification, audio-command-detection, audio-paralinguists-classification, audio-speech-to-text, audio-speech-synthesis, audio-synthesis, audio-pitch-extraction}"
        else:
            print("Error: API name is not supported.")

        prompt = question + "\nWrite a python program in 1 to 2 lines to call API in " + self.args.api_name + ".\n\nThe answer should follow the format: <<<DOMAIN>>>: $DOMAIN, <<<API_CALL>>>: $API_CALL, <<<API_PROVIDER>>>: $API_PROVIDER, <<<EXPLANATION>>>: $EXPLANATION, <<<CODE>>>: $CODE. Here are the requirements:\n" + domains + "\n2. The $API_CALL should have only 1 line of code that calls api.\n3. The $API_PROVIDER should be the programming framework used.\n4. $EXPLANATION should be a step-by-step explanation.\n5. The $CODE is the python code.\n6. Do not repeat the format in your answer."
        return dict(
            source=prompt,
            target=str(self.api_bench[instance["question_id"] - 1]),
        )

    @property
    def references(self):
        return [str(self.api_bench[instance["question_id"] - 1]) for instance in self.evaluation_data]