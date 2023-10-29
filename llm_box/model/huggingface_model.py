from .model import Model
from .utils import load_llm_and_tokenizer
from ..utils import ModelArguments


class HuggingFaceModel(Model):

    def __init__(self, model_name_or_path: str, args: ModelArguments):
        super().__init__(args)
        self.model_name_or_path = model_name_or_path

        model, tokenizer = load_llm_and_tokenizer(model_name_or_path)
        self.model = model
        self.tokenizer = tokenizer

    def get_ppl(self, batched_inputs):
        batched_prompts = [src + tgt for src, tgt in batched_inputs]
        batched_results = self.model(batched_prompts)
        print(batched_results)
        ppls = []
        for result, (src, _) in zip(batched_results, batched_inputs):
            tgt_start = result['logprobs']['text_offset'].index(len(src))
            tgt_end = len(result['logprobs']['text_offset'])
            ppl = -sum(result['logprobs']['token_logprobs'][tgt_start:])
            ppls.append((ppl, tgt_end - tgt_start))
        return ppls

    def generation(self, batch):
        prompt = [question for question in batch]
        results = self.request(prompt, self.generation_kwargs)
        answers = []
        for result, _ in zip(results, batch):
            if self.name == 'gpt-3.5-turbo':
                answer = result[0]['message']['content']
            else:
                answer = result['text']
            answers.append(answer)
        return answers
