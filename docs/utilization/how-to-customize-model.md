# How to Customize Model

## Customizing HuggingFace Models

If you are building on your own model, such as using a fine-tuned model, you can evaluate it easily from python script. Detailed steps and example code are provided in the [customize HuggingFace model guide](https://github.com/RUCAIBox/LLMBox/tree/main/docs/examples/customize_huggingface_model.py).

## Adding a New Model Provider

If you're integrating a new model provider, begin by extending the [`Model`](https://github.com/RUCAIBox/LLMBox/tree/main/utilization/model/model.py) class. Implement essential methods such as `generation`, `get_ppl` (get perplexity), and `get_prob` (get probability) to support different functionalities. For instance, here's how you might implement the `generation` method for a new model:

```python
class NewModel(Model):

    model_backend = "new_provider"

    def call_model(self, batched_inputs: List[str]) -> List[Any]:
        return ...  # call to model, e.g., self.model.generate(...)

    def to_text(self, result: Any) -> str:
        return ...  # convert result to text, e.g., result['text']

    def generation(self, batched_inputs: List[str]) -> List[str]:
        results = self.call_model(batched_inputs)
        results = [to_text(result) for result in results]
        return results
```

And then, you should register your model with `register_model` decorator:

```python
@register_model(model_backend="new_provider")
def load_new_model(args: "ModelArguments"):
    logger.info(f"Loading OpenAI API model `{args.model_name_or_path}`.")

    return NewModel(args)
```
