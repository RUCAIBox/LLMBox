# [Trouble Shooting] No module named packaging when installing vllm

When installing LLMBox, some of dependencies may not be installed correctly, especially the `vLLM` and `Flash Attention` module.

```txt
$ pip install -r requirements.txt
...
ModuleNotFoundError: No module named 'packaging'
```

It often fails even after installing the `packaging` module. This is because compiling the `vLLM` or the `Flash Attention` module sets up an isolated environment, and the `packaging` module is not installed in that environment.

Since it often fails to install the `packaging` module in the isolated environment, you can install the `packaging` module manually.

## Solution

1. Remove `vllm` and `flash_attention` from the `requirements.txt` file.
2. Upgrade the pip module: `pip install --upgrade pip`.
3. Install the packaging modules manually: `pip install packaging wheel setuptools`.
4. Install the `vllm` or `flash_attention` modules: `pip install vllm flash-attn --no-build-isolation`.
