import argparse
import transformers
from accelerate import init_empty_weights


def calc_memory_usage(parameters_num, bytes_per_parameter=2, lora_parameters_num=None, zero_config='z3', gpu_num=2, batch_size=1, seq_length=2048, layer_num=32, hidden_size=4096, vocab_size=32000):
    if lora_parameters_num is None:
        optimizer_parameters_num = parameters_num
    else:
        parameters_num += lora_parameters_num
        optimizer_parameters_num = lora_parameters_num
    activation_memory = (12 + 2 * layer_num) * batch_size * seq_length * hidden_size + 12 * batch_size * seq_length * vocab_size
    # print(f'activation_memory = {activation_memory/2**30:.2f} GB')
    if zero_config == 'z0':
        memory = bytes_per_parameter * parameters_num + 14 * optimizer_parameters_num
        memory += max(2 * optimizer_parameters_num, activation_memory)
    elif zero_config == 'z1':
        memory = bytes_per_parameter * parameters_num + 2 * optimizer_parameters_num + 12 * optimizer_parameters_num / gpu_num
        memory += max(2 * optimizer_parameters_num + 2 * optimizer_parameters_num / gpu_num, activation_memory)
    elif zero_config == 'z2':
        memory = bytes_per_parameter * parameters_num + 14 * optimizer_parameters_num / gpu_num
        memory += max(2 * optimizer_parameters_num + 2 * optimizer_parameters_num / gpu_num, activation_memory)
    elif zero_config == 'z2+oo':
        memory = bytes_per_parameter * parameters_num
        memory += max(2 * optimizer_parameters_num, activation_memory)
    elif zero_config == 'z3':
        memory = bytes_per_parameter * parameters_num / gpu_num + 14 * optimizer_parameters_num / gpu_num
        memory += max(2 * optimizer_parameters_num / gpu_num, activation_memory)
    elif zero_config == 'z3+oo':
        memory = bytes_per_parameter * parameters_num / gpu_num
        memory += max(2 * optimizer_parameters_num / gpu_num, activation_memory)
    elif zero_config == 'z3+oo+op':
        memory = activation_memory
    else:
        raise ValueError('invaild zero_config!')
    return memory


def get_gpu_memory_size(gpu_name='A100'):
    if gpu_name.lower() in ('h100', 'h800', 'a100', 'a800'):
        return 80 * 2**30
    elif gpu_name.lower() in ('a100-40g', 'a800-40g'):
        return 40 * 2**30
    elif gpu_name.lower() in ('a6000', 'a40'):
        return 48 * 2**30
    elif gpu_name.lower() in ('v100',):
        return 32 * 2**30
    elif gpu_name.lower() in ('3090', '4090', 'rtx-3090', 'rtx-4090'):
        return 24 * 2**30
    else:
        raise ValueError('unsupported gpu!')


def get_bytes_per_parameter(model_dtype):
    if model_dtype.lower() in ('bf16', 'fp16'):
        return 2.0
    elif model_dtype.lower() in ('int8', 'fp8'):
        return 1.0
    elif model_dtype.lower() in ('int4'):
        return 0.5
    else:
        raise ValueError('unsupported model_dtype!')


def calc_gpu_num(model_name='meta-llama/Llama-2-7b-hf', model_dtype='bf16', lora_module='q,k,v,o', lora_rank=8, seq_length=2048, zero_config='z3', gpu_name='A100', batch_size=1):
    config = transformers.AutoConfig.from_pretrained(model_name)
    layer_num = config.num_hidden_layers
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    gpu_memory = get_gpu_memory_size(gpu_name)
    bytes_per_parameter = get_bytes_per_parameter(model_dtype)
    with init_empty_weights():
        model = transformers.AutoModelForCausalLM.from_config(config)
        parameters_num = model.num_parameters()
        print(f'parameters_num = {parameters_num:e}')
    if lora_module is not None:
        lora_module = lora_module.split()
        lora_parameters_num = 2 * hidden_size * lora_rank * layer_num * len(lora_module)
        print(f'lora_parameters_num = {lora_parameters_num:e}')
    else:
        lora_parameters_num = None
    for gpu_num in range(1, 100):
        memory_usage = calc_memory_usage(
            parameters_num,
            bytes_per_parameter=bytes_per_parameter,
            lora_parameters_num=lora_parameters_num,
            zero_config=zero_config,
            gpu_num=gpu_num,
            batch_size=batch_size,
            seq_length=seq_length,
            layer_num=layer_num,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )
        if memory_usage <= 0.95 * gpu_memory:
            return gpu_num, memory_usage
    raise ValueError('Can not find suitable gpu_num for this config!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--model_dtype",
        type=str,
        default="bf16",
    )
    parser.add_argument(
        "--zero_config",
        type=str,
        default="",
    )
    parser.add_argument(
        "--lora_module",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--gpu_name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    gpu_num, memory_usage = calc_gpu_num(
        model_name=args.model_name,
        model_dtype=args.model_dtype,
        lora_module=args.lora_module,
        seq_length=args.seq_length,
        zero_config=args.zero_config,
        gpu_name=args.gpu_name,
        batch_size=args.batch_size,
    )
    print(f'gpu_num = {gpu_num}, memory_usage = {memory_usage / 2**30:.2f} GB')
