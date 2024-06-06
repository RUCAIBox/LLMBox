import random

import numpy as np


def set_seed(seed: int, device_specific: bool = False, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        device_specific (`bool`, *optional*, defaults to `False`):
            Whether to differ the seed on each device slightly with `self.process_index`.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    import torch

    try:
        import accelerate
        from accelerate.utils.imports import (
            is_mlu_available, is_npu_available, is_torch_xla_available, is_xpu_available
        )

        if is_torch_xla_available():
            import torch_xla.core.xla_model as xm
    except (ModuleNotFoundError, ImportError):
        accelerate = None

    if device_specific and accelerate is not None:
        seed += accelerate.state.AcceleratorState().process_index

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if accelerate is not None:
        if is_xpu_available():
            torch.xpu.manual_seed_all(seed)
        elif is_npu_available():
            torch.npu.manual_seed_all(seed)
        elif is_mlu_available():
            torch.mlu.manual_seed_all(seed)
        else:
            torch.cuda.manual_seed_all(seed)
            # ^^ safe to call this function even if cuda is not available
        if is_torch_xla_available():
            xm.set_rng_state(seed)
    else:
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
