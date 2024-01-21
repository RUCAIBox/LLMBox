from typing import Iterable, Optional, Union

import tqdm


class dynamic_stride_tqdm(tqdm.tqdm):
    def __init__(
        self,
        iterable=None,
        strides: Iterable[float] = None,
        stride_scale: Union[float, bool] = False,
        desc: Optional[str] = None,
        disable: bool = False,
        unit: str = "it",
        dynamic_ncols: bool = False,
        **kwargs
    ):
        """Tqdm progress bar with dynamic strides. Use `strides` to specify the strides for each step and `stride_scale` to scale the strides. For example, if `strides` is `[1, 2, 3]` and `stride_scale` is `2`, then the fianl strides will be `[2, 4, 6]`, which require 12 iterations to stop. Different from `unit_scale` which changes the unit of the progress bar., `stride_scale` only changes the stride of each iteration. `total` is set to the length of `strides` list by default."""
        self.strides = list(strides)
        self.stride_scale = float(stride_scale)
        super().__init__(
            iterable=iterable,
            desc=desc,
            disable=disable,
            unit=unit,
            dynamic_ncols=dynamic_ncols,
            total=len(self.strides),
            **kwargs
        )

    def __iter__(self):
        """Overwrite the original tqdm iterator to support dynamic strides."""

        iterable = self.iterable

        if self.disable:
            for obj in iterable:
                yield obj
            return

        mininterval = self.mininterval
        last_print_t = self.last_print_t
        last_print_n = self.last_print_n
        min_start_t = self.start_t + self.delay
        n = self.n
        time = self._time

        try:
            for obj in iterable:
                yield obj
                if int(n) >= len(self.strides):
                    # allow overflow of progress bar to avoid out of range error
                    n += 1 * self.stride_scale
                else:
                    # Update the progress bar with dynamic strides
                    n += 1 * self.stride_scale / self.strides[int(n)]

                if n - last_print_n >= self.miniters:
                    cur_t = time()
                    dt = cur_t - last_print_t
                    if dt >= mininterval and cur_t >= min_start_t:
                        self.update(n - last_print_n)
                        last_print_n = self.last_print_n
                        last_print_t = self.last_print_t
        finally:
            self.n = n
            self.close()
