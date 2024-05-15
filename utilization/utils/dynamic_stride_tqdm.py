from logging import getLogger
from typing import Iterable, Optional, Union

import tqdm

logger = getLogger(__name__)


class dynamic_stride_tqdm(tqdm.tqdm):

    def __init__(
        self,
        iterable,
        strides: Iterable[int],
        desc: Optional[str] = None,
        disable: bool = False,
        unit: str = "it",
        dynamic_ncols: bool = False,
        miniters: Optional[Union[int, float]] = 1,
        continue_from: Optional[float] = None,
        **kwargs
    ):
        """Tqdm progress bar with dynamic strides. Use `strides` to specify the strides for each step and `stride_scale` to scale the strides. For example, if `strides` is `[1, 2, 3]` and `stride_scale` is `2`, then the fianl strides will be `[2, 4, 6]`, which require 12 iterations to stop. Different from `unit_scale` which changes the unit of the progress bar., `stride_scale` only changes the stride of each iteration. `total` is set to the length of `strides` list by default."""
        self.strides = list(strides)
        self.accumulated = [0]
        for stride in self.strides:
            self.accumulated.append(self.accumulated[-1] + stride)
        self.continue_from = continue_from
        self._hold = False
        super().__init__(
            iterable=iterable,
            desc=desc,
            disable=disable,
            unit=unit,
            dynamic_ncols=dynamic_ncols,
            total=len(self.strides),
            miniters=miniters,
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
        self.delta_a = 1
        n = self.n
        a = self.continue_from or self.accumulated[self.n]
        time = self._time

        try:
            for obj in iterable:
                yield obj
                if self.delta_a == 0:
                    continue
                if int(n) >= len(self.accumulated):
                    continue
                while a + self.delta_a > self.accumulated[int(n)]:
                    n += 1
                    if int(n) >= len(self.accumulated):
                        break
                a += self.delta_a
                self.delta_a = 1
                n = min(n, len(self.strides))

                if n - last_print_n >= self.miniters:
                    cur_t = time()
                    dt = cur_t - last_print_t
                    if dt >= mininterval and cur_t >= min_start_t:
                        self.update(int(n - last_print_n))
                        last_print_n = self.last_print_n
                        last_print_t = self.last_print_t
        # except IndexError as e:
        #     print("Index Error", len(self.accumulated), n, e)
        finally:
            self.n = int(n)
            self.close()
            logger.info(
                f"Finished at {self.n}{self.unit} after {self.format_interval(self.last_print_t - self.start_t)}."
            )

    def step(self, a):
        """Hold the progress bar for one iteration."""
        self.delta_a = a
