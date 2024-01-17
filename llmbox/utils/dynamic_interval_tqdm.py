import tqdm


class dynamic_interval_tqdm(tqdm.tqdm):

    def __init__(self, iterable=None, intervals=None, desc=None, disable=False, unit='it',
                 dynamic_ncols=False, total=None, **kwargs):
        super().__init__(iterable=iterable, desc=desc, disable=disable, unit=unit, unit_scale=False,
                         dynamic_ncols=dynamic_ncols, total=total, **kwargs)
        self.intervals = intervals
        if len(intervals) != total:
            raise ValueError(f"Length of intervals {len(intervals)} does not match total {total}.")

    def __iter__(self):
        """Overwrite the original tqdm iterator to support dynamic intervals."""

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
                # Update the progress bar with dynamic intervals
                n += 1 / self.intervals[int(n)]

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
