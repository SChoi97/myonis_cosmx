"""Progress output for interactive use and Slurm .out files."""

import sys
import time


def progress(items, *, total=None, desc="Progress", unit="item"):
    try:
        from tqdm import tqdm
    except ImportError:
        if total is None:
            try:
                total = len(items)
            except TypeError:
                pass
        last = time.monotonic()
        count = 0
        print(f"{desc}: 0/{total if total is not None else '?'} {unit}", flush=True)
        for index, item in enumerate(items, 1):
            yield item
            count = index
            now = time.monotonic()
            if now - last >= 10 or index == total:
                print(f"{desc}: {index}/{total} {unit}", flush=True)
                last = now
        if total is None:
            print(f"{desc}: {count} {unit} completed", flush=True)
        return
    yield from tqdm(items, total=total, desc=desc, unit=unit, file=sys.stdout,
                    mininterval=10, ncols=110, dynamic_ncols=False)
