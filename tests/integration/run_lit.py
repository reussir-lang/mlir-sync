#!/usr/bin/env python3

import multiprocessing
import sys

from lit.main import main


def _force_fork_start_method() -> None:
    try:
        multiprocessing.set_start_method("fork", force=True)
    except Exception:
        pass


if __name__ == "__main__":
    _force_fork_start_method()
    sys.exit(main())
