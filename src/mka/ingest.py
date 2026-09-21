from __future__ import annotations

import sys

from mka.config import Config


def run_ingest(cfg: Config) -> int:
    del cfg
    print("ingest is not implemented yet", file=sys.stderr)
    return 1
