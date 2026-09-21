from __future__ import annotations

import sys

from mka.config import Config
from mka.types import Role


def run_ask(cfg: Config, role: Role, query: str) -> int:
    del cfg, role, query
    print("ask is not implemented yet", file=sys.stderr)
    return 1
