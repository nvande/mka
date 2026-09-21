from __future__ import annotations

import re

PRICE = re.compile(r"\$|list price|add-on price", re.I)


def contains_pricing(text: str) -> bool:
    return bool(PRICE.search(text))
