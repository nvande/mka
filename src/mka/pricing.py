from __future__ import annotations

import re

# doc_type is not enough. FAQ rows are tagged faq and still contain prices.
PRICE = re.compile(r"\$|list price|add-on price", re.I)

# A query asking for a price. Broader than PRICE: users say "cost" and
# "quote", not "list price".
PRICE_ASK = re.compile(r"\$|\bpric(?:e|es|ed|ing)\b|\bcosts?\b|\bquotes?\b", re.I)


def contains_pricing(text: str) -> bool:
    return bool(PRICE.search(text))


def asks_for_price(query: str) -> bool:
    return bool(PRICE_ASK.search(query))
