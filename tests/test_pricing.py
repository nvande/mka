from mka.pricing import contains_pricing

STAINLESS_LIP = (
    "Stainless steel lip option (+$1,200) is recommended for direct "
    "food-contact zones or caustic wash-down environments."
)

COLD_STORAGE_OTHER = (
    "The standard recommendation is an MD-9000 air-powered dock leveler "
    "paired with a ThermaGuard 600 insulated sectional door."
)


def test_price_dollar_sign() -> None:
    assert contains_pricing(STAINLESS_LIP) is True


def test_list_price_and_addon_phrase() -> None:
    assert contains_pricing("MD-7000 list price is published annually.") is True
    assert contains_pricing("See the add-on price for heated tracks.") is True


def test_no_price_signal() -> None:
    assert contains_pricing(COLD_STORAGE_OTHER) is False
    assert contains_pricing("") is False
