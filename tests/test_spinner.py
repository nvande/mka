from __future__ import annotations

import io

from mka.spinner import Spinner


def test_spinner_writes_nothing_when_not_a_tty() -> None:
    stream = io.StringIO()
    spin = Spinner(stream=stream)
    spin.stop()
    assert stream.getvalue() == ""
