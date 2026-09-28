from __future__ import annotations

import itertools
import sys
import threading

FRAMES = r"\|/-"


class Spinner:
    """Rotate \\ | / - on stderr so a long ask/ingest does not look frozen."""

    def __init__(self, stream=None) -> None:
        self._stream = stream or sys.stderr
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if self._stream.isatty():
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        for frame in itertools.cycle(FRAMES):
            self._stream.write(f"\r{frame}")
            self._stream.flush()
            if self._stop.wait(0.1):
                break
        self._stream.write("\r \r")
        self._stream.flush()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=0.5)
        self._thread = None
