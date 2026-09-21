from __future__ import annotations

import argparse

from mka.ask import run_ask
from mka.config import load_config
from mka.ingest import run_ingest
from mka.scan import run_scan


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="mka",
        description="Meridian Knowledge Assistant",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Join the manifest, chunk, and upsert to Pinecone")

    ask = sub.add_parser("ask", help="Ask a question against the index")
    ask.add_argument("--role", required=True, choices=("sales", "technician"))
    ask.add_argument("query")

    sub.add_parser("scan", help="List flagged_outdated catalog rows (PoC stub)")

    args = parser.parse_args(argv)
    cfg = load_config()

    if args.command == "ingest":
        raise SystemExit(run_ingest(cfg))
    if args.command == "ask":
        raise SystemExit(run_ask(cfg, args.role, args.query))
    if args.command == "scan":
        raise SystemExit(run_scan(cfg))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
