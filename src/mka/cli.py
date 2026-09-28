from __future__ import annotations

import argparse

from mka.ask import run_ask
from mka.clear import run_clear
from mka.config import load_config
from mka.eval import run_eval
from mka.ingest import run_ingest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="mka",
        description="Meridian Knowledge Assistant",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--stats",
        action="store_true",
        help="Print token cost, latency, and Pinecone usage for this run",
    )

    sub.add_parser(
        "ingest",
        parents=[common],
        help="Join the manifest, chunk, and upsert to Pinecone",
    )

    ask = sub.add_parser(
        "ask",
        parents=[common],
        help="Ask a question against the index",
    )
    role = ask.add_mutually_exclusive_group(required=True)
    role.add_argument(
        "-s",
        "--sales",
        action="store_const",
        const="sales",
        dest="role",
        help="same as --role sales",
    )
    role.add_argument(
        "-t",
        "--technician",
        action="store_const",
        const="technician",
        dest="role",
        help="same as --role technician",
    )
    role.add_argument("--role", choices=("sales", "technician"))
    ask.add_argument("query")

    sub.add_parser(
        "clear",
        help="Wipe the Pinecone index and Python caches",
    )

    sub.add_parser(
        "eval",
        help="Run corpus/questions.json under both roles and print 0-100 scores",
    )

    args = parser.parse_args(argv)
    cfg = load_config()

    if args.command == "ingest":
        raise SystemExit(run_ingest(cfg, stats=args.stats))
    if args.command == "ask":
        raise SystemExit(run_ask(cfg, args.role, args.query, stats=args.stats))
    if args.command == "clear":
        raise SystemExit(run_clear(cfg))
    if args.command == "eval":
        raise SystemExit(run_eval(cfg))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
