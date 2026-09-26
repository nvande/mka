# Meridian Knowledge Assistant

Knowledge assistant for internal product, service, and related docs.

Powered by OpenAI API + Pinecone Vector DB

## Setup

Needs Python 3.11+, [uv](https://docs.astral.sh/uv/), an OpenAI key, and a Pinecone key.

```sh
uv sync
cp .env.example .env
```

Fill in `OPENAI_API_KEY` and `PINECONE_API_KEY` in `.env`.

Obtain the corpus separately and put it at `./corpus`:

```text
corpus/
  manifest.json
  docs/*.md
```

`mka ingest`, `mka clear`, and in-scope `mka ask` need the API keys.

```sh
uv run pytest
```

## Run

```sh
uv run mka clear
uv run mka ingest
uv run mka ask -s "<query>"
uv run mka ask -t "<query>"
uv run mka eval
```

`mka eval` runs every question in `corpus/questions.json` as both roles and prints PASS/FAIL per case. A case passes when the expected sources are cited (minus documents that role cannot see), the must-contain strings appear, and the must-not-contain strings do not. The string checks live in `src/mka/eval.py`. Exit code is 1 if any case fails.

Ask needs a role: `-s` / `--sales` / `--role sales`, or `-t` / `--technician` / `--role technician`. `mka clear` deletes the Pinecone index and Python caches (`__pycache__`, `.pyc`, `.pytest_cache`). It does not touch `corpus/` or `.env`. Ingest recreates the index and wipes the namespace before rebuilding. If a current document conflicts with a `flagged_outdated` revision, the answer uses the current value and notes the old conflict. If two current documents disagree, the answer names both values and does not pick a winner.

## Write-up

Screening Part 3 (production, evaluation, extensibility) is in [writeup.md](writeup.md).