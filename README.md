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

`mka eval` runs every question in `corpus/questions.json` as both roles and prints a 0–100 score per case, then an overall score (the mean). Each case is scored from independent checks: expected sources cited (minus documents that role cannot see), must-contain strings present, and must-not-contain strings absent. The string checks live in `src/mka/eval.py`. Exit code is 1 if the overall score is not 100.

Ask needs a role: `-s` / `--sales` / `--role sales`, or `-t` / `--technician` / `--role technician`. A technician price ask is refused before retrieve. `mka clear` deletes the Pinecone index and Python caches (`__pycache__`, `.pyc`, `.pytest_cache`). It does not touch `corpus/` or `.env`. Ingest recreates the index and wipes the namespace before rebuilding. Hazard notes are extracted from marked sections at ingest and stapled under the answer; they are not an LLM call. If a current document conflicts with a `flagged_outdated` revision, the answer uses the current value and notes the old conflict. If two current documents disagree, the answer names both values and does not pick a winner. `--stats` on ingest or ask prints token cost, latency, and Pinecone usage.

## Write-up

Screening Part 3 (production, evaluation, extensibility) is in [writeup.md](writeup.md).