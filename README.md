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

`mka scan` only needs the corpus. `mka ingest` and in-scope `mka ask` need the API keys.

```sh
uv run pytest
```

## Run

```sh
uv run mka ingest
uv run mka scan
uv run mka ask --role sales "<query>"
uv run mka ask --role technician "<query>"
```

`--role` is required on `ask` and must be `sales` or `technician`. Ingest wipes the Pinecone namespace before rebuilding the vector DB.