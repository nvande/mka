## AI Engineer Screening Exercise

#### Completed by Nicholas Vander Woude

#### Tools used: Cursor AI code editor (mostly Grok 4.6), Claude AI chat by Anthropic (Sonnet 5.5, for planning only - not for writeup), Structurizr for planning

This application is for demonstration purposes only and is not intended for deployment

# Meridian Knowledge Assistant

 `mka` (Meridian Knowledge Assistant) is a Command Line Interface for sales and technicians. It accomplishes safety-first retrieval by including explicit document receipts and safety warnings with every response. The system also minimizes the possibility of hallucination by requiring attribution for every claim that the system produces. It's built in Python 3 via uv for fast package management.

The main goal of this project was to avoid wrong answers at all costs. We bias towards being conservative. We cannot allow a hallucination to slip through. We will not make an operational mistake. To accomplish this, the following decisions were made:

- GPT-5.4-nano is used specifically for its low (~3%) hallucination rate and low latency ([https://github.com/vectara/hallucination-leaderboard/](https://github.com/vectara/hallucination-leaderboard/)).
- The vector store is Pinecone DB, chosen for low latency retrieval (~30ms in ideal conditions) and ease to set up.
- For chunking:
  - `mka` stores *some* documents in full -- this is an intentional choice which maintains maximal context proximity present the original document and reduces the risk of important context loss from improper chunking of spec sheets. GPT-5.4-nano's context window is large, and because of the lack of chat log or ReAct reasoning history to maintain, an overloaded context is not a concern for this MVP. This decision is also based on small file size in the example corpus.
  - Only FAQs, Service Procedures, and Diagnostic Procedures are split by question or procedure; they represent a clear repeated structure that can be optimized with relevance sorting with no risk of context loss.
- Every answer goes through an in-context gate, an initial generation, and a receipt pass that drops the answer if any citation id was missing or invented. This process is optimized with parallel execution of LLM calls.
- Every document lookup requires at max two Pinecone calls: an initial call which grabs documents semantically related to the question, and a second search which pulls spec sheets for any product mentioned in the first search.

Still, the system is also designed to be robust from a user perspective. A user is not required to explicitly state they are asking about Meridian documents; all questions are interpreted as being Meridian-product specific, and the corpus is assumed to be catalog-complete, both of which contribute to further reducing the chance of errors or hallucinations from attempting to reason about imagined products.

This project does not use LangChain. LangChain could provide a chat wrapper, a generic splitter / chunker, and a retriever. These are unneeded in the context of OpenAI and Pinecone SDKs. The work this project actually needs (role filters, citation receipts, fail-closed gates, verbatim safety notes) would require Python code regardless.  LangChain only hides those decisions outside of code, and does not make them easier to implement. There was no reason to add a larger dependency for those capabilities.

## Setup

`mka` requires Python 3.11+, [uv](https://docs.astral.sh/uv/), an [OpenAI API key](https://openai.com/api/), and a [Pinecone Vector  DB key](https://www.pinecone.io/).

After you've cloned the repo, to initialize the Python environment and the assistant, run:

```sh
uv sync
cp .env.example .env
```

Fill in `OPENAI_API_KEY` and `PINECONE_API_KEY` in `.env`.

## Ingest Documents

Before you can respond to queries with `mka`, you will first need to ingest the documents.

Ensure you have the correct corpus at `./corpus`:

```text
corpus/
  manifest.json
  questions.json
  docs/*.md
```

Only .md (markup) files are supported for this project.

After you've verified the corpus, to ingest the documents, run:

```sh
uv run mka ingest
```

On a successful ingest, you should see: 

`ingest complete: X chunks`

## Answering Queries

To ask a question, you can run (for example):

```sh
uv run mka ask -t "What is the rated lifting capacity of the MD-7000?"
```

Ask needs a role: `-s` / `--sales` / `--role sales`, or `-t` / `--technician` / `--role technician`.

## Obtaining usage statistics

`--stats` on ingest, ask, or eval prints token cost, latency, and Pinecone usage for that run.

# Running the tests

The project includes a basic unit testing suite. To run it:

```sh
uv run pytest
```

The project also has an evaluation script for end-to-end testing of the AI-enabled features. To test this, run:

```sh
uv run mka eval
```

`mka eval` will compute a score (0-100) for every question in `corpus/questions.json`. You also will see an overall score (the mean). Each case is scored from independent checks: expected sources cited (minus documents that role cannot see), must-contain strings present, and must-not-contain strings absent.

You can also run evaluations with stat tracing to monitor costs. To do this, run:

```sh
uv run mka eval --stats
```



## Write-up

Screening Part 3 (production, evaluation, extensibility) is in [writeup.md](writeup.md).
