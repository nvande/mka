## AI Engineer Screening Exercise - RiteHite Hiring Process

#### Completed by Nicholas Vander Woude on 09/21/26

#### Tools used: Cursor AI Code editor, Claude AI chat by Anthropic, Structurizr for planning

This application is for demonstration purposes only and is not intended for deployment

# Meridian Knowledge Assistant

I built mka (Meridian Knowledge Assistant) as a Command Line Interface for sales and technicians. It accomplishes safety-first retrieval by including explicit document receipts and safety warnings with every response. The system also minimizes the possibility of hallucination by requiring attritbution for every claim that the system produces. It's built in Python 3 via uv for fast package management.

- I've selected GPT-5.4-nano specifically for its low hallucination rate / speed tradeoff ([https://github.com/vectara/hallucination-leaderboard/](https://github.com/vectara/hallucination-leaderboard/)).
- The vector store is Pinecone DB, which I've chosen for low latency retrieval and ease to set up.
- For chunking, we store most documents in full -- this maintains maximal context proximity of from the original document and reduces the risk of important context loss from improper chunking. GPT-5.4-nano's context window is large, and because of the lack of long chat log or reasoning history to maintain, an overloaded context is not of significant concern for this MVP. Only FAQs, Service Procedures, and Diagnostic Procedures are split, because they represent a clear repeated structure that we can optimize with relevance sorting.

The system is also designed to be robust from a user perspective: a user is not required to explicitly state they are asking about Meridian documents, as all questions will be interpetted as being Meridian-product specific, further reducing the chance of errors or hallucinations.

I chose not to use LangChain. It would have given me a chat wrapper, a generic splitter / chunker, and a retriever. These are unneeded with the OpenAI and Pinecone SDKs. The work this project actually needs (role filters, citation receipts, fail-closed gates, verbatim safety notes) would regardless require Python code. A chain here can actually hide those decisions outside of code, and wouldn't have made them easier to implement. There was no reason to add a larger dependency capabilities I could quickly build myself.

#### Tools used: Cursor AI Code editor, Claude AI chat by Anthropic, Structurizr for Planning



## 1. Production readiness

*What would you change to make this scalable and reliable?*

The assistant was designed from a guardrail and retrival standpoint with production in mind, but it is not production ready.

Notably:

- It's stateless, because the assistant doesn't need to maintain context for ongoing conversation. A production application may require a need for state, depending on if conversation history should live on the UI host or on the server. This is primarily driven by the technical specifications of the host machine. Conversations will never be long, so ideally they will live on the host, but we may desire statefulness / persistance to offload conversation state to the server.
- The small corpus & manifest means I can trust which documents are outdated without preprocessing. In a production environment, ingestion requires building a pipeline that can pull metadata, use an LLM alongside a schema to fill in the rest (audience, product, expiration), and re-elevate anything too low-confidence for human review. This needs to run automatically whenever a doc is added or changed.
- Even with a provided manifest, we should not blindly trust the manifest. Some amount of error-correction pipeline would be advantageous. For this, I'd choose to require every ingested document to still go through a cheap LLM classifier to calculate the percentage chance of errors or discrepency between the manifest metadata and the document content. Above a certain threshhold, I'd escalate the document and look to include a HITL who can be trusted to quickly identify and patch the document.
- The provided  `questions.json` essentially gave me the answer key quality of answers ahead of time. Real world situations are not as clean. A production assistant would require more iteration, including a "thumbs up/down" feedback system for users so we can improve answer quality over time. We'd also be incentivized to grow our own testing dataset ahead of time, including doing personalle interviews, and using a science-aware model capable of producing extremely high quality questions to test the edge cases of thermodynamics, mechanical physics, etc
- It's only a CLI. In a production setting, we'd need an API layer, so mka could respond to queries as an MCP server. To scale, I'd choose AWS Lambda behind a API Gateway for security, authentication, rate limiting, etc. If a more stateful application was needed, I think ECS Fargate might be worth considering after containerizing the app. The tradeoff here is easy scale with Lambda to support high peaks, or ECS for a more deliberate environment supported by a containerized approach.
- It has no re-ranking strategy. It's simply not neccessary. The corpus is small, and the answer is almost certain to be within the 8 documents we pull for each question. Our chunks are relatively large, so reordering will provide no benefit. Re-ranking is only beneficial when
- We have allowed, for this exercise only, to assume the documents cover the full Meridian product line. This allows the system to answer queries like, "What is the lowest temperature a meridian dock leveler will work at?" In a real production scenario, we would be required to either nsure our documents actually cover the breadth of the product catalog, or specifically disallow questions that require that confidence.

## 2. Evaluation strategy

*What would you change to make this scalable and reliable?*

There are a number of ways we can measure answer quality:

- We can allow / require users to give a 1-5 score with each query responses, giving us insight to what types of questions we answer the worst compared to what the question answer wanted to see
- Build a system to check which questions get repeated or restated. Does the user seem to be asking the same or similar answers over and over again? This is a good signal of a frustation point and we can investigate that topic.
- Observability. We need interactive, reactive dashboards that tell us everything knowable about assistant usage. Climbing steadily? That's a good signal that technicians are organically turning towards the assistant to answer their queries.
- Overall performance on the floor. Did a measurable positive metric go up since we deployed the system?
- Log all questions and answers. Use a seperate, more expensive model to assess our work. Do our answers classify as high quality or are we providing under-reasoned answers that don't get to the heart of the query?

## 3. Extensibility

*How would you integrate a second data source (e.g., ERP or ticketing API)?*

- Second data sources, like ERP or ticketing API, should be treated as tool-calls, not a new type of document. There's a need for fresh data and any upsert process would cause a concurrency nightmare. Without knowing the technical specifications of the ticketing service or ERP, its impossible to make a concrete judgement, but I would investigate those specific services and investigate what semantic search options they provide to supply fast retrieval of highly-relevant information. When possible, we look for instances where immediate deterministic recall can circumvent the need for semantic search entirely.

