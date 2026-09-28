## AI Engineer Screening Exercise - RiteHite Hiring Process

#### Completed by Nicholas Vander Woude

#### Tools used: Cursor AI code editor (mostly Grok 4.6), Claude AI chat by Anthropic (Sonnet 5.5, for planning only - not for writeup), Structurizr for planning

This application is for demonstration purposes only and is not intended for deployment

## 1. Production readiness

*What would you change to make this scalable and reliable?*

The assistant was designed from a guardrail and retrival standpoint with production in mind, but it is not production ready.

Notably:

- It's only a CLI. In a production setting, we'd need an API layer, so mka could respond to queries as an MCP server. To scale, I'd choose AWS Lambda behind a API Gateway for security, authentication, rate limiting, etc. If a more stateful application was needed, I think ECS Fargate might be worth considering after containerization. The tradeoff here is easy scale with Lambda to support high peaks, or ECS for a more deliberate environment supported by a containerized approach. With Lambda, we need to be aware of cold starts and consider a concurrency provision strategy.
- It's stateless; the assistant doesn't need to maintain context for ongoing conversation. A production application may require state, depending on if conversation history should live on the UI host or on the server. Chat history could also be handled with the OpenAI API. The decision here will be primarily driven by technical specifications of the host machine. Conversations should never be long, so ideally they will live on the host, but we may desire statefulness / persistance to offload conversation state to the server if we expect to support extended user sessions.
- The small corpus & manifest means I can trust which documents are outdated without preprocessing. In a production environment, ingestion requires building a pipeline that can pull metadata, use an LLM alongside a schema to fill in the rest (audience, product, expiration), and re-elevate anything too low-confidence for human review. This needs to run automatically whenever a doc is added or changed.
- Even with a provided manifest, we should not fall back to blind trust. An error-correction pipeline would be advantageous. Every ingested document should go through LLM classification for prediction % of errors or discrepency between the manifest metadata and the document content. Above a threshhold, I'd escalate the document and look to include a HITL who can be trusted to quickly identify and patch the document.
- For this exerise, I've allowed the assistant to assume the documents cover the full Meridian product line. This allows the system to answer queries like, "What is the lowest temperature a meridian dock leveler will work at?" In a real production scenario, we would be required to either ensure our documents actually cover the breadth of the product catalog, or specifically disallow questions that require that confidence.
- It has no re-ranking strategy. It's not neccessary. The corpus is small, and the answer is almost certain to be within the top-k documents we pull for each question. Our chunks are relatively large, so reordering will provide no benefit. Re-ranking is only beneficial when we plan to pull a dozen or more chunks, and lifting the chunks with the best exact-word matches to the top would meaningfully impact the organization of the context window.
- The provided  `questions.json` gave me the answer key for readiness. Real world situations are not as clean: A production assistant would require more iteration to determine success. We'd need to assemble a testing dataset ahead of time. We could source from: personalle interviews, existing Q&As, or potentially a science-aware model capable of producing extremely high quality questions to test the edge cases of thermodynamics, mechanical physics, etc. This would need to evolve organically over time as we identify weak points. Determining what success means is potentially the most challenging and time-consuming process of creating a production-ready RAG assistant.
- I'd consider Jev to replace classifiers. More investigation is needed here.



## 2. Evaluation strategy

*How would you measure answer quality over time?*

There are a number of ways we can measure answer quality:

- We can allow / require users to give a 1-5 score or thumbs up / down after each response, giving us insight to what types of questions we answer the worst compared to what the question answer wanted to see. Great starting point for observability of answer quality. It's important to remind them this score helps improve the system, in turn making their job easier.
  - 1-5 is best when we just want to gauge which answers are best, middling, or failures. Thumbs up / down is better for incentivizing specific feedback, ie, "What didn't you like about this response?"
- Build a system to check which questions get repeated or restated. Does the user seem to be asking the same or similar answers over and over again? This is a good signal of a frustation point and we can investigate that topic. We can consider adding monitors for these events, making it immediately noticable on a dashboard if users are, 'bouncing off' the assistant without receiving worthwile answers.
- Log all questions and answers. Use a seperate, more expensive model to assess selections from that dataset. Do our answers classify as high quality or are we providing under-reasoned answers that don't get to the heart of the query?
- Observability. We need interactive, reactive dashboards that tell us everything knowable about assistant usage. Climbing steadily? That's a good signal that technicians are organically turning towards the assistant to answer their queries. Set up alerts for any abborant behavior: usage above or below expected thresholds.
- Overall performance of the team using the assistant. Did a measurable positive metric go up since we deployed the system? Are other measurements that typicall signal a lack of in-the-moment knowledge (delays, service spend) falling behind?



## 3. Extensibility

*How would you integrate a second data source (e.g., ERP or ticketing API)?*

I'd bring in either an ERP or a ticketing API as an extra tool call we make deterministically with fresh data. An upsert to the VDB would cause a nightmare for concurrency. Some details:

- To find correct data, we'd need to build a solution which is tailored specifically for each data source. If the source already supports semantic search, that's good; we can easily pass keywords to the data source in parallel to our VDB cosine similarity lookup and merge the two with no latency cost. If semantic search is not supported, we may need a custom solution. Agent Search on Gemini Enterprise Agent Platform is one possible example for how we can retrieve relevant data from live sources.
  - If the user already gave us a key (ticket number, order id, SKU), call that lookup directly. That skips semantic search, which matters because a near-miss hit is how a similar but wrong record gets into the answer. I'd only use the service's own search when there is no key, and I'd still only receipt the rows that search returned. I don't know these APIs yet, so I'd read them before deciding which calls are lookups and which are search.
- To keep near-zero hallucinations, these snippets should be treated exactly the same as chunks pulled from the VDB: we'd enforce role access, check receipts for every claim made, and staple the attribution to the response.
  - The role filter wraps the tool the same way it wraps retrieval. A technician call does not come back with pricing just because the ERP has it.
- If a tool error occurs, or a timeout, or an empty body fails closed, we simply refuse. The model cannot not fill the gap.
- Eval has to grow with the tools as well as the document set. Fixture responses, expected source ids that include the tool record, and the same must-contain / must-not-contain checks. If a change lets an answer through without a receipt against a real chunk or a real tool payload, that case fails. Citation is considered as important as the answer itself.
- If the second data source was a video library, for example how-to or safety videos, we'd likely want to process that library via audio-to-text transcription, and create a second VDB for semantic search on the video library. This potentially also requires some timely in-transcript descriptions of what is happening visually on screen at key moments. Video transcription chunking is a seperate problem entirely, with completely different trade-offs to document transcription, and would require specific work required to ensure we can reference the correct video and the correct timestamp relevant to the user's query.

