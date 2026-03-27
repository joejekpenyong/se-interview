# Travel Assistant Agent

A production-ready LLM-powered travel assistant built with LangGraph, FastAPI, and Arize Phoenix. The agent helps users find travel information including weather, currency exchange rates, and destination recommendations.

## Features

- **Multi-tool agent** — web search, live weather lookup, and real-time currency conversion
- **Conversation memory** — persistent multi-turn conversations via LangGraph checkpointing
- **Full observability** — traces sent to Arize Phoenix Cloud with LLM-as-judge evaluations
- **Dockerized** — runs as a containerized service
- **Tested** — unit tests covering tools, agent structure, and API endpoints

## Prerequisites

- Python 3.12
- [Poetry](https://python-poetry.org/docs/#installation)
- Docker
- OpenAI API key
- OpenWeatherMap API key (free tier at openweathermap.org)
- ExchangeRate API key (free tier at exchangerate-api.com)
- Arize Phoenix Cloud account (free at app.phoenix.arize.com)

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/joejekpenyong/se-interview
cd se-interview
```

**2. Install dependencies**

```bash
poetry install
```

**3. Configure environment variables**

```bash
cp .env.example .env
```

Open `.env` and fill in all values:

```
OPENAI_API_KEY=your_openai_key
OPENWEATHER_API_KEY=your_openweather_key
EXCHANGE_RATE_API_KEY=your_exchangerate_key
PHOENIX_CLIENT_HEADERS=Authorization=Bearer your_phoenix_key
PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/s/your-space
PHOENIX_API_KEY=your_phoenix_key
```

## Running the app

**Option 1 — local development**

```bash
poetry run uvicorn api:app --reload
```

**Option 2 — Docker**

```bash
docker build -t travel-assistant .
docker run -p 8000:8000 --env-file .env travel-assistant
```

The API will be available at `http://localhost:8000`.

## API endpoints

### POST /chat

Send a message to the travel assistant.

**Request:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in Tokyo?", "thread_id": "session-1"}'
```

**Response:**

```json
{
  "response": "The current weather in Tokyo is..."
}
```

The `thread_id` field is optional and defaults to `"default"`. Use the same `thread_id` across requests to maintain conversation history.

### GET /health

```bash
curl http://localhost:8000/health
```

## Running tests

```bash
poetry run pytest tests.py -v
```

## Evaluation

The evaluation pipeline uses LLM-as-judge: GPT-4o-mini reads each span's input and output, scores it against a rubric, and attaches the result back to the span in Phoenix Cloud as a filterable annotation. This lets you filter, sort, and investigate failure cases at any scale without reading every conversation manually.

### Running the evaluation script

Make sure the app has been running long enough to generate traces in Phoenix Cloud, then:

```bash
poetry run python evaluate.py
```

The script connects to Phoenix Cloud, fetches all LLM spans from the project, runs the three evaluations below, and pushes the results back as span annotations. You will see the eval columns appear in the Phoenix spans view immediately after the script completes.

### Evaluation methods

**1. User frustration** (required)

Detects signs of user frustration in the conversation — ALL CAPS input, impatient phrasing, complaints about previous responses, or expressions of urgency. Uses binary classification rails.

```
Rails: frustrated / not_frustrated
```

A frustrated session signals a user at churn risk. Filtering Phoenix to frustrated spans lets you investigate the root cause and identify patterns across many conversations at once.

**2. Tool selection correctness** (custom)

Checks whether the agent called the right tool for the user's query. A weather question should trigger `get_weather`, a currency question should trigger `get_exchange_rate`, and general destination questions should use web search. Penalizes unnecessary or missing tool calls.

```
Rails: correct / incorrect
```

Wrong tool selection produces a worse answer. This eval makes the error rate visible and verifiable — a system prompt change that improves tool routing will show up immediately in the scores.

**3. Answer completeness** (custom)

Checks whether the agent fully addressed every part of the user's request — all sub-questions answered, all requested details provided. An agent that answers only part of a multi-part question scores incomplete.

```
Rails: complete / incomplete
```

An incomplete answer means the user has to ask a follow-up or go elsewhere. This eval surfaces the gap between what the user asked and what the agent actually delivered.

### Viewing results in Phoenix

After running the script, open your Phoenix Cloud project. Evaluation results are attached to LLM spans specifically — the spans where the prompt and response text live. To see them, filter the spans view to `span_kind == 'LLM'`. The eval columns will not appear on root LangGraph spans or tool spans.

Once filtered you can:

- Filter to `user_frustration = frustrated` to find all at-risk sessions
- Filter to `tool_selection_correctness = incorrect` to build a dataset of tool routing failures
- Sort by `answer_completeness` to find the worst responses first
- Use any combination as training data for prompt improvement

### Usage notes

The evaluation script currently fetches a maximum of 200 spans. For this project that is sufficient, but in a high-traffic production environment 200 spans may represent only a small and potentially stale slice of your traffic. To address this, increase the limit in `evaluate.py` or add a time window filter so the script always evaluates the most recent spans rather than the oldest 200.

## Project structure

```
se-interview/
├── agent.py          # LangGraph agent with tools and memory
├── api.py            # FastAPI server with Phoenix instrumentation
├── evaluate.py       # Phoenix evaluation script
├── tests.py          # Unit tests
├── Dockerfile        # Container definition
├── pyproject.toml    # Dependencies
├── .env.example      # Environment variable template
└── README.md         # This file
```

## Agent architecture

The agent is built on LangGraph with three nodes:

- `llm_call` — calls GPT-4o with the current conversation and available tools
- `tool_node` — executes any tools the LLM requested
- `should_continue` — routes back to the LLM if tools were called, or ends the graph

**Tools:**
- `DuckDuckGoSearchRun` — web search for general travel information
- `get_weather` — current weather for any city via OpenWeatherMap
- `get_exchange_rate` — live currency conversion via ExchangeRate API

**Memory:**
Conversation state is persisted using LangGraph's `MemorySaver` checkpointer. Each session is identified by a `thread_id` so multiple users can have independent conversations.

## Observability

All LLM calls and tool invocations are traced to Arize Phoenix Cloud via OpenTelemetry. Each trace shows:
- Full input/output for every LLM call
- Tool invocations with arguments and responses
- Token usage and latency per span
- Cost per request

## How it works

1. User sends a message to `/chat` with an optional `thread_id`
2. LangGraph loads the conversation history for that thread
3. GPT-4o decides which tools to call based on the user's message
4. All relevant tools are called in parallel
5. GPT-4o reads the tool results and generates a response
6. The full trace is sent to Phoenix Cloud
7. The response is returned to the user

## Production architecture notes

### Memory

`MemorySaver` stores conversation state in RAM inside a single process. This works in development but breaks in production where multiple container instances are running — each instance has its own memory and cannot see another instance's conversations. Replace `MemorySaver` with a Redis-backed checkpointer so all instances share the same conversation state.

### Scaling

The app is stateless once Redis handles memory, which means you can scale horizontally by running multiple container instances behind a load balancer. Add instances under load and remove them off-peak. No coordination between instances is needed.

### Secrets management

In development, secrets live in a `.env` file. In production, use a secrets manager such as AWS Secrets Manager or GCP Secret Manager. Secrets should be injected at runtime and never committed to the repository or baked into a Docker image.

### Cost

GPT-4o is the dominant cost driver at roughly $0.01 per request. In production, route simple single-tool queries to GPT-4o-mini (approximately 10x cheaper) and reserve GPT-4o for complex multi-part queries. Caching repeated queries — such as weather for popular destinations — will also reduce both cost and latency.

### Latency

The system prompt instructs the agent to identify all needed tools upfront and call them in a single parallel `tool_node` execution. This eliminated one redundant LLM call and reduced average latency from 8.7s to 8.2s. This optimisation was identified by comparing Phoenix traces before and after the change. Continued trace analysis is the most reliable way to find further latency improvements.

### CI/CD

Run `pytest tests.py` as a required step in your CI pipeline. No code should reach production without passing the unit test suite. Pair this with the evaluation script run against a staging environment to catch regressions in LLM behavior before they reach production traffic.

### Span processor

This app uses `SimpleSpanProcessor`, which sends each span to Phoenix immediately as it is created — one HTTP request per span. In production, switch to `BatchSpanProcessor`, which queues spans and sends them in batches. This is significantly more efficient under load and is what Arize recommends for production environments. You will see a warning about this in the server startup logs.