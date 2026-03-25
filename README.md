# Travel Assistant Agent

An LLM-powered travel assistant built with LangGraph, FastAPI, and Arize Phoenix. The agent helps users find travel information including weather, currency exchange rates, and destination recommendations.

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

## Running evaluations

Make sure your Phoenix Cloud credentials are set in `.env`, then:

```bash
poetry run python evaluate.py
```

This fetches your traces from Phoenix Cloud and runs three LLM-as-judge evaluations:
- **User frustration** — detects signs of user frustration in conversations
- **Tool selection correctness** — checks whether the agent chose the right tools
- **Answer completeness** — checks whether the agent fully addressed the user's request

Results are pushed back to Phoenix Cloud as span annotations.

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