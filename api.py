import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from agent import build_agent
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

load_dotenv()

# register(
#     project_name="travel-assistant",
#     endpoint="http://localhost:6006/v1/traces",
#     auto_instrument=True,
#     verbose=True
# )

register(
    project_name="travel-assistant",
    auto_instrument=True
)
#LangChainInstrumentor().instrument()

agent = build_agent()

app = FastAPI(
    title="LangGraph Agent API",
    description="A simple API for interacting with a LangGraph agent that can search the web",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Send a message to the agent and get a response."""
    config = {"configurable": {"thread_id": request.thread_id}}
    result = agent.invoke({"messages": [HumanMessage(content=request.message)]}, config)
    return ChatResponse(response=result["messages"][-1].content)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}
