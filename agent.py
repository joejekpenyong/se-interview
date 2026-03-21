import operator 
import os       # For accessing environment variables
import requests # For making HTTP requests to external APIs

from typing import Annotated, Literal 
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage 
from langchain_core.tools import tool # For defining custom tools that the agent can use
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver # For saving the state of the agent's interactions
from typing_extensions import TypedDict

load_dotenv()

# ── Existing tool ────────────────────────────────────────────────────────────
search_tool = DuckDuckGoSearchRun()

# ── New tool 1: Currency converter ───────────────────────────────────────────
@tool
def get_exchange_rate(base_currency: str, target_currency: str) -> dict:
    """Get the current exchange rate between two currencies.
    
    Args:
        base_currency: The currency to convert from (e.g. USD, EUR, GBP)
        target_currency: The currency to convert to (e.g. JPY, NGN, CAD)
    
    Returns:
        A dict with the exchange rate and currency pair
    """
    api_key = os.getenv("EXCHANGE_RATE_API_KEY")
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{base_currency}/{target_currency}"
    
    response = requests.get(url)
    data = response.json()
    
    if data["result"] == "success":
        return {
            "base_currency": base_currency.upper(),
            "target_currency": target_currency.upper(),
            "exchange_rate": data["conversion_rate"],
            "last_updated": data["time_last_update_utc"],
            "disclaimer": "Exchange rates fluctuate daily. Please check again closer to your travel date for the most accurate rate."

        }
    else:
        return {"error": f"Could not fetch exchange rate: {data.get('error-type', 'unknown error')}"}

# ── New tool 2: Weather lookup ────────────────────────────────────────────────
@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city.
    
    Args:
        city: The name of the city to get weather for (e.g. Paris, Tokyo, Lagos)
    
    Returns:
        A dict with current weather conditions
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    
    response = requests.get(url)
    data = response.json()
    
    if response.status_code == 200:
        return {
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature_celsius": data["main"]["temp"],
            "feels_like_celsius": data["main"]["feels_like"],
            "condition": data["weather"][0]["description"],
            "humidity_percent": data["main"]["humidity"],
            "wind_speed_mps": data["wind"]["speed"]
        }
    else:
        return {"error": f"Could not fetch weather for {city}: {data.get('message', 'unknown error')}"}

# ── Tool registry ─────────────────────────────────────────────────────────────
tools = [search_tool, get_exchange_rate, get_weather]
tools_by_name = {tool.name: tool for tool in tools}

model = ChatOpenAI(model="gpt-4o", temperature=0)
model_with_tools = model.bind_tools(tools)

# ── Graph state ───────────────────────────────────────────────────────────────
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# ── Graph nodes ───────────────────────────────────────────────────────────────
def llm_call(state: MessagesState) -> dict:
    """Call the LLM with the current messages and available tools."""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a helpful travel assistant. You can search the web, "
                            "get current exchange rates between currencies, and check the weather "
                            "in any city. Use the appropriate tool when the user asks about "
                            "currency conversion or weather. Use web search for attractions, "
                            "hotels, flights, and general travel information."
                        )
                    )
                ]
                + state["messages"]
            )
        ]
    }

def tool_node(state: MessagesState) -> dict:
    """Execute tool calls from the last message."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    """Determine whether to continue to tool execution or end."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END

# ── Build agent ───────────────────────────────────────────────────────────────
def build_agent():
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("llm_call", llm_call)
    graph_builder.add_node("tool_node", tool_node)
    graph_builder.add_edge(START, "llm_call")
    graph_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    graph_builder.add_edge("tool_node", "llm_call")
    
    checkpointer = MemorySaver()
    agent = graph_builder.compile(checkpointer=checkpointer)
    return agent