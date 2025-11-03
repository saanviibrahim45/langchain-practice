from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import Dict, Any
import os
import requests # Python library for making HTTP requests
import yfinance as yf
from dotenv import load_dotenv
import re
import time

load_dotenv()

ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
if not ALPHA_VANTAGE_KEY:
    raise ValueError("Alpha Vantage API key missing in .env")

# Alpha Vantage API call
def call_alpha_vantage(ticker: str):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED", # tells the API what kind of data you want
        "symbol": ticker,
        "outputsize": "compact",
        "apikey": ALPHA_VANTAGE_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=10) # Makes an HTTP GET request to the URL you specified
        response.raise_for_status() # Checks the HTTP status code of the response.
        data = response.json()
        if "Error Message" in data:
            raise RuntimeError(f"Alpha Vantage API error: {data['Error Message']}")
        return data
    except Exception as e:
        print(f"Alpha Vantage call failed: {e}")
        return {}


# Yahoo Finance API call
def call_yfinance(symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1mo")  # last month of daily data
        news_list = []  
        return {
            "prices": hist.reset_index().to_dict(orient="records"), # Converts the hist DataFrame to a list of dictionaries, one per row (record), with reset_index() to include the Date as a field.
            "news": news_list
        }
    except Exception as e:
        print(f"Yahoo Finance call failed: {e}")
        return {}

# unified tool caller
def call_tool(tool_name: str, ticker: str):
    normalized = tool_name.lower().replace(" ", "")  # remove spaces, lowercase
    if normalized == "alphavantage":
        return call_alpha_vantage(ticker)
    elif normalized == "yahoofinance":
        return call_yfinance(ticker)
    else:
        raise ValueError(f"Unknown tool: {tool_name}")


# Local state passed in: messages, global state
def fetcher_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    ticker = getattr(state["global_state"], "ticker", None)
    tools_to_call = getattr(state["global_state"], "tools_to_call", [])

    tool_results = {}
    successful_tools = []

    print(f"DEBUG: Fetching data for {ticker} using tools: {tools_to_call}")

    for tool in tools_to_call:
        retries = 3
        success = False
        for attempt in range(1, retries + 1):
            try:
                result = call_tool(tool, ticker)
                print(f"DEBUG: {tool} returned: {result}")
                tool_results[tool] = result
                success = True
                successful_tools.append(tool)
                break
            except Exception as e:
                print(f"Attempt {attempt} failed for {tool}: {e}")
                time.sleep(1)
        if not success:
            print(f"Persistent failure calling {tool}")
            continue

        print(f"DEBUG: Tool results: {tool_results}")

    # Update global state
    state["global_state"].raw_api_data = tool_results
    state["global_state"].status = "fetched"

    # Optional: append summary message
    ai_msg = AIMessage(content=f"Fetched and normalized data from tools: {', '.join(successful_tools)}")

    # Return JSON envelope
    return {
        "status": "success" if successful_tools else "partial_failure",
        "data": {"raw_api_data": tool_results},
        "metadata": {
            "api_sources": successful_tools,
            "task_id": "fetch_001", #not the best to hardcode
            "parent_task_id": "query_001"
        },
        "messages": [ai_msg]
    }