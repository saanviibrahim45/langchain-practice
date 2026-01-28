from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import Dict, Any
from dotenv import load_dotenv
import re
import time

load_dotenv()

# ---- Initialize LLM ----
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# helper function to extract tools identified by LLM
def parse_tools(text: str) -> list[str]:
    """
    Extracts tool names by searching for 'Alpha Vantage' and 'Yahoo Finance' in the text.
    """
    tools = []
    
    # Search for Alpha Vantage (case-insensitive)
    if re.search(r"alpha\s*vantage", text, re.IGNORECASE):
        tools.append("Alpha Vantage")
    
    # Search for Yahoo Finance (case-insensitive)
    if re.search(r"yahoo\s*finance", text, re.IGNORECASE):
        tools.append("Yahoo Finance")
    
    return tools


# Local state passed in: messages, global state
def query_agent(state):
    #precondition: ticker must exist
    ticker = getattr(state["global_state"], "ticker", None)
    msgs = state.get("messages", [])

    # collect last message
    context = "\n".join(
        [f"{m.type}: {getattr(m, 'content', '')}" for m in msgs[-1:]]
    )

    prompt = f"""You are a financial assistant. You can only use the following APIs/tools: 
    - Alpha Vantage
    - Yahoo Finance

    Your goal is to identify either/both of these tools are required to summarize the stock performance of {ticker}.

    User context: {context}

    Respond ONLY with the tool names separated by commas, nothing else.
    Example response: "Alpha Vantage, Yahoo Finance" 
    """
    resp = model.invoke([HumanMessage(content=prompt)])
    try:
        text = resp.content
        print(f"DEBUG: LLM response: {text}")
    except Exception:
        text = str(resp)

    # parse tools from LLM output
    tools_to_call = parse_tools(text)

    print(f"DEBUG: Parsed tools: {tools_to_call}")

    # update global state
    state["global_state"].tools_to_call.extend(tools_to_call)

    # append new msg to convo
    ai_msg = AIMessage(content=text)

    state["global_state"].status = "planned"

    # return JSON envelope
    return {
        "status": "planned",
        "messages": [ai_msg],
        "data": {"tools_to_call": tools_to_call},
        "metadata": {"task_id": "query_001", "parent_task_id": None}
    }

    