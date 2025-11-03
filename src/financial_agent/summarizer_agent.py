from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import Dict, Any
from dotenv import load_dotenv
import re

load_dotenv()

# ---- Initialize LLM ----
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Local state passed in: messages, global state
def summarizer_agent(state):
    tool_results = getattr(state["global_state"], "raw_api_data", [])
    ticker = getattr(state["global_state"], "ticker", "")
    msgs = state.get("messages", [])

    print(f"DEBUG: Summarizer received data: {tool_results.keys()}")

    # collect last 3 messages
    context = "\n".join(
        [f"{m.type}: {getattr(m, 'content', '')}" for m in msgs[-3:]]
    )
    
    prompt = f"""
    You are a financial analyst. Summarize the stock performance for {ticker} based on this data:
    
    {tool_results}
    
    Provide specific statistics, trends, and insights.
    """

    resp = model.invoke([HumanMessage(content=prompt)])
    try:
        text = resp.content
    except Exception:
        text = str(resp)
    
    # update global state
    state["global_state"].final_response = text
    state["global_state"].status = "success"

    # append new msg to convo
    ai_msg = AIMessage(content=text)

    return {
        "status": "complete",
        "messages": [ai_msg],
        "data": {"summary_text": text},
        "metadata": {
            "task_id": "summarize_001",  
            "parent_task_id": "fetch_001"
        }
    }
