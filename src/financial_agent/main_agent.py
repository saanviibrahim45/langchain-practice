from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os
from query_agent import query_agent
from fetcher_agent import fetcher_agent
from summarizer_agent import summarizer_agent
from langchain_core.runnables import RunnableLambda
from typing import Dict, Any, Literal, TypedDict, Annotated
import re
import operator


load_dotenv()

# ---- Initialize LLM ----
# ChatOpenAI automatically reads OPENAI_API_KEY from os.environ
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# define global state
class GlobalState:
    def __init__(self):
        self.ticker = ""
        self.tools_to_call = []
        self.raw_api_data = []
        self.status = "idle"
        self.final_response = ""

global_state = GlobalState()

# Define the state schema for LangGraph
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # Messages accumulate
    global_state: GlobalState  # Your global state object

# extract ticker fxn
def extract_ticker(text: str) -> str:
    """Extracts a ticker symbol (1-5 uppercase letters) from the text."""
    match = re.search(r"\b[A-Z]{1,5}\b", text)
    return match.group(0) if match else ""

# Router function - decides which path to take
def route_query(state):
    """Routes to stock lookup or general conversation based on user input."""
    last_msg = state["messages"][-1]
    
    if isinstance(last_msg, HumanMessage):
        user_text = getattr(last_msg, "content", "")
        if user_text.lower().startswith("lookup stock:"):
            ticker = extract_ticker(user_text)
            if ticker:
                state["global_state"].ticker = ticker
                return "query_agent"
    
    return "general_llm_node"

# Precondition: No tool used
def general_llm_node(state) -> Dict[str, Any]:
    msgs = state.get("messages", [])
    # collect last few contents
    context = "\n".join(
        [f"{m.type}: {getattr(m, 'content', '')}" for m in msgs[-3:]]
    )
    prompt = f"""Respond concisely to the conversation, prioritize most recent message (the last one). 
    Other messages may be responses from tool invocation: \n{context}"""
    resp = model.invoke([HumanMessage(content=prompt)])
    try:
        text = resp.content
    except Exception:
        text = str(resp)
    return {"messages": [AIMessage(content=text)]}


# ---- Graph setup ----
graph = StateGraph(AgentState)

graph.add_node("general_llm_node", general_llm_node)
graph.add_node("query_agent", query_agent)
graph.add_node("fetcher_agent", fetcher_agent)
graph.add_node("summarizer_agent", summarizer_agent)    

# Start with conditional routing
graph.add_conditional_edges(
    START,
    route_query,  # This function returns either "query_agent" or "general_llm_node"
    {
        "query_agent": "query_agent",
        "general_llm_node": "general_llm_node"
    }
)


graph.add_edge("query_agent", "fetcher_agent")
graph.add_edge("fetcher_agent", "summarizer_agent")

graph.add_edge("summarizer_agent", END)
graph.add_edge("general_llm_node", END)

graph = graph.compile()

# ---- Interactive loop ----
if __name__ == "__main__":
    print("LangGraph base agent ready. Type 'exit' to quit.")
    print("Use 'lookup stock: <stock ticker>' to try the stock lookup tool. Anything else will be handled by the LLM.\n")

    initial_state = {
        "messages": [],          # conversation messages
        "global_state": global_state
    }

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break

        initial_state["messages"].append(HumanMessage(content=user_input))  # store user input into convo history

        # pass full message history into the graph -> pass as LOCAL state (dictionary)
        result = graph.invoke(initial_state)

        # after all nodes in graph have been visited
        msgs = result.get("messages", [])

        # find and print AI reply
        ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
        if ai_msgs:
            reply = ai_msgs[-1].content
            print("AI:", reply)
        else:
            print("Result:", msgs)
