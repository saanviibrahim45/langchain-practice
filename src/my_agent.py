# src/base_agent.py
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os
from typing import Dict, Any
import yfinance as yf

load_dotenv()



# ---- Simple tool function ----
def calculator_tool(expr: str) -> str:
    try:
        allowed = "0123456789+-*/(). "
        if any(c not in allowed for c in expr):
            return "Error: expression contains invalid characters."
        result = eval(expr)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"

def stock_close_tool (query:str) -> str:
    try:
        ticker_input = (query).strip().upper()
        allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-^ "
        if not ticker_input or any(c not in allowed for c in query):
            return "Error: expression contains invalid characters"
        ticker = yf.Ticker(ticker_input)
        price = None
        try:
            price = ticker.info['currentPrice']
            # print("debug: ",price)
            return f"{ticker} last price (approx): ${price:,.2f}"
        except Exception:
            pass
    except Exception as e:
        return f"Error fetching stock price: {e}"

# ---- Initialize LLM ----
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---- Node 1: tool_node ----
def tool_node(state: MessagesState) -> Dict[str, Any]:
    last_msg = state["messages"][-1]
    # use attribute access since it's a HumanMessage object
    user_text = getattr(last_msg, "content", "")
    if isinstance(last_msg, HumanMessage) and user_text.lower().startswith("calc:"):
        expr = user_text.split(":", 1)[1].strip()
        tool_out = calculator_tool(expr)
        return {"messages": [SystemMessage(content=f"Tool output: {tool_out}")]}
    elif isinstance(last_msg, HumanMessage) and user_text.lower().startswith("stock:"):
        ticker = user_text.split(":", 1)[1].strip()
        tool_out = stock_close_tool(ticker)
        return {"messages": [SystemMessage(content=f"Tool output: {tool_out}")]}
    else:
        return {"messages": [SystemMessage(content="No tool used.")]}

# ---- Node 2: llm_node ----
def llm_node(state: MessagesState) -> Dict[str, Any]:
    msgs = state.get("messages", [])
    # collect last few contents
    context = "\n".join(
        [f"{m.type}: {getattr(m, 'content', '')}" for m in msgs[-3:]]
    )
    prompt = f"Respond concisely to the conversation:\n{context}"
    resp = model.invoke([HumanMessage(content=prompt)])
    try:
        text = resp.content
    except Exception:
        text = str(resp)
    return {"messages": [AIMessage(content=text)]}

# ---- Graph setup ----
graph = StateGraph(MessagesState)
graph.add_node("tool_node", tool_node)
graph.add_node("llm_node", llm_node)
graph.add_edge(START, "tool_node")
graph.add_edge("tool_node", "llm_node")
graph.add_edge("llm_node", END)
graph = graph.compile()

# ---- Interactive loop ----
if __name__ == "__main__":
    print("LangGraph base agent ready. Type 'exit' to quit.")
    print("Use 'calc: 2+2' to try the calculator tool.")
    print("Use 'stock: TICKER' to try the stock tool. Anything else will be handled by the LLM.\n")


    conversation = []  # keep all messages here

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break

        conversation.append(HumanMessage(content=user_input))  # store user input

        # pass full message history into the graph
        result = graph.invoke({"messages": conversation})
        msgs = result.get("messages", [])

        # find and print AI reply
        ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
        if ai_msgs:
            reply = ai_msgs[-1].content
            print("AI:", reply)
            conversation.append(ai_msgs[-1])  # store AI reply
        else:
            print("Result:", msgs)

