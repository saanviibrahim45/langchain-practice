# src/my_agent_template.py
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os
from typing import Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import reverse_text, word_count, text_uppercase, get_stock_info

load_dotenv()

# Initialize LLM
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Node 1: decision_node
def decision_node(state: MessagesState) -> Dict[str, Any]:
    last_msg = state["messages"][-1]
    user_text = getattr(last_msg, "content", "")
    
    if isinstance(last_msg, HumanMessage):
        # LLM classifies the intent
        classifier_prompt = f"""Analyze this user request and respond with ONLY one of these exact words:
- "reverse" if they want text reversed/flipped
- "count" if they want word counting/text analysis  
- "upper" if they want uppercase/capitalization conversion
- "stock" if they want stock price/market data
- "chat" for general conversation or if unclear

User input: "{user_text}"

Respond with just one word:"""
        
        classification = model.invoke([HumanMessage(content=classifier_prompt)])
        intent = classification.content.strip().lower()
        
        return {"messages": [SystemMessage(content=f"Intent: {intent}|{user_text}")]}
    else:
        return {"messages": [SystemMessage(content="Intent: chat|No user input")]}

# Node 2: tool_node  
def tool_node(state: MessagesState) -> Dict[str, Any]:
    last_msg = state["messages"][-1]
    
    if isinstance(last_msg, SystemMessage) and "Intent:" in last_msg.content:
        intent_data = last_msg.content.replace("Intent: ", "").split("|", 1)
        intent = intent_data[0]
        user_text = intent_data[1] if len(intent_data) > 1 else ""
        
        if intent == "reverse":
            result = reverse_text.invoke({"text": user_text})
            return {"messages": [SystemMessage(content=f"Tool output: {result}")]}
        elif intent == "count":
            result = word_count.invoke({"text": user_text})
            return {"messages": [SystemMessage(content=f"Tool output: {result}")]}
        elif intent == "upper":
            result = text_uppercase.invoke({"text": user_text})
            return {"messages": [SystemMessage(content=f"Tool output: {result}")]}
        elif intent == "stock":
            # Extract stock symbol from natural language
            symbol_prompt = f"""Extract the stock ticker symbol from this text. If no clear ticker is found, extract the company name and guess the most likely ticker.

Text: "{user_text}"

Respond with ONLY the ticker symbol (like AAPL, MSFT, GOOGL):"""
            symbol_response = model.invoke([HumanMessage(content=symbol_prompt)])
            symbol = symbol_response.content.strip().upper()
            result = get_stock_info.invoke({"symbol": symbol})
            return {"messages": [SystemMessage(content=f"Tool output: {result}")]}
        else:
            return {"messages": [SystemMessage(content="No tool used.")]}
    else:
        return {"messages": [SystemMessage(content="No tool used.")]}

# Node 3: llm_node
def llm_node(state: MessagesState) -> Dict[str, Any]:
    msgs = state.get("messages", [])
    
    # Check if the last message was from a tool
    if msgs and isinstance(msgs[-1], SystemMessage) and "Tool output:" in msgs[-1].content:
        tool_output = msgs[-1].content.replace("Tool output: ", "")
        return {"messages": [AIMessage(content=f"I processed your request and got: {tool_output}")]}
    else:
        return {"messages": [AIMessage(content="I can help you with text processing (reversing, word counting, uppercase conversion) and stock information. Just ask me naturally! For example: 'reverse hello world', 'count words in this sentence', 'make this uppercase', or 'what's Apple's stock price?'")]}

# Graph setup
graph = StateGraph(MessagesState)
graph.add_node("decision_node", decision_node)
graph.add_node("tool_node", tool_node)
graph.add_node("llm_node", llm_node)
graph.add_edge(START, "decision_node")
graph.add_edge("decision_node", "tool_node")
graph.add_edge("tool_node", "llm_node")
graph.add_edge("llm_node", END)
graph = graph.compile()

# Interactive loop
if __name__ == "__main__":
    print("My LangGraph agent ready. Type 'exit' to quit.")
    print("I understand natural language! Try things like:")
    print("  'reverse hello world'")
    print("  'count words in this sentence'")
    print("  'make this text UPPERCASE'")
    print("  'what is Apple stock price?'")
    print("  'how are you today?'\n")

    conversation = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break

        conversation.append(HumanMessage(content=user_input))

        result = graph.invoke({"messages": conversation})
        msgs = result.get("messages", [])

        ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
        if ai_msgs:
            reply = ai_msgs[-1].content
            print("AI:", reply)
            conversation.append(ai_msgs[-1])
        else:
            print("Result:", msgs)