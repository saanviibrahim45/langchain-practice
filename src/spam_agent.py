# src/spam_agent.py
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import sys
import os
from typing import Dict, Any
from loader_subagent import LoaderSubAgent

load_dotenv()

# initialize LLM
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Node 1: tool_node
def tool_node(state: MessagesState) -> Dict[str, Any]:
    last_msg = state["messages"][-1]
    # access last user input
    user_input = getattr(last_msg, "content", "")
    if isinstance(last_msg, HumanMessage) and user_input.lower().startswith("check spam:"):
        user_sentence = user_input.split(":", 1)[1].strip()
        tool_output = spam_word_count_tool(user_sentence)
        return {"messages": [SystemMessage(content=f"Spam Tool output: {tool_output}")]}
    else:
        return {"messages": [SystemMessage(content="No tool used.")]}

# Node 2: llm_node
def llm_node(state: MessagesState) -> Dict[str, Any]:
    # get current conversation history
    msgs = state.get("messages", [])
    # collect last 5 messages from any type
    context = "\n".join(
        [f"{m.type}: {getattr(m, 'content', '')}" for m in msgs[-5:]]
    )
    prompt = f"Respond concisely to the conversation:\n{context}"
    # invokes model on last message with human prompt
    resp = model.invoke([HumanMessage(content=prompt)])
    try:
        text = resp.content
    except Exception:
        text = str(resp)
    return {"messages": [AIMessage(content=text)]}

# Graph setup
graph = StateGraph(MessagesState)

graph.add_node("loader_subagent", loader_subagent)
graph.add_node("extractor_subagent", extractor_subagent)
graph.add_node("classifier_subagent", classifier_subagent)

graph.add_edge(START, "loader_subagent")
graph.add_edge("loader_subagent", "extractor_subagent")
graph.add_edge("extractor_subagent", "classifier_subagent")
graph.add_edge("classifier_subagent", END)

graph = graph.compile()

# Iteractive loop
if __name__ == "__main__":
    print("LangGraph base agent ready. Type 'exit' to quit.")
    print("Use 'check spam: <your file path>' to try the spam function. Anything else will be handled by the LLM.\n")

    conversation = []  # keep all messages here

    while True:
        while True:
            #user inputs a file (email)
            user_input = input("You: ").strip()
            file_path = input("Path to your email file: ")
            if user_input.lower() in ("exit", "quit"):
                sys.exit(0)
            try:
                #open file for reading
                with open(file_path, 'r') as f:
                    conversation.append(HumanMessage(content = file_path))
                    print("File path: ", file_path)
                    break
            except FileNotFoundError:
                print("Error: File not found. Please try again.")

        # pass full message history into the graph
        result = graph.invoke({"messages": conversation})
        msgs = result.get("messages", [])

        # find and print AI reply
        ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
        if ai_msgs:
            reply = ai_msgs[-1].content
            print("AI:", reply)
            conversation.append(ai_msgs[-1]) # update conversation with latest AI reply
        else:
            print("Result: ", msgs)