# src/spam_agent.py
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os
from typing import Dict, Any
import nltk, string, ssl
from nltk.corpus import words # import gives access to english dictionary words 

load_dotenv()

# SSL verification temporarily, o/w cannot load dictionary (safe for local testing but not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context 
nltk.download('words')
# load word list
english_words = set(words.words()) 

# Tool function: Calculate fraction of sentence comprised of mispelled (non-existent) words 
def spam_word_count_tool(sentence : str) -> str:
    wrong_count = 0
    # split sentence into array of words by spacing
    word_arr = sentence.split(" ")
    if len(word_arr) == 0:
        return f"Empty sentence"
    else:
        for w in word_arr:
            # remove punctuation
            w = w.strip(string.punctuation)
            if w.lower() not in english_words:
                wrong_count += 1
        return f"Fraction of sentence mispelled: {wrong_count / len(word_arr)}"

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
graph.add_node("tool_node", tool_node)
graph.add_node("llm_node", llm_node)
graph.add_edge(START, "tool_node")
graph.add_edge("tool_node", "llm_node")
graph.add_edge("llm_node", END)
graph = graph.compile()


# Iteractive loop
if __name__ == "__main__":
    print("LangGraph base agent ready. Type 'exit' to quit.")
    print("Use 'check spam: <your sentence of choice>' to try the spam checking tool. Anything else will be handled by the LLM.\n")

    conversation = []  # keep all messages here

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break

        conversation.append(HumanMessage(content = user_input)) # store user input

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