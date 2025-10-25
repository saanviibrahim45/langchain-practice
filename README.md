# LangChain Onboarding Project


## My Tool

The stock_close_tool retrieves the most recent stock price for a given ticker symbol using the yfinance API.
When a user enters a command like stock: AAPL, the tool queries Yahoo Finance for the latest available price, returning either a near real-time quote or the most recent daily close.
This computed result is passed to the LLM, which then formats and delivers the response naturally to the user.

## Purpose

This repository is for onboarding SWE interns to **LangChain** by building a simple AI agent.
Each engineer will learn how to use LangChain’s core components — models, tools, and agents — to create a runnable prototype demonstrating structured reasoning and output generation.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/saanviibrahim45/langchain-practice.git
cd langchain-practice
```

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation and API Key

Check that all key libraries are installed and importable:

```bash
pip show langchain langgraph langchain-openai python-dotenv
```

Then confirm imports work and your environment is set up correctly:

```bash
python - <<'PY'
try:
    import langgraph, langchain, langchain_openai, dotenv
    print("Imports OK")
except Exception as e:
    print("Import error:", e)
PY
```

You should see:
`Imports OK`

Next, verify your OpenAI key is loaded:

```bash
python - <<'PY'
import os
print("OPENAI_API_KEY present:", bool(os.environ.get("OPENAI_API_KEY")))
PY
```

You should see:
`OPENAI_API_KEY present: True`

---

### 5. Test Model Connection

Confirm that your OpenAI model can respond:

```bash
python - <<'PY'
from langchain_openai import ChatOpenAI
m = ChatOpenAI(model="gpt-4o-mini", temperature=0)
resp = m.invoke([{"role":"user","content":"Say hello in one short sentence."}])
print(resp.content)
PY
```

Expected output:
`Hello there! How can I help you today?`

---

### 6. Run the Example Agent

From the project root, run:

```bash
python src/base_agent.py
```

You should see:

```
LangGraph base agent ready. Type 'exit' to quit.
Use 'calc: 2+2' to try the calculator tool.
```

**Example Run**

```bash
You: calc: 5+5
AI: The result is 10.
You: What did I just calculate?
AI: You just calculated 5 plus 5, which equals 10.
```

---

## Deliverable Expectations

Each SWE must:

1. Work in their own branch (named after them) and add a new file in `src/` to build their agent.
2. Implement one **working LangChain agent** with structured output (JSON or clearly formatted text).
3. Write a short `README.md` in their folder explaining what their agent does and how it works.
4. Ensure their `.env` file is configured so their code runs locally end-to-end.

When finished, submit a pull request or share your branch for review.

---

## Notes

* Keep your API keys private — **never commit `.env` files**.
* If you add new dependencies, update `requirements.txt`.
* Write clean, modular, and well-commented code.
* Test your agent manually using interactive input.

---

### Example Command Summary

```bash
git clone https://github.com/saanviibrahim45/langchain-practice.git
cd langchain-practice
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your_key_here"
python src/base_agent.py
```

---

## Technical Walkthrough: Base LangGraph Agent

The included `src/base_agent.py` demonstrates how to build a minimal **LangGraph-powered agent** using LangChain’s OpenAI integration. It models an LLM-driven reasoning flow using modular nodes that communicate through a shared state.

### Core Concepts

**1. StateGraph and MessagesState**

* `StateGraph` defines a **directed workflow** of agent steps.
* Each node processes and returns a `MessagesState` (a structured list of `HumanMessage`, `AIMessage`, and `SystemMessage` objects).
* This enables memory and context passing across nodes.

**2. Nodes and Edges**

* Nodes are Python functions that take in state and return updates.
* Edges define the **flow** of execution (`START → tool_node → llm_node → END`).

| Node        | Purpose                                                              |
| ----------- | -------------------------------------------------------------------- |
| `tool_node` | Detects calculator commands like `calc: 2+2` and appends tool output |
| `llm_node`  | Reads recent messages and produces an AI response via OpenAI         |

**3. Tools**

* A “tool” is a callable function the agent can invoke when needed.
* The calculator function (`calculator_tool`) is a simple example — future tools can include:

  * Web search (e.g. Tavily, Serper)
  * Data analysis
  * File summarization

**4. Execution Loop**

* The interactive loop lets you type messages directly into the terminal.
* Each message is wrapped as a `HumanMessage` and passed through the compiled graph.
* The agent then decides how to respond based on the state.

---

**Example Run**

```bash
You: calc: 10*(5+2)
AI: The result is 70.
You: What did I just calculate?
AI: You just calculated 10 times (5 plus 2), which equals 70.
```
