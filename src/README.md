# Text Processing Agent

A LangGraph-powered AI agent that performs text processing operations using structured tools and reasoning.

## Quick Start

Run the agent:
```bash
python src/my_agent_template.py
```

## How to Use

The agent understands natural language! No need for special command formats.

### Natural Language Examples

- **Text Reversal**
  ```
  You: reverse hello world
  AI: I processed your request and got: dlrow olleh
  ```

- **Word Counting**
  ```
  You: count words in this sentence
  AI: I processed your request and got: Word count: 5
  ```

- **Uppercase Conversion**
  ```
  You: make this text UPPERCASE
  AI: I processed your request and got: MAKE THIS TEXT UPPERCASE
  ```

- **Stock Information**
  ```
  You: what's Apple's stock price?
  AI: I processed your request and got: Apple Inc. (AAPL): $150.25, Market Cap: $2,400,000,000,000
  ```

### General Conversation

The agent can also handle casual conversation:
```
You: how are you today?
AI: I can help you with text processing (reversing, word counting, uppercase conversion) and stock information. Just ask me naturally!
```

### Exit

Type `exit` or `quit` to stop the agent.

## Architecture

The agent uses a three-node LangGraph structure:
1. **Decision Node** - LLM analyzes user input and determines intent (reverse, count, upper, stock, chat)
2. **Tool Node** - Executes appropriate tools based on classified intent
3. **LLM Node** - Formats and presents results to the user

Flow: `START → decision_node → tool_node → llm_node → END`

### Key Features
- **Natural Language Processing** - No command prefixes required
- **Intent Classification** - LLM determines what the user wants to do
- **Smart Stock Symbol Extraction** - Handles "Apple stock" → "AAPL" automatically
- **Fallback to Conversation** - Gracefully handles unrecognized requests

## Extension Ideas

### 1. Memory/Database (Easiest)
SQLite database for conversation history - save/retrieve exchanges between user and agent.

### 2. Decision Node with LLM (✅ **Implemented**)  
Intelligent intent classification handles natural language instead of command prefixes.
Current flow: `START → decision_node → tool_node → llm_node → END`

### 3. Web Frontend (Medium)
FastAPI + HTML template for browser-based interaction with real-time chat interface.

## Memory System

The agent includes a SQLite-based memory system in `src/memory.py`:

### What is Stored
- **Conversations**: Every user input + agent response with timestamps and session tracking
- **Stock Queries**: Stock symbols, prices, market cap data for portfolio analytics

### How it Works
- **Persistent SQLite database** (`conversation_history.db`) survives between sessions
- **Two specialized tables** for different data types
- **Session-based isolation** for multi-user support
- **Automatic tracking** without manual intervention

### What Memory Tracks

**Every conversation exchange:**
```sql
-- conversations table stores:
timestamp: "2024-01-15T10:30:45"
user_input: "what's Apple's stock price?"
agent_response: "Apple Inc. (AAPL): $150.25, Market Cap: $2,400,000,000,000"
tool_used: "stock"
session_id: "default"
```

**Stock queries specifically:**
```sql
-- stock_queries table stores:
timestamp: "2024-01-15T10:30:45"
symbol: "AAPL"
price: 150.25
market_cap: "$2,400,000,000,000"
session_id: "default"
```

### Concrete Memory Examples

**What you could ask the memory system:**

1. **"What stocks have I searched for?"**
   ```python
   memory.get_stock_history()
   # Returns: [("2024-01-15T10:30:45", "AAPL", 150.25, "$2.4T"),
   #           ("2024-01-15T09:15:20", "MSFT", 420.80, "$3.1T")]
   ```

2. **"What are my most-watched stocks?"**
   ```python
   memory.get_frequently_queried_stocks()
   # Returns: [("AAPL", 15), ("MSFT", 8), ("GOOGL", 3)]
   # You've searched AAPL 15 times, MSFT 8 times, etc.
   ```

3. **"Show my recent conversations"**
   ```python
   memory.get_conversation_history(limit=3)
   # Returns your last 3 exchanges with timestamps
   ```

### How Memory Could Be Used (Future Enhancement)

**Price tracking across sessions:**
- "AAPL was $150.25 when you last checked yesterday, now it's $152.10 (+$1.85)"

**Portfolio insights:**
- "You've been tracking mostly tech stocks: AAPL, MSFT, GOOGL"
- "You typically check stocks on Monday mornings"

**Conversation continuity:**
- "Earlier you asked about Apple's stock, would you like an update?"

*Note: Memory system stores data automatically but contextual usage is not yet implemented in the agent flow.*