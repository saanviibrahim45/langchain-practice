# Simple Spam Checker Agent

## Overview
This project implements a basic conversational agent using **LangGraph** and **LangChain**, enhanced with a simple **spam detection tool**. The agent can process user messages and either:

1. Check for spam-like content in a sentence, based on the fraction of misspelled words.
2. Respond naturally using a GPT-4-based LLM.

## Features

### Spam Detection Tool
- Triggered by messages starting with:  'check spam: '
- Computes the fraction of words in the sentence that are not found in an English dictionary (using NLTK's word corpus).
- Returns a concise system message indicating the spam score.

### LLM Response
- Handles all other messages not starting with `check spam:`.
- Uses the last 5 messages as context to generate a concise AI reply.
- Powered by `gpt-4o-mini` through LangChain’s `ChatOpenAI`.

## Architecture
- **Nodes:**  
- `tool_node`: Runs the spam detection tool when applicable.  
- `llm_node`: Generates AI responses for all other messages.  
- **Graph:** A `StateGraph` orchestrates the conversation flow:

## Running the agent
python src/spam_agent.py