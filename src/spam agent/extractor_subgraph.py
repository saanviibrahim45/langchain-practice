import re
from langgraph import StateGraph, MessagesState
import nltk, string, ssl
from nltk.corpus import words # import gives access to english dictionary words 

# SSL verification temporarily, o/w cannot load dictionary (safe for local testing but not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context 
nltk.download('words')
# load word list
english_words = set(words.words()) 

# Tool function
def spam_word_count_tool(body_text):
    """Output number of mispelled (non-existent) words and length of text"""
    wrong_count = 0
    # split sentence into array of words by spacing
    word_arr = body_text.split()
    for w in word_arr:
        # remove punctuation
        w = w.strip(string.punctuation)
        if w.lower() not in english_words:
            wrong_count += 1
    return wrong_count, len(word_arr)

# The following 3 nodes can run in parallel. 
def count_links_node(inputs):
    """
    Counts the number of links in the email HTML body.
    Input: Email dictionary
    Output: Number of links
    """
    links = inputs["email_dict"].get("links", [])
    return {"num_links": len(links)}

def fraction_suspicious_words_node(inputs):
    """
    Counts fraction of suspicious words in the email text that often appear in spam.
    Input: Email dictionary
    Output: Number of suspicious words
    """
    body_text = inputs["email_dict"].get("body_text", "").lower()
    suspicious_words = ["win", "prize", "free", "urgent", "cash", "reward"] # hardcoded example
    count = sum(body_text.count(word) for word in suspicious_words) 
    mispelled, total_len = spam_word_count_tool(body_text)
    count += mispelled
    return {"fraction_suspicious": count / total_len}

def has_html_node(inputs):
    """
    Determines if the email contains an HTML body.
    Input: Email dict
    Output: True if HTML body exists
    """
    html_body = inputs["email_dict"].get("html_body", "")
    return {"has_html: ": bool(html_body.strip())}

# awaits previous 3 nodes' completion
def combine_features_node(inputs):
    """
    Combines all extracted features into a single dictionary for classification.
    Input: Computation of previous 3 nodes
    Output: Feature dictionary
    """
    features = {
        "num_links": inputs.get("num_links", 0),
        "fraction_suspicious": inputs.get("fraction_suspicious", 0),
        "has_html": inputs.get("has_html", False),
    }
    return {"features": features}

#build subgraph
extractor_subgraph = StateGraph(MessagesState)

extractor_subgraph.add_node("count_links_node", count_links_node)
extractor_subgraph.add_node("fraction_suspicious_words_node", fraction_suspicious_words_node)
extractor_subgraph.add_node("has_html_node", has_html_node)
extractor_subgraph.add_node("combine_features_node", combine_features_node)

# Parallel edges from START to first three nodes
extractor_subgraph.add_edge(START, "count_links_node")
extractor_subgraph.add_edge(START, "fraction_suspicious_words_node")
extractor_subgraph.add_edge(START, "has_html_node")

# Edges from the three nodes to combine_features_node
extractor_subgraph.add_edge("count_links_node", "combine_features_node")
extractor_subgraph.add_edge("fraction_suspicious_words_node", "combine_features_node")
extractor_subgraph.add_edge("has_html_node", "combine_features_node")

# Edge from combine_features_node to END
extractor_subgraph.add_edge("combine_features_node", END)

extractor_subgraph = extractor_subgraph.compile()
