from langgraph import StateGraph, MessagesState

def simple_classifier_node(inputs):
    """
    Classifies email as spam if suspicious features exceed a threshold.
    Input: Features dict
    Output: Prediction and spam score
    """
    features = inputs["features"]
    num_links = features.get("num_links", 0)
    fraction_suspicious = features.get("fraction_suspicious", 0)
    has_html = features.get("has_html", False)

    # vv basic heuristic: adding for each present spam feature
    spam_score = 0
    spam_score += 1 if has_html else 0
    spam_score += 1 if num_links >= 1 else 0
    spam_score += 1 if fraction_suspicious > 0.2 else 0

    label = "spam" if spam_score >= 2 else "not_spam"
    return {"spam_label": label, "spam_score": spam_score}

def format_prediction_node(inputs):
    """
    Formats the classifier output into a readable result string.
    Input: Spam label and score
    Output: Formatted string
    """
    label = inputs.get("spam_label", "unknown")
    score = inputs.get("spam_score", 0)

    result_str = f"Prediction: {label.upper()} (spam score: {score})"
    return {"formatted_result": result_str}

#build subgraph
classifier_subgraph = StateGraph(MessagesState)

classifier_subgraph.add_node("simple_classifier_node", simple_classifier_node)
classifier_subgraph.add_node("format_prediction_node", format_prediction_node)

classifier_subgraph.add_edge(START, "simple_classifier_node")
classifier_subgraph.add_edge("simple_classifier_node", "format_prediction_node")
classifier_subgraph.add_edge("format_prediction_node", END)

# Compile
classifier_subgraph = classifier_subgraph.compile()
