import re
from langgraph import StateGraph, MessagesState
from langchain.schema import HumanMessage
from email import policy
from email.parser import BytesParser


def read_file_node(inputs):
    """
    Read email file and parse into EmailMessage object.
    Input: file path (main agent has to load in file path)
    Output: EmailMessage object
    """
    file_path = inputs.get("file_path")
    with open(file_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    return {"msg": msg}

def extract_text_html_node(inputs):
    """
    Extract plain text and HTML body from EmailMessage.
    Input: EmailMessage object
    Output: Tuple of plain text and html body
    """
    msg = inputs["msg"]
    body_text = ""
    html_body = ""
    if msg.is_multipart(): # split into different segments of text & html
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                body_text += part.get_content()
            elif content_type == "text/html":
                html_body += part.get_content()
    else:
        content_type = msg.get_content_type()
        if content_type == "text/plain":
            body_text = msg.get_content()
        elif content_type == "text/html":
            html_body = msg.get_content()
    return {"body_text": body_text, "html_body": html_body}


def extract_links_node(inputs):
    """
    Extract all components of an email.
    Input: Separated body text and HTML
    Output: Components of an email (e.g. subject, date, links)
    """
    html_body = inputs["html_body"]
    links = re.findall(r'https?://[^\s<>"]+', html_body)
    email_dict = {
        "subject": inputs["msg"].get("subject"),
        "from": inputs["msg"].get("from"),
        "to": inputs["msg"].get("to"),
        "date": inputs["msg"].get("date"),
        "body_text": inputs.get("body_text", "").strip(),
        "html_body": html_body.strip(),
        "links": links
    }
    return {"email_dict": email_dict}


#build subgraph
loader_subgraph = StateGraph(MessagesState)

loader_subgraph.add_node("read_file_node", read_file_node)
loader_subgraph.add_node("extract_text_html_node", extract_text_html_node)
loader_subgraph.add_node("extract_links_node", extract_links_node)

loader_subgraph.add_edge(START, "read_file_node")
loader_subgraph.add_edge("read_file_node", "extract_text_html_node")
loader_subgraph.add_edge("extract_text_html_node", "extract_links_node")
loader_subgraph.add_edge("extract_links_node", END)

loader_subgraph = loader_subgraph.compile()





