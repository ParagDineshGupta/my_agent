from typing import Dict, Any
from tavily import TavilyClient
from langchain.tools import tool
from dotenv import load_dotenv
import readline  # enables arrow keys, history, and line editing

load_dotenv()

tavily_client = TavilyClient()


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web ONLY for current or unknown facts.
    Do NOT use this when the user asks to save, write, or export something. We want to use it only for searching the web for information when really needed to minimize the number of web searches. The input should be a search query string."""

    return tavily_client.search(query)


web_search.description = (
    "Search the web for information. Input should be a search query string."
)


@tool
def save_answer_to_file(
    answer: str, user_question: str, filename: str = "answer.txt"
) -> str:
    """Save text to a local file. Use this whenever the user says
    'save', 'write to file', or 'export'. Pass the PREVIOUS assistant
    answer in `answer`. Do not search the web for this."""

    with open(filename, "w") as f:
        f.write(f"Question: {user_question}\nAnswer: {answer}")
    return f"Answer saved to {filename}"
