from typing import Dict, Any
from tavily import TavilyClient
from langchain.tools import tool, ToolRuntime
from dotenv import load_dotenv
import readline  # enables arrow keys, history, and line editing
from langgraph.types import Command
from langchain.messages import ToolMessage

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
    answer: str,
    runtime: ToolRuntime,
) -> str:
    """Save text to a local file. Use this whenever the user says
    'save', 'write to file', or 'export'. Pass the PREVIOUS assistant
    answer in `answer`. Do not search the web for this."""

    filename = "answer.txt"

    with open(filename, "w") as f:
        f.write(f"Question: {runtime.context.user_question}\nAnswer: {answer}")
    return f"Answer saved to {filename}"


@tool
def read_previous_question(runtime: ToolRuntime) -> str:
    """Read the previous question from the state."""
    try:
        print("Reading previous question from state...", runtime.state)
        return runtime.state["previous_question"]
    except KeyError:
        return "No previous question found in state"

# rely on the model to call a tool
@tool
def update_previous_question(question: str, runtime: ToolRuntime) -> Command:
    """Save the user's question as the previous question."""
    return Command(                  # 2) return a Command
        update={
            "previous_question": question,
            "messages": [            # 3) close the tool call
                ToolMessage(
                    content=f"Previous question updated to: {question}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
