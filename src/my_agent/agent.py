import os
import sys

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import (
    ToolCallLimitMiddleware,
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
)

from my_agent.tools import web_search, save_answer_to_file
from dataclasses import dataclass


load_dotenv()
SYSTEM_PROMPT = (
    "You are a helpful assistant. Use tools when they help: "
    ""
    "Be concise and say which tool you used."
    "Save final answer to a file using the save_answer_to_file tool."
    "Answer from your own knowledge when you can. "
    "Use web_search only for recent or unknown information. "
    "If the user asks to save an answer, call save_answer_to_file "
    "with your previous answer. Never search for that request."
)


  # global values are shared across all runs. Context is isolated per run.
@dataclass
class Context:
    """Context schema for the agent's state."""
    user_question: str = ""

class ToolCallLogger(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "tool")
        print(f"\n[tool call] {name}: {input_str}")

    def on_tool_end(self, output, **kwargs):
        print(f"[tool result] {output}")

    def on_tool_error(self, error, **kwargs):
        print(f"[tool error] {error}")


def build_agent():
    """Create the agent: a model + tools + instructions."""
    from langchain.agents import create_agent  # imported here so tests need no SDK
    # from .tools_lc import TOOLS

    MODEL_NAME = os.getenv("AGENT_MODEL", "openai:gpt-4o-mini")  # "provider:model-name"
    MODEL = init_chat_model(
        MODEL_NAME,
        temperature=0,
        max_tokens=500,
    )
    return create_agent(
        MODEL,
        tools=[web_search, save_answer_to_file],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
        middleware=[
            ToolCallLimitMiddleware(thread_limit=30, run_limit=2),
            SummarizationMiddleware(
                model="gpt-5.4-mini",
                trigger=("tokens", 100),
                keep=("messages", 2),
            ),
            HumanInTheLoopMiddleware(  # state saved by the checkpointer
                interrupt_on={
                    "web_search": False,
                    "save_answer_to_file": True,
                },
                description_prefix="Tool execution requires approval",
            ),
        ],
        context_schema=Context,          # declare the shape
        # context_validator=ContextValidator()  # validate the shape
    )  # type: ignore


def run_agent(
    user_message: str, agent=None, config: RunnableConfig | None = None
) -> str:
    """Run the agent with a simple message."""
    if agent is None:
        agent = build_agent()

    run_config: RunnableConfig = {**(config or {})}
    callbacks = run_config.get("callbacks")
    if callbacks is None:
        run_config["callbacks"] = [ToolCallLogger()]
    elif isinstance(callbacks, list):
        run_config["callbacks"] = [*callbacks, ToolCallLogger()]
    else:
        callbacks.add_handler(ToolCallLogger())

    result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": user_message}
            ],  # only the new message
        },
        config=run_config,
    )

    # Keep resuming until the agent is no longer paused
    while "__interrupt__" in result:
        print(result["__interrupt__"])
        print(
            "Do you approve the tool execution / save this answer? (type 'approve' or 'reject')"
        )

        choice = input("approve / reject: ").strip().lower()

        if choice == "approve":
            decision = {"type": "approve"}
        else:
            decision = {"type": "reject", "message": "User rejected saving."}

        result = agent.invoke(
            Command(resume={"decisions": [decision]}),
            config=config,  # same thread_id
            context={"user_question": user_message},
        )

    return result["messages"][-1].content


def get_thread_messages(agent, thread_id: str) -> list:
    state = agent.get_state({"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY first (see README).")
    print("Agent ready. Type 'quit' to exit.")
    agent = build_agent()
    # history: list = []
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit"}:
            break
        if text:
            configurable = {"thread_id": "cli-session"}
            print(
                "\nagent>",
                run_agent(text, agent, config={"configurable": configurable}),
            )
            print("\n---")
            # print("History:", history)
    print(
        "\nThread Messages:", get_thread_messages(agent, "cli-session")
    )  # for debugging


# __name__: a built-in variable holding the name of the current module.
if __name__ == "__main__":
    main()
