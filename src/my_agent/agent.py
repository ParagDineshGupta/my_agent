import os
import sys

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import ToolCallLimitMiddleware, SummarizationMiddleware

from my_agent.tools import web_search

load_dotenv()
SYSTEM_PROMPT = (
    "You are a helpful assistant. Use tools when they help: "
    ""
    "Be concise and say which tool you used."
)


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
        tools=[web_search],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
        middleware=[
            ToolCallLimitMiddleware(thread_limit=30, run_limit=2),
            SummarizationMiddleware(
                model="gpt-5.4-mini",
                trigger=("tokens", 100),
                keep=("messages", 2),
            ),
        ],
    )


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
