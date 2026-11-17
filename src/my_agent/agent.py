import os
import sys

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langchain.messages import SystemMessage, ToolMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import (
    ToolCallLimitMiddleware,
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
)
from my_agent.middlewares import handle_tool_errors


from my_agent.tools import web_search, save_answer_to_file, read_previous_question, update_previous_question, broken_tool
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
    "If the user asks to send a message, call send_message."
    "If the user asks to read the previous question, call read_previous_question."
    "If the user asks to update the previous question, call update_previous_question."
    "If a tool fails, return the error message to the user use the handle_tool_errors middleware to catch tool errors and return a message instead of crashing."
    "If the user explicitly asks you to call a named tool, call it. Never claim a tool was called unless you received its result."
)

system_message = SystemMessage(content=SYSTEM_PROMPT)
  # global values are shared across all runs. Context is isolated per run and immutable.
@dataclass
class Context:
    """Context schema for the agent's state."""
    user_question: str = ""

from langchain.agents import AgentState

class CustomState(AgentState): # cant set default value
    current_question: str = ""
    previous_question: str = ""
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
    # common interface for all models, but some models have extra features, use specific Model SDK like OpenAIChatModel for OpenAI models to access those features.
    model = init_chat_model(
        MODEL_NAME,
        temperature=0.1, # deterministic, decides randomness of the model's output. 0 is deterministic, 1 is random. 0 to 1.0/2.0
        max_tokens=200, # Maximum number of output tokens
    )
    return create_agent(
        model,
        tools=[web_search, save_answer_to_file, read_previous_question, update_previous_question,broken_tool],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
        middleware=[
            handle_tool_errors,  # catch tool errors and return a message instead of crashing
            ToolCallLimitMiddleware(thread_limit=30, run_limit=2),
            SummarizationMiddleware(
                model="gpt-5.4-mini",
                trigger=("tokens", 100),
                keep=("messages", 2),
            ),
            HumanInTheLoopMiddleware(  # state saved by the checkpointer
                interrupt_on={
                    "web_search": False,
                    "save_answer_to_file": True                },
                description_prefix="Tool execution requires approval",
            ),
        ],
        context_schema=Context,
        state_schema=CustomState,        # declare the shape
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
            "messages": [ # structured input for Multi-turn conversations.
                {"role": "user", "content": user_message}
            ],  # only the new message
            # "messages": [UserMessage(user_message)],
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

    total_in = total_out = 0

    for msg in result["messages"]:
        msg.pretty_print()
        if msg.type == "ai" and msg.usage_metadata:
            u = msg.usage_metadata
            print(f"input={u['input_tokens']}, output={u['output_tokens']}, total={u['total_tokens']}")
            total_in += u["input_tokens"]
            total_out += u["output_tokens"]

    print(f"\nBilled usage -> input: {total_in}, output: {total_out}, total: {total_in + total_out}")

    return result["messages"][-1].content


def get_thread_messages(agent, thread_id: str) -> list:
    state = agent.get_state({"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY first (see README).")
    print("Agent ready. Type 'quit' to exit.")
    agent = build_agent()
    print(agent.get_graph().draw_mermaid())
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
