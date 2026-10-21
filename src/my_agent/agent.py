import os
import sys
from dotenv import load_dotenv


from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from my_agent.tools import web_search

load_dotenv()
SYSTEM_PROMPT = (
    "You are a helpful assistant. Use tools when they help: "
    ""
    "Be concise and say which tool you used."
)

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
    return create_agent(MODEL, tools=[web_search], system_prompt=SYSTEM_PROMPT, checkpointer=InMemorySaver())

def run_agent(user_message: str, agent=build_agent(), config={}) -> str:
    """Run the agent with a simple message."""
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": user_message}],  # only the new message
        },
        config=config,
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
            print("\nagent>", run_agent(text, agent, config={"configurable": configurable}))
            print("\n---")
            # print("History:", history)
    print("\nThread Messages:", get_thread_messages(agent, "cli-session"))  # for debugging

# __name__: a built-in variable holding the name of the current module.
if __name__ == "__main__":
    main()

