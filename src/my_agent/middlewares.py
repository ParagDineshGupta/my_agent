from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

# A wrap_tool_call hook receives the tool request and a handler that actually runs the tool
@wrap_tool_call
def handle_tool_errors(request, handler):
    try:
        print(f"Handling tool call for {request.tool_call['name']} with input: {request.tool_call['input']}")
        return handler(request)                      # run the tool
    except Exception as e:
        print("error type:", type(e).__name__, "| error:", repr(e), flush=True)
        print("tool call args:", request.tool_call["args"], flush=True)
        traceback.print_exc()          # shows the exact line that raised
        return ToolMessage(
            content=f"Tool failed: {e}. Check the input or try another approach.",
            tool_call_id=request.tool_call["id"],    # required
        )