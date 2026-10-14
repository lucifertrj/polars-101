import streamlit as st
from haystack.dataclasses import ChatMessage

from app import create_agent  
from nav import render_nav
from tool import get_csv_context

from dotenv import load_dotenv
load_dotenv()

@st.cache_resource(show_spinner=False)
def get_agent():
    """Create and cache the agent (built from app.py's config)."""
    return create_agent()


def extract_trace(messages) -> list[dict]:
    """Flatten a Haystack run's messages into ordered tool-call records.

    Each record pairs a ToolCall (tool name + arguments, e.g. the generated
    Polars code) with its ToolCallResult (printed output or error), matched by
    the call id so calls and their results line up across messages.
    """
    calls: dict[str, dict] = {}
    order: list[str] = []

    for msg in messages:
        for call in msg.tool_calls or []:
            cid = call.id or f"{call.tool_name}-{len(order)}"
            calls[cid] = {
                "tool": call.tool_name,
                "arguments": dict(call.arguments or {}),
                "result": None,
                "error": False,
            }
            order.append(cid)
        for res in msg.tool_call_results or []:
            cid = res.origin.id if res.origin else None
            entry = calls.get(cid)
            if entry is None:  # result with no matching call we tracked
                cid = f"orphan-{len(order)}"
                entry = {
                    "tool": res.origin.tool_name if res.origin else "tool",
                    "arguments": {},
                    "result": None,
                    "error": False,
                }
                calls[cid] = entry
                order.append(cid)
            entry["result"] = res.result
            entry["error"] = bool(res.error)

    return [calls[c] for c in order]


def render_trace(trace: list[dict]) -> None:
    """Render tool calls as an expandable trace under an assistant message."""
    if not trace:
        return
    summary = ", ".join(f"{t}×{sum(1 for s in trace if s['tool'] == t)}"
                        for t in dict.fromkeys(s["tool"] for s in trace))
    with st.expander(f"🔍 Agent trace — {len(trace)} tool call(s) · {summary}"):
        for i, step in enumerate(trace, 1):
            st.markdown(f"**{i}. `{step['tool']}`**")
            code = step["arguments"].get("code")
            if code:
                st.code(code, language="python")
                extra = {k: v for k, v in step["arguments"].items() if k != "code"}
                if extra:
                    st.json(extra)
            elif step["arguments"]:
                st.json(step["arguments"])
            if step["result"] is not None:
                st.caption("Error output" if step["error"] else "Output")
                st.code(str(step["result"]), language="text")


def main():
    st.set_page_config(
        page_title="Sales Data Analytics Agent",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Sales Data Analytics Agent")
    st.caption("Ask questions about your sales data in natural language — Polars runs the numbers.")

    # Sidebar: navigation + data preview + example questions
    render_nav()
    with st.sidebar:
        st.header("📁 Data Preview")
        if st.button("Show Data Context"):
            with st.spinner("Loading data context..."):
                st.code(get_csv_context(), language="text")

        st.divider()
        st.header("💡 Example Questions")
        example_questions = [
            "What is the total revenue by category?",
            "Which sales rep has the highest sales?",
            "Show me monthly revenue trends",
            "What are the top 5 products by quantity sold?",
            "Compare revenue across regions",
            "What is the average order value by category?",
        ]
        for q in example_questions:
            if st.button(q, key=q):
                st.session_state.question = q

        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay history, including persisted traces
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_trace(message.get("trace", []))

    question = st.chat_input("Ask a question about the sales data...")

    # Example-question buttons feed the same input path
    if "question" in st.session_state:
        question = st.session_state.question
        del st.session_state.question

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                trace: list[dict] = []
                try:
                    agent = get_agent()
                    result = agent.run(messages=[ChatMessage.from_user(question)])
                    answer = result["last_message"].text
                    trace = extract_trace(result["messages"])
                except Exception as e:
                    answer = f"Try again later..."

            st.markdown(answer)
            render_trace(trace)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "trace": trace}
            )


if __name__ == "__main__":
    main()
