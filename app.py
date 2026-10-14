from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIResponsesChatGenerator
from haystack.dataclasses import ChatMessage
from tool import create_polars_tools

from dotenv import load_dotenv
load_dotenv()

SYSTEM_PROMPT = """You are a data analytics agent for a sales dataset. Answer with numbers you computed, never estimates.

Tool:
- execute_polars_code: runs Python with `pl` (Polars 2.0) and `csv_path` predefined. Only printed output comes back.

How to write the code:
1. Start lazy: `lf = pl.scan_csv(csv_path, try_parse_dates=True)`. Never use read_csv or pandas.
2. On your first call, `print(lf.collect_schema())` to learn the column names, dtypes, and confirm date parsing before doing anything else. This reads no data.
3. Select only the columns the question needs, and filter early.
4. Before collecting the answer, run `print(q.collect_schema())` on the final query to check column names and types without reading the data. Fix any error there first.
5. Then `print(q.collect())`. Sort explicitly when order matters: group_by and join output order is not guaranteed.
6. Keep output small: aggregate, or use .head(20). Never print the full table.

Polars 2.0 rules:
- Parse strings with .str.to_date() or .str.to_datetime(), not .cast(pl.Date).
- is_in() needs matching types. Cast first.
- Use .unpivot(), not .melt(). Use nulls_equal=, not join_nulls=.

If the code fails, read the error, fix it, and retry. Stop after 3 attempts and say what blocked you.

Answer format: the direct answer first, with the key numbers. Then at most 2 short observations. Mention nulls or filters that changed the result.
"""

def create_agent() -> Agent:
    """Create and configure the data analytics agent."""
    agent = Agent(
        chat_generator=OpenAIResponsesChatGenerator(model="gpt-5.6-luna"),
        tools=create_polars_tools(),
        system_prompt=SYSTEM_PROMPT,
    )
    agent.warm_up()
    return agent


def run_query(agent: Agent, question: str) -> str:
    """
    Run a query against the agent and return the response.

    Args:
        agent: The configured Haystack agent
        question: User's question about the data

    Returns:
        Agent's response as a string
    """
    result = agent.run(messages=[ChatMessage.from_user(question)])
    return result["last_message"].text


def interactive_mode():
    """Run the agent in interactive mode."""
    agent = create_agent()

    while True:
        try:
            question = input("\n📊 Your question: ").strip()

            if not question:
                continue

            if question.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            print("\nAnalyzing...\n")
            response = run_query(agent, question)
            print("\nAnswer:")
            print("-" * 40)
            print(response)
            print("-" * 40)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\Try again later...")


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        agent = create_agent()
        response = run_query(agent, question)
        print(response)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
