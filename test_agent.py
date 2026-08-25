# -------------------------------------------------------
# Terminal Testing for Smart Utility Tool Agent
# -------------------------------------------------------

from langchain_core.messages import HumanMessage

from agents import smart_utility_agent


def ask_agent(question):
    """
    Send a question to the Smart Utility Agent
    and print the final answer.
    """

    print("\n" + "=" * 60)

    print(f"User: {question}")

    print("-" * 60)

    result = smart_utility_agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content=question
                )
            ]
        }
    )

    final_answer = result["messages"][-1].content

    print(f"Agent: {final_answer}")

    print("=" * 60)


# -------------------------------------------------------
# Test Queries
# -------------------------------------------------------

ask_agent(
    "Calculate 18% VAT on £74,500"
)

ask_agent(
    "Convert 15 kilometers into miles"
)

ask_agent(
    "How many words and characters are in this sentence?"
)

ask_agent(
    "What is the capital and currency of India?"
)

ask_agent(
    "What is the current date and time in the United Kingdom?"
)

ask_agent(
    "What is the capital of Germany?"
)

ask_agent(
    "Hello, how are you?"
)