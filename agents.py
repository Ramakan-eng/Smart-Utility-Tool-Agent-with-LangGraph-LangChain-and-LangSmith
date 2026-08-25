"""
Agent and LangGraph workflow for the Smart Utility Tool Agent.

This file connects the LLM with the available tools and builds
the LangGraph tool-calling workflow.
"""

# -------------------------------------------------------
# Imports
# -------------------------------------------------------

from langchain_core.messages import SystemMessage
# Imports SystemMessage for providing instructions to the LLM.

from langgraph.graph import StateGraph, MessagesState, START
# Imports the components needed to create the LangGraph workflow.

from langgraph.prebuilt import ToolNode, tools_condition
# ToolNode executes requested tools.
# tools_condition decides whether the LLM requested a tool.

from tools import tools
# Imports the list containing all five tools from tools.py.

from llm import llm
# Imports the configured LLM from llm.py.

# -------------------------------------------------------
# Bind the tools to the LLM
# -------------------------------------------------------

llm_with_tools = llm.bind_tools(tools)
# Gives the LLM access to all five tools so it can decide whether a tool is required.


# -------------------------------------------------------
# System Prompt
# -------------------------------------------------------

SYSTEM_PROMPT = """
You are a Smart Utility Tool Agent.

You have access to five tools:

1. calculator
2. current_datetime
3. convert_units
4. text_counter
5. lookup_country_information

Rules:

- If the question requires mathematical calculation, you MUST use the calculator tool.

- If the question asks for the current date or time in the United Kingdom,
  you MUST use the current_datetime tool.

- If the question requires unit conversion,
  you MUST use the convert_units tool.

- If the question asks to count words, sentences, or characters,
  you MUST use the text_counter tool.

- If the question asks for information about a country from the predefined
  country dataset, you MUST use the lookup_country_information tool.

- For percentage questions:
  Use the calculator tool.

  Example:
  18% of 74500:
  operation = "percentage"
  a = 18
  b = 74500

  Add 12% to 50000:
  First calculate 12% of 50000, then add it to 50000.

- If the question does not require a tool,
  answer directly.

- After receiving a tool result,
  provide a clear and concise final answer.
"""
# Defines the behavioural instructions for the Smart Utility Agent.

# -------------------------------------------------------
# Agent Node
# -------------------------------------------------------

def agent_node(state: MessagesState):
    """
    Main LLM node.

    Decides whether to answer directly or request a tool.
    """

    system = SystemMessage(content=SYSTEM_PROMPT)
    # Creates a system message containing the agent instructions.

    response = llm_with_tools.invoke([system] + state["messages"])
    # Sends the system instructions and conversation messages
    # to the tool-enabled LLM.

    return {
        "messages": [response]
    }
    # Returns the LLM response to the LangGraph state.


# -------------------------------------------------------
# Tool Node
# -------------------------------------------------------

tool_node = ToolNode(tools)
# Creates a node responsible for executing tools requested by the LLM.


# -------------------------------------------------------
# Function to Build the Agent
# -------------------------------------------------------

def create_smart_utility_agent():
    """
    Build and compile the Smart Utility Tool Agent.
    """

    builder = StateGraph(MessagesState)
    # Creates a LangGraph workflow using MessagesState.

    builder.add_node(
        "agent",
        agent_node
    )
    # Adds the main LLM agent node.

    builder.add_node(
        "tools",
        tool_node
    )
    # Adds the node responsible for executing tools.

    builder.add_edge(
        START,
        "agent"
    )
    # Starts the workflow by sending the request to the agent.

    builder.add_conditional_edges(
        "agent",
        tools_condition
    )
    # Checks whether the agent requested a tool.
    #
    # If a tool is requested:
    # agent → tools
    #
    # If no tool is requested:
    # agent → END

    builder.add_edge(
        "tools",
        "agent"
    )
    # Sends the tool result back to the LLM so it can
    # generate the final answer.

    return builder.compile()
    # Compiles and returns the complete LangGraph agent.


# -------------------------------------------------------
# Create the Smart Utility Agent
# -------------------------------------------------------

smart_utility_agent = create_smart_utility_agent()
# Builds and compiles the Smart Utility Tool Agent.