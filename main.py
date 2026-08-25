"""
Streamlit user interface for the Smart Utility Tool Agent.

This application allows the user to ask one question at a time
and receive answers from the LangGraph-powered utility agent.
"""

# -------------------------------------------------------
# Imports
# -------------------------------------------------------

import streamlit as st
# Imports Streamlit for building the web application.

from langchain_core.messages import HumanMessage
# Imports HumanMessage to represent the user's question.

from agents import smart_utility_agent
# Imports the compiled LangGraph Smart Utility Agent.


# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Smart Utility Agent",
    page_icon="🤖",
    layout="wide"
)
# Configures the browser tab title, icon, and page layout.


# -------------------------------------------------------
# Custom CSS
# -------------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0;
    }

    .subtitle {
        text-align: center;
        color: gray;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .tool-card {
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)
# Adds some custom styling to make the interface cleaner.


# -------------------------------------------------------
# Initialize Chat History
# -------------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []

# Creates an empty chat history when the application
# is opened for the first time.


# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

with st.sidebar:

    st.title("🤖 Smart Utility Agent")

    st.markdown(
        """
        ### Available Tools

        🧮 **Calculator**

        📅 **UK Date & Time**

        📏 **Unit Converter**

        📝 **Text Counter**

        🌍 **Country Lookup**
        """
    )

    st.divider()

    st.markdown("### Example Questions")

    st.markdown(
        """
        - Calculate 18% VAT on £74,500

        - Convert 15 kilometers into miles

        - What is the current date and time in the UK?

        - Count the words in this paragraph

        - What is the capital of Japan?
        """
    )

    st.divider()

    # Clear chat button.
    if st.button("🗑️ Clear Conversation"):

        st.session_state.messages = []

        st.rerun()


# -------------------------------------------------------
# Main Application Header
# -------------------------------------------------------

st.markdown(
    '<div class="main-title">🤖 Smart Utility Tool Agent</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Your personal AI assistant powered by LangGraph, LangChain and OpenAI
    </div>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------------
# Welcome Message
# -------------------------------------------------------

if not st.session_state.messages:

    with st.chat_message("assistant"):

        st.markdown(
            """
            👋 **Hello! I'm your Smart Utility Tool Agent.**

            I can help you with:

            🧮 Calculations  
            📅 Current UK date and time  
            📏 Unit conversions  
            📝 Word, sentence and character counting  
            🌍 Country information  

            **Ask me one question at a time!**
            """
        )


# -------------------------------------------------------
# Display Previous Conversation
# -------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# -------------------------------------------------------
# User Input
# -------------------------------------------------------

prompt = st.chat_input(
    "Ask your Smart Utility Agent a question..."
)
# Displays a chat input box at the bottom of the page.


# -------------------------------------------------------
# Process User Question
# -------------------------------------------------------

if prompt:

    # Display the user's question.
    with st.chat_message("user"):

        st.markdown(prompt)

    # Save the user's question in the chat history.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )


    # ---------------------------------------------------
    # Call the LangGraph Agent
    # ---------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("🤔 Thinking..."):

            try:

                result = smart_utility_agent.invoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=prompt
                            )
                        ]
                    }
                )

                # Gets the final AI message from the graph.
                answer = result["messages"][-1].content


                # Display the final answer.
                st.markdown(answer)


                # Save the AI response in chat history.
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )


            except Exception as error:

                error_message = (
                    f"❌ An error occurred: {str(error)}"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message
                    }
                )