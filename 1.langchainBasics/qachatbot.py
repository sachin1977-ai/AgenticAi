
"""
Simple Langchain Streamlit App with Groq
A beginner-friendly version focusing on core concepts
"""

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Page config
st.set_page_config(
    page_title="Simple Langchain Chatbot with Groq",
    page_icon="🤖"
)

# Title
st.title("🤖 Simple Langchain Chat with Groq")
st.markdown("Learn LangChain basics with Groq's ultra-fast inference!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "GROQ API Key",
        type="password",
        help="Get a free API key at console.groq.com"
    )

    model_name = st.selectbox(
        "Model",
        [
            "openai/gpt-oss-safeguard-20b",
            "canopylabs/orpheus-v1-english",
            "openai/gpt-oss-20b"
        ],
        index=2
    )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# Initialize LLM
@st.cache_resource
def get_chain(api_key, model_name):

    if not api_key:
        return None

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.7,
        streaming=True
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant powered by Groq. "
            "Answer questions clearly and concisely."
        ),
        ("user", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    return chain


# Get chain
chain = get_chain(api_key, model_name)

if not chain:

    st.warning(
        "Please enter your Groq API Key in the sidebar to start chatting!"
    )

    st.markdown(
        "[Get your API Key here](https://console.groq.com)"
    )

else:

    # Display previous chat messages
    for chat in st.session_state.messages:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

    # Chat input
    if question := st.chat_input("Ask me anything"):

        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        # Generate response
        with st.chat_message("assistant"):

            message_placeholder = st.empty()
            full_response = ""

            try:

                for chunk in chain.stream({"question": question}):

                    full_response += chunk

                    message_placeholder.markdown(
                        full_response + "▌"
                    )

                message_placeholder.markdown(full_response)

                # Add assistant response to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })

            except Exception as e:
                st.error(f"Error: {e}")


# Examples
st.markdown("---")
st.markdown("### Try these examples:")

col1, col2 = st.columns(2)

with col1:
    st.markdown("- What is LangChain?")
    st.markdown("- Explain Groq's LPU technology")

with col2:
    st.markdown("- How do I learn programming?")
    st.markdown("- Write a haiku about AI")

# Footer
st.markdown("---")
st.markdown("Built with LangChain and Groq | Experience the speed")
