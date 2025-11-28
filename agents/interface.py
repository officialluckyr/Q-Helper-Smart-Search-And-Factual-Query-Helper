from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))

def interface_llm(state):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a smart assistant. Classify the user's message into one of two categories: 'Factual' (if it asks for a definition, fact, or specific information about a topic like Python, AI, etc.) or 'Conversational' (if it is a greeting, small talk, general chat, or questions about yourself, the user, or previous context like 'What is my name?'). Return ONLY the category name."),
            ("user", "Message: {message}")
        ]
    )
    chain = prompt|llm
    user_msg = state["messages"][-1].content
    response = chain.invoke({"message": user_msg})
    return {"messages": state["messages"] + [response]}


def router_decision(state) :
    classification = state["messages"][-1].content.lower().strip()

    if "factual" in classification:
        return "search_node"
    else:
        return "conversation_node"