from tools import search_dictionary
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))

def search_node(state):
    """
    Node that handles factual queries by calling the search tool.
    """
    # The last message is the classification from interface_node. 
    # The user message is the one before that.
    user_msg = state["messages"][-2].content
    # Extract the actual query from the message if needed, or just pass the whole message
    # For simplicity, we'll pass the whole message as the query
    result = search_dictionary(user_msg)
    
    # We return a message with the result
    return {"messages": state["messages"] + [{"role": "assistant", "content": result}]}

def conversation_node(state):
    """
    Node that handles normal conversation.
    """
    # Filter out classification messages to keep history clean for the LLM
    history = [m for m in state["messages"] if m.content.strip() not in ["Factual", "Conversational"]]
    
    # The last message in history should be the user's message.
    # We can pass the whole history to the LLM.
    
    # We need to ensure the system message is first.
    system_message = ("system", "You are a helpful and friendly AI assistant. Engage in normal conversation with the user.")
    
    # If we use invoke with a list of messages, it's easier.
    # We don't need a template if we just pass the list, but we need to prepend system message.
    
    messages_to_send = [system_message] + history
    
    response = llm.invoke(messages_to_send)
    return {"messages": state["messages"] + [response]}
