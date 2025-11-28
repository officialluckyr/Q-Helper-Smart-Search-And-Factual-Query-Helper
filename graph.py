from typing import TypedDict
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from agents.interface import interface_llm, router_decision
from agents.general import search_node, conversation_node
from langgraph.checkpoint.memory import MemorySaver
import os

class State(TypedDict):
    messages : Annotated[list, add_messages]

def create_graph():
    graph_builder = StateGraph(State)


    graph_builder.add_node("interface_node", interface_llm)
    graph_builder.add_node("search_node", search_node)
    graph_builder.add_node("conversation_node", conversation_node)


    graph_builder.add_edge(START, "interface_node")
    graph_builder.add_conditional_edges(
        "interface_node",
        router_decision,
        {
            "search_node": "search_node",
            "conversation_node": "conversation_node"
        }
    )
    graph_builder.add_edge("search_node", END)
    graph_builder.add_edge("conversation_node", END)

    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)

graph = create_graph()