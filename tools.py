def search_dictionary(query: str) -> str:
    """
    A simple dictionary-based search tool that returns factual answers.
    """
    data = {
        "ai": "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems.",
        "python": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.",
        "langchain": "LangChain is a framework designed to simplify the creation of applications using large language models (LLMs).",
        "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of LangChain.",
        "agent": "An intelligent agent is a system that perceives its environment and takes actions that maximize its chances of success.",
        "fastapi": "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints."
    }
    
    query = query.lower().strip()
    
    # Simple keyword matching
    for key, value in data.items():
        if key in query:
            return value
            
    return "I'm sorry, I don't have information about that specific topic in my dictionary."
