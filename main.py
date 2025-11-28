from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from graph import graph
load_dotenv()

app = FastAPI(title="AgenticAI Support Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageRequest(BaseModel):
    message: str
    thread_id: str = "default_thread"

@app.get("/")
def root():
    return {"message" : "Welcome to AgenticAI Support Bot API"}

@app.post("/chat")
def chat(request: MessageRequest):
    
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # We need to pass the message as a list of messages for langgraph
    response = graph.invoke({"messages": [("user", request.message)]}, config=config)
    messages = response["messages"]

    # The last message is the final response
    final_response = messages[-1].content
    
    # The classification is in the second to last message (from interface_llm)
    # But since we have memory, the history grows. We need to find the classification for THIS turn.
    # However, for simplicity in this specific architecture:
    # 1. User message
    # 2. Interface LLM response (Classification)
    # 3. Search/Conversation Node response
    # So the last 3 messages are relevant to the current turn if we just started.
    # But with memory, it's safer to just return the final response. 
    # If we really need classification, we can look at the second to last message added in this turn.
    
    return {
        "response": final_response
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)