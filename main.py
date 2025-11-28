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


    
    return {
        "response": final_response
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
