from fastapi import APIRouter, Depends, HTTPException, Body, BackgroundTasks,Path
from typing import List, Dict, Optional
from pydantic import BaseModel
from controllers.llm_controller import retrieval_llama_background


router = APIRouter()

@router.get("/llm/{user_id}/{req_id}")
async def llama_pinecone_result_fetcher_route(background_tasks: BackgroundTasks, user_id: str = Path(...), req_id: str = Path(...)):
    background_tasks.add_task(retrieval_llama_background, background_tasks, user_id, req_id)
    return {"message": "Task started"}