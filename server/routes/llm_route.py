from fastapi import APIRouter, Depends, HTTPException, Body, BackgroundTasks,Path
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from controllers.llm_controller import retrieval_llama_background
from controllers.llm_controller1 import retrieval_llama
from controllers.saq_controller1 import security_assesment_questionnaire_result


router = APIRouter()

@router.get("/llm1/{user_id}/{req_id}")
async def llama_pinecone_result_fetcher_route_1(background_tasks: BackgroundTasks, user_id: str = Path(...), req_id: str = Path(...)):
    background_tasks.add_task(retrieval_llama_background, background_tasks, user_id, req_id)
    return {"message": "Task started"}

@router.get("/llm/{user_id}/{req_id}")
async def llama_pinecone_result_fetcher_route(user_id: str = Path(...),
    req_id: str = Path(...),# Optional body parameter
    question_details: Optional[Dict[str, Any]] = Body(None)  # Optional body parameter
):
    return await retrieval_llama(
        user_id, req_id, question_details["question_details"] or [], True
    )

@router.get("/saq/{user_id}/{req_id}")
async def security_assesment_questionnaire_result_route(
    user_id: str = Path(...),
    req_id: str = Path(...),
    previous_question_details: Optional[list] = Body(None),  # Optional body parameter
    question_details: Optional[list] = Body(None)  # Optional body parameter
):
    return await security_assesment_questionnaire_result(
        user_id, req_id, previous_question_details or [], question_details or []
    )