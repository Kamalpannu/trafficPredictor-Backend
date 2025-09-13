"""
Router for traffic recommendations using LLM.
"""
from fastapi import APIRouter, HTTPException
from app.schemas import RecommendationRequest, RecommendationResponse
from app.services.llm_service import get_recommendations
import logging

router = APIRouter(prefix="", tags=["Recommendations"])
logger = logging.getLogger(__name__)

@router.post("/recommendations", response_model=RecommendationResponse)
async def get_traffic_recommendations(request: RecommendationRequest):
    """Get actionable recommendations for drivers based on traffic data using GPT-4."""
    try:
        recs = await get_recommendations(request.current, request.prediction)
        return RecommendationResponse(recommendations=recs)
    except Exception as e:
        logger.error(f"Error in /recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recommendations.")
