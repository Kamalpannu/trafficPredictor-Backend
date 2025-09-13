from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from app.database import get_db
from app.models.models import PredictionRecord
from app.services.llm_service import get_recommendations


router = APIRouter()

@router.get("/recommendations")
async def recommendations(db: AsyncSession = Depends(get_db)):
    zones = ["Zone 1", "Zone 2", "Zone 3"]
    latest_preds = []

    for zone in zones:
        stmt = select(PredictionRecord).where(
            PredictionRecord.zone == zone
        ).order_by(desc(PredictionRecord.timestamp)).limit(1)

        result = await db.execute(stmt)
        latest = result.scalar_one_or_none()
        if latest:
            latest_preds.append({
                "zone": latest.zone,
                "predicted_count": latest.predicted_count
            })

    if not latest_preds:
        raise HTTPException(status_code=404, detail="No predictions found.")

    hints = await get_recommendations(latest_preds)
    return {"recommendations": hints}
