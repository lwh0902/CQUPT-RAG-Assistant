"""Account-scoped model generation settings."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from database import engine
from models import User, UserModelSettings
from security import get_current_user


router = APIRouter(prefix="/settings", tags=["settings"])


class ModelSettingsUpdate(BaseModel):
    temperature: float = Field(ge=0.0, le=1.0)
    top_p: float = Field(ge=0.1, le=1.0)


def _read_or_create_settings(db: Session, user_id: str) -> UserModelSettings:
    settings = db.scalar(select(UserModelSettings).where(UserModelSettings.user_id == user_id))
    if settings is None:
        settings = UserModelSettings(user_id=user_id)
        db.add(settings)
        db.flush()
    return settings


@router.get("/model")
def read_model_settings(current_user: User = Depends(get_current_user)) -> dict[str, float]:
    with Session(engine) as db:
        settings = _read_or_create_settings(db, current_user.id)
        db.commit()
        return {"temperature": settings.temperature, "top_p": settings.top_p}


@router.put("/model")
def update_model_settings(
    body: ModelSettingsUpdate,
    current_user: User = Depends(get_current_user),
) -> dict[str, float]:
    with Session(engine) as db:
        settings = _read_or_create_settings(db, current_user.id)
        settings.temperature = body.temperature
        settings.top_p = body.top_p
        db.commit()
        return {"temperature": settings.temperature, "top_p": settings.top_p}
