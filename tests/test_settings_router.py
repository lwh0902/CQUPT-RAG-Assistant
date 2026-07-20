from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import routers.settings as settings_router
from database import Base
from models import User


def test_model_settings_default_and_update_are_scoped_to_current_user(monkeypatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(settings_router, "engine", engine)
    user = User(id="user-1", username="user_1")

    with Session(engine) as db:
        db.add(user)
        db.commit()

    current_user = User(id="user-1", username="user_1")

    initial = settings_router.read_model_settings(current_user=current_user)
    updated = settings_router.update_model_settings(
        body=settings_router.ModelSettingsUpdate(temperature=0.6, top_p=0.7),
        current_user=current_user,
    )

    assert initial == {"temperature": 0.3, "top_p": 0.8}
    assert updated == {"temperature": 0.6, "top_p": 0.7}
