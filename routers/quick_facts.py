"""Public quick-fact listing for the empty-state quick links."""

from fastapi import APIRouter

from services.quick_facts import list_public_facts

router = APIRouter(prefix="/quick-facts", tags=["quick_facts"])


@router.get("")
def list_quick_facts() -> dict[str, list[dict[str, object]]]:
    return {"facts": list_public_facts()}
