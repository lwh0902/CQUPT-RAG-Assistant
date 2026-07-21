import pytest
from fastapi import HTTPException

from security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    validate_jwt_secret,
)


def test_validate_jwt_secret_rejects_placeholders() -> None:
    with pytest.raises(RuntimeError):
        validate_jwt_secret("change-me-to-a-random-string")
    with pytest.raises(RuntimeError):
        validate_jwt_secret("short")
    assert len(validate_jwt_secret("x" * 32)) == 32


def test_access_and_refresh_tokens_have_distinct_types() -> None:
    access = create_access_token("user-1")
    refresh = create_refresh_token("user-1")

    access_payload = decode_token(access, expected_type="access")
    refresh_payload = decode_token(refresh, expected_type="refresh")

    assert access_payload["sub"] == "user-1"
    assert refresh_payload["sub"] == "user-1"
    assert access_payload["type"] == "access"
    assert refresh_payload["type"] == "refresh"

    with pytest.raises(HTTPException) as exc:
        decode_token(access, expected_type="refresh")
    assert exc.value.status_code == 401
