from routers.auth import AUTH_FAILURE_DETAIL, build_registration_user


def test_registration_users_with_matching_phone_suffixes_get_distinct_usernames() -> None:
    first = build_registration_user("18128160001", "Test123456")
    second = build_registration_user("19900000001", "Test123456")

    assert first.username != second.username
    assert first.username.startswith("user_")
    assert second.username.startswith("user_")


def test_auth_failure_message_is_generic() -> None:
    assert AUTH_FAILURE_DETAIL == "手机号或密码错误"



def test_normalize_and_generate_invite_code() -> None:
    from routers.auth import _generate_invite_code_value, _normalize_invite_code

    assert _normalize_invite_code(" ab12cd ") == "AB12CD"
    code = _generate_invite_code_value()
    assert len(code) == 6
    assert code.isalnum()
    assert code == code.upper()


def test_bootstrap_admin_phone_gate(monkeypatch) -> None:
    from routers import auth as auth_router
    from models import User

    monkeypatch.setattr(auth_router, "BOOTSTRAP_ADMIN_PHONE", "18128161378")
    admin = User(id="1", username="a", phone="18128161378", hashed_password="x")
    other = User(id="2", username="b", phone="19900001111", hashed_password="x")
    assert auth_router._user_can_manage_invites(admin) is True
    assert auth_router._user_can_manage_invites(other) is False
