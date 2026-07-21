from routers.auth import AUTH_FAILURE_DETAIL, build_registration_user


def test_registration_users_with_matching_phone_suffixes_get_distinct_usernames() -> None:
    first = build_registration_user("18128160001", "Test123456")
    second = build_registration_user("19900000001", "Test123456")

    assert first.username != second.username
    assert first.username.startswith("user_")
    assert second.username.startswith("user_")


def test_auth_failure_message_is_generic() -> None:
    assert AUTH_FAILURE_DETAIL == "手机号或密码错误"
