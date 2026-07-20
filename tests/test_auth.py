from routers.auth import build_registration_user


def test_registration_users_with_matching_phone_suffixes_get_distinct_usernames() -> None:
    first = build_registration_user("18128160001", "Test123456")
    second = build_registration_user("19900000001", "Test123456")

    assert first.username != second.username
    assert first.username.startswith("user_")
    assert second.username.startswith("user_")
