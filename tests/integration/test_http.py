from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from config.settings import Settings
from main import create_app


def test_health_does_not_depend_on_database_schema(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["model"] == "not_configured"
        response = client.get("/api/v1/profiles")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "database_unavailable"
        assert "SELECT" not in response.text
        assert settings.database_url.get_secret_value() not in response.text


def test_create_read_and_persist_profile_across_app_restart(migrated_settings: Settings) -> None:
    with TestClient(create_app(migrated_settings)) as client:
        assert client.get("/ready").status_code == 200
        response = client.post("/api/v1/profiles", json={"name": "  Арсений  "})
        assert response.status_code == 201
        profile = response.json()
        UUID(profile["id"])
        assert profile["name"] == "Арсений"
        assert profile["created_at"].endswith("Z")
        assert response.headers["location"] == f"/api/v1/profiles/{profile['id']}"
        assert client.get(response.headers["location"]).json() == profile
    with TestClient(create_app(migrated_settings)) as client:
        response = client.get("/api/v1/profiles")
        assert response.status_code == 200
        assert response.json()["items"] == [profile]


def test_profile_pagination_and_distinct_identifiers(migrated_settings: Settings) -> None:
    with TestClient(create_app(migrated_settings)) as client:
        first = client.post("/api/v1/profiles", json={"name": "Первый"}).json()
        second = client.post("/api/v1/profiles", json={"name": "Второй"}).json()
        assert first["id"] != second["id"]
        response = client.get("/api/v1/profiles?limit=1&offset=1")
        assert response.json() == {"items": [second], "limit": 1, "offset": 1}


@pytest.mark.parametrize("body", [{"name": " "}, {"name": "x" * 81}, {}, {"name": "Valid", "id": "spoofed"}])
def test_invalid_profile_input_never_persists(migrated_settings: Settings, body: dict) -> None:
    with TestClient(create_app(migrated_settings)) as client:
        assert client.post("/api/v1/profiles", json=body).status_code == 422
        assert client.get("/api/v1/profiles").json()["items"] == []


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "offset=-1"])
def test_invalid_pagination(migrated_settings: Settings, query: str) -> None:
    with TestClient(create_app(migrated_settings)) as client:
        assert client.get(f"/api/v1/profiles?{query}").status_code == 422


def test_missing_and_malformed_profile_id(migrated_settings: Settings) -> None:
    with TestClient(create_app(migrated_settings)) as client:
        assert client.get("/api/v1/profiles/not-a-uuid").status_code == 422
        response = client.get(f"/api/v1/profiles/{uuid4()}")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "profile_not_found"


def test_cors_accepts_only_configured_frontend_origins(migrated_settings: Settings) -> None:
    with TestClient(create_app(migrated_settings)) as client:
        response = client.options(
            "/api/v1/profiles",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
        response = client.options(
            "/api/v1/profiles",
            headers={
                "Origin": "https://unrelated.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code == 400
        assert "access-control-allow-origin" not in response.headers
