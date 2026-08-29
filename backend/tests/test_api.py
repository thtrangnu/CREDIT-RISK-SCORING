import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.db.models import Base
from backend.app.dependencies import get_scorer, get_store
from backend.app.db.session import get_db
from backend.app.main import app
from backend.tests.conftest import APPLICANT_NO_HISTORY, APPLICANT_WITH_HISTORY


@pytest.fixture
def client(synthetic_store, real_scorer):
    # StaticPool: by default every new connection to "sqlite:///:memory:" is a SEPARATE
    # empty database. Without StaticPool, create_all() and the later inserts would run
    # against two different "databases" -> "no such table".
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)

    def _get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_store] = lambda: synthetic_store
    app.dependency_overrides[get_scorer] = lambda: real_scorer
    app.dependency_overrides[get_db] = _get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_search_applicants_returns_all_when_no_query(client):
    resp = client.get("/api/applicants")
    assert resp.status_code == 200
    ids = {row["sk_id_curr"] for row in resp.json()}
    assert ids == {APPLICANT_WITH_HISTORY, APPLICANT_NO_HISTORY}


def test_search_applicants_filters_by_query(client):
    resp = client.get(f"/api/applicants?q={APPLICANT_WITH_HISTORY}")
    assert resp.status_code == 200
    assert all(row["sk_id_curr"] == APPLICANT_WITH_HISTORY for row in resp.json())


def test_score_unknown_applicant_returns_404(client):
    resp = client.post("/api/score/999999")
    assert resp.status_code == 404


def test_score_known_applicant_returns_valid_response_and_writes_history(client):
    resp = client.post(f"/api/score/{APPLICANT_WITH_HISTORY}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["sk_id_curr"] == APPLICANT_WITH_HISTORY
    assert 0.0 <= body["pd_score"] <= 1.0
    assert len(body["reasons"]) == 5
    assert body["target_actual"] == 1  # SK_ID_CURR=100002 really does have TARGET=1

    history = client.get("/api/history").json()
    assert history["total"] == 1
    assert history["items"][0]["sk_id_curr"] == APPLICANT_WITH_HISTORY


def test_history_filters_by_sk_id_curr(client):
    client.post(f"/api/score/{APPLICANT_WITH_HISTORY}")
    client.post(f"/api/score/{APPLICANT_NO_HISTORY}")

    resp = client.get(f"/api/history?sk_id_curr={APPLICANT_NO_HISTORY}")
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["sk_id_curr"] == APPLICANT_NO_HISTORY


def test_insights_endpoint_returns_global_importance_and_metrics(client):
    resp = client.get("/api/insights")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["global_importance"]) > 0
    assert "engineered" in body["model_metrics"]
    assert len(body["monotonic_features"]) == 18
