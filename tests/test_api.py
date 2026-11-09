"""Tests de l'API (le modèle entraîné doit exister : models/model.joblib)."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import EXAMPLE


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # "with" déclenche le chargement du modèle
        yield c


def test_health(client):
    assert client.get("/health").json() == {"statut": "ok", "modele_charge": True}


def test_model_info(client):
    info = client.get("/model").json()
    assert 0 < info["seuil"] < 1
    assert info["performances_test"]["auc"] > 0.7
    assert "pas de compte courant" in info["categories"]["statut_compte"]


def test_predict(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probabilite_defaut"] <= 1
    assert 300 <= body["score"] <= 900
    assert body["decision"]["code"] in {"accorde", "etude", "refuse"}
    assert all(f["effet"] > 0 for f in body["facteurs_risque"])
    assert all(f["effet"] < 0 for f in body["facteurs_favorables"])


def test_un_profil_fragile_est_plus_risque(client):
    solide = {**EXAMPLE, "statut_compte": "pas de compte courant", "duree_mois": 12, "epargne": "1 000 DM ou plus"}
    fragile = {**EXAMPLE, "statut_compte": "négatif (< 0 DM)", "duree_mois": 60, "historique_credit": "retards de paiement passés"}
    p_solide = client.post("/predict", json=solide).json()["probabilite_defaut"]
    p_fragile = client.post("/predict", json=fragile).json()["probabilite_defaut"]
    assert p_fragile > p_solide


@pytest.mark.parametrize("change", [{"age": 12}, {"statut_compte": "millionnaire"}, {"duree_mois": 0}])
def test_entrees_invalides_refusees(client, change):
    assert client.post("/predict", json={**EXAMPLE, **change}).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"demandes": [EXAMPLE, EXAMPLE]})
    assert r.status_code == 200 and len(r.json()) == 2
