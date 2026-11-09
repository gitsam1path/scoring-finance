"""Tests du chargement et du décodage des données."""

from scoring import data as D


def test_dimensions_et_taux_de_defaut():
    df = D.load_dataset()
    assert len(df) == 1000
    assert df[D.TARGET].mean() == 0.3  # 300 mauvais payeurs, comme indiqué par l'UCI


def test_aucun_code_restant():
    df = D.load_dataset()
    for col in D.CODES:
        assert not df[col].astype(str).str.match(r"^A\d+$").any(), col


def test_variables_sensibles_exclues_du_modele():
    for col in D.SENSITIVE:
        assert col not in D.FEATURES
