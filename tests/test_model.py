"""Tests des règles métier : coût, seuil, score, décision."""

import numpy as np

from scoring.model import best_threshold, decide, expected_cost, proba_to_score


def test_cout_tout_accepter():
    y = np.array([1, 0, 0, 1])
    # tout accepter : 2 mauvais payeurs acceptés × 5 / 4 dossiers
    assert expected_cost(y, np.zeros(4), threshold=0.5) == 2.5


def test_seuil_optimal_sur_un_cas_simple():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.9])
    thr, curve = best_threshold(y, p)
    assert expected_cost(y, p, thr) == 0
    assert curve["cout"].min() == 0


def test_score_monotone_et_borne():
    scores = [proba_to_score(p) for p in [0.01, 0.1, 0.3, 0.5, 0.9, 0.999]]
    assert scores == sorted(scores, reverse=True)
    assert proba_to_score(0.5) == 600
    assert all(300 <= s <= 900 for s in scores)


def test_trois_zones_de_decision():
    assert decide(0.10, threshold=0.24).code == "accorde"
    assert decide(0.30, threshold=0.24).code == "etude"
    assert decide(0.70, threshold=0.24).code == "refuse"
