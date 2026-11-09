"""
Construction des pipelines, score sur échelle de points, décision et motifs.

Tout ce qui est utilisé à la fois par l'entraînement et par l'API se trouve ici,
pour que les deux appliquent exactement les mêmes règles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, COST_FALSE_ACCEPT, COST_FALSE_REJECT, NUMERIC_FEATURES

SEED = 42
LOG_FEATURES = ["montant", "duree_mois"]  # très asymétriques : passage au logarithme
OTHER_NUMERIC = [c for c in NUMERIC_FEATURES if c not in LOG_FEATURES]


# ─────────────────────────────────────────────────────────────
# Pipelines : préparation des données + modèle, en un seul objet
# ─────────────────────────────────────────────────────────────
def preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("log", Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                              ("scale", StandardScaler())]), LOG_FEATURES),
            ("num", StandardScaler(), OTHER_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def candidate_models() -> dict[str, Pipeline]:
    """Les modèles comparés. Les hyperparamètres restent volontairement simples (1 000 lignes seulement)."""
    return {
        "Régression logistique": Pipeline([
            ("prep", preprocessor()),
            ("model", LogisticRegression(C=0.3, max_iter=2000)),
        ]),
        "Forêt aléatoire": Pipeline([
            ("prep", preprocessor()),
            ("model", RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features="sqrt",
                                             random_state=SEED, n_jobs=-1)),
        ]),
        "Gradient boosting": Pipeline([
            ("prep", preprocessor()),
            ("model", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_depth=3,
                                                     l2_regularization=1.0, random_state=SEED)),
        ]),
    }


# ─────────────────────────────────────────────────────────────
# Coût métier et seuil de décision
# ─────────────────────────────────────────────────────────────
def expected_cost(y_true, proba, threshold: float) -> float:
    """Coût moyen par dossier : on refuse si proba >= seuil."""
    y_true = np.asarray(y_true)
    refuse = np.asarray(proba) >= threshold
    false_accept = ((~refuse) & (y_true == 1)).sum()  # mauvais payeur accepté
    false_reject = (refuse & (y_true == 0)).sum()  # bon payeur refusé
    return (COST_FALSE_ACCEPT * false_accept + COST_FALSE_REJECT * false_reject) / len(y_true)


def best_threshold(y_true, proba, grid=None) -> tuple[float, pd.DataFrame]:
    """Seuil qui minimise le coût moyen, et la courbe coût = f(seuil)."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    curve = pd.DataFrame({"seuil": grid, "cout": [expected_cost(y_true, proba, t) for t in grid]})
    return float(curve.loc[curve["cout"].idxmin(), "seuil"]), curve


# ─────────────────────────────────────────────────────────────
# Score en points (comme les scores de crédit des banques)
# ─────────────────────────────────────────────────────────────
SCORE_BASE = 600  # score pour 1 chance sur 2 de défaut
SCORE_PDO = 50  # "points to double the odds" : +50 points = 2 fois moins de risque relatif
SCORE_MIN, SCORE_MAX = 300, 900


def proba_to_score(p_default: float) -> int:
    p = min(max(p_default, 1e-6), 1 - 1e-6)
    score = SCORE_BASE + SCORE_PDO / math.log(2) * math.log((1 - p) / p)
    return int(round(min(max(score, SCORE_MIN), SCORE_MAX)))


# ─────────────────────────────────────────────────────────────
# Décision et motifs
# ─────────────────────────────────────────────────────────────
@dataclass
class Decision:
    code: str  # "accorde" | "etude" | "refuse"
    libelle: str


def decide(p_default: float, threshold: float, refuse_above: float = 0.5) -> Decision:
    """
    Trois zones, comme dans une banque :
      - risque sous le seuil optimal          → crédit accordé
      - entre le seuil et 50 %                → étude manuelle par un conseiller
      - risque de défaut de 50 % ou plus      → refus
    """
    if p_default < threshold:
        return Decision("accorde", "Crédit accordé")
    if p_default < refuse_above:
        return Decision("etude", "Étude manuelle par un conseiller")
    return Decision("refuse", "Crédit refusé")


def reference_profile(X: pd.DataFrame) -> dict:
    """Le "client type" : médiane des variables numériques, valeur la plus fréquente des catégorielles."""
    ref = {c: float(X[c].median()) for c in NUMERIC_FEATURES}
    ref.update({c: X[c].mode().iloc[0] for c in CATEGORICAL_FEATURES})
    return ref


def explain(pipeline: Pipeline, row: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """
    Motifs de la décision, valables pour n'importe quel modèle.
    Pour chaque variable, on la remplace par la valeur du client type et on regarde
    de combien la probabilité de défaut change :
      effet > 0 : la valeur de ce client AUGMENTE son risque par rapport au client type ;
      effet < 0 : elle le DIMINUE.
    Les 18 variantes sont prédites en un seul appel, pour rester rapide.
    """
    base = float(pipeline.predict_proba(row)[0, 1])
    variants = pd.concat([row] * len(reference), ignore_index=True)
    for i, (col, value) in enumerate(reference.items()):
        variants.loc[i, col] = value
    probas = pipeline.predict_proba(variants)[:, 1]
    effects = pd.DataFrame({
        "variable": list(reference),
        "valeur": [row.iloc[0][c] for c in reference],
        "effet": base - probas,
    })
    effects = effects[effects["valeur"].astype(str) != pd.Series(reference).astype(str).values]
    return effects.reindex(effects["effet"].abs().sort_values(ascending=False).index).reset_index(drop=True)
