"""
API de scoring de crédit.

Lancer (depuis la racine du projet) :
    uvicorn api.main:app --reload
Documentation interactive : http://localhost:8000/docs
Page de démonstration      : http://localhost:8000
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # rend le package `scoring` importable

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from scoring.data import categories  # noqa: E402
from scoring.model import SCORE_MAX, SCORE_MIN, decide, explain, proba_to_score  # noqa: E402

from .schemas import Demande, DemandesLot, Facteur, Prediction  # noqa: E402

MODEL_PATH = ROOT / "models" / "model.joblib"
STATIC = Path(__file__).parent / "static"

LABELS = {
    "statut_compte": "Solde du compte courant",
    "duree_mois": "Durée du crédit",
    "historique_credit": "Historique de remboursement",
    "objet_credit": "Objet du crédit",
    "montant": "Montant demandé",
    "epargne": "Épargne",
    "anciennete_emploi": "Ancienneté dans l'emploi",
    "taux_effort": "Taux d'effort",
    "autres_debiteurs": "Co-emprunteur / garant",
    "anciennete_residence": "Ancienneté dans le logement",
    "patrimoine": "Patrimoine",
    "age": "Âge",
    "autres_credits": "Autres crédits en cours",
    "logement": "Logement",
    "nb_credits": "Crédits dans cette banque",
    "emploi": "Emploi",
    "nb_personnes_a_charge": "Personnes à charge",
    "telephone": "Téléphone",
}
UNITS = {"duree_mois": " mois", "montant": " DM", "age": " ans"}

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Le modèle est chargé UNE fois au démarrage, pas à chaque requête.
    if not MODEL_PATH.exists():
        raise RuntimeError("Modèle introuvable : lancez d'abord l'entraînement (python -m scoring.train).")
    state["bundle"] = joblib.load(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(
    title="Scoring Finance — API de scoring de crédit",
    description=(
        "Estime la probabilité de défaut d'un demandeur de crédit, la traduit en score (300-900), "
        "propose une décision et explique les principaux facteurs. Données : German Credit (UCI)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def fmt_value(variable: str, value) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value}{UNITS.get(variable, '')}"


def score_one(demande: Demande) -> Prediction:
    b = state["bundle"]
    row = pd.DataFrame([demande.model_dump()])[b["features"]]
    p = float(b["pipeline"].predict_proba(row)[0, 1])
    effects = explain(b["pipeline"], row, b["reference"])

    def facteurs(df):
        return [
            Facteur(variable=r.variable, libelle=LABELS[r.variable], valeur=fmt_value(r.variable, r.valeur),
                    effet=round(float(r.effet) * 100, 1))
            for r in df.itertuples()
        ]

    risk = effects[effects["effet"] > 0.005].head(3)
    good = effects[effects["effet"] < -0.005].head(3)
    d = decide(p, b["seuil"])
    return Prediction(
        probabilite_defaut=round(p, 4),
        score=proba_to_score(p),
        decision={"code": d.code, "libelle": d.libelle},
        seuil=b["seuil"],
        facteurs_risque=facteurs(risk),
        facteurs_favorables=facteurs(good),
    )


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/health", tags=["Supervision"])
def health():
    """Vérifie que l'API tourne et que le modèle est chargé."""
    return {"statut": "ok", "modele_charge": "bundle" in state}


@app.get("/model", tags=["Modèle"])
def model_info():
    """Informations sur le modèle : type, seuil, performances sur le jeu de test, valeurs acceptées."""
    b = state["bundle"]
    m = b["metrics"]
    return {
        "modele": b["modele"],
        "version": b["version"],
        "seuil": b["seuil"],
        "echelle_score": {"min": SCORE_MIN, "max": SCORE_MAX},
        "performances_test": {k: m[k] for k in ["auc", "gini", "ks", "rappel_defauts", "taux_acceptation",
                                                  "cout_moyen_modele", "cout_moyen_tout_accepter"]},
        "categories": categories(),
    }


@app.post("/predict", response_model=Prediction, tags=["Scoring"])
def predict(demande: Demande):
    """Score une demande de crédit et explique la décision."""
    return score_one(demande)


@app.post("/predict/batch", response_model=list[Prediction], tags=["Scoring"])
def predict_batch(lot: DemandesLot):
    """Score jusqu'à 100 demandes en une fois."""
    try:
        return [score_one(d) for d in lot.demandes]
    except Exception as err:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(err)) from err
