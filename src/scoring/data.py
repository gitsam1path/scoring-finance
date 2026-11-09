"""
Chargement et préparation du German Credit Dataset (UCI Statlog).

Le fichier d'origine (data/raw/german.data) utilise des codes (A11, A34…).
Ce module les traduit en libellés français lisibles, d'après la documentation
officielle du jeu de données, et sépare :
  - les variables utilisées par le modèle ;
  - les variables sensibles, exclues du modèle mais conservées pour l'audit d'équité.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "german.data"

# Ordre des 21 colonnes du fichier d'origine
RAW_COLUMNS = [
    "statut_compte", "duree_mois", "historique_credit", "objet_credit", "montant",
    "epargne", "anciennete_emploi", "taux_effort", "situation_sexe", "autres_debiteurs",
    "anciennete_residence", "patrimoine", "age", "autres_credits", "logement",
    "nb_credits", "emploi", "nb_personnes_a_charge", "telephone", "travailleur_etranger", "cible",
]

# Traduction des codes (documentation UCI Statlog German Credit, 1 DM = Deutsche Mark)
CODES = {
    "statut_compte": {
        "A11": "négatif (< 0 DM)",
        "A12": "0 à 200 DM",
        "A13": "200 DM ou plus",
        "A14": "pas de compte courant",
    },
    "historique_credit": {
        "A30": "aucun crédit / tous remboursés",
        "A31": "crédits de cette banque remboursés",
        "A32": "crédits en cours remboursés à temps",
        "A33": "retards de paiement passés",
        "A34": "compte critique / crédits ailleurs",
    },
    "objet_credit": {
        "A40": "voiture neuve",
        "A41": "voiture d'occasion",
        "A42": "mobilier / équipement",
        "A43": "radio / télévision",
        "A44": "électroménager",
        "A45": "réparations",
        "A46": "études",
        "A47": "vacances",
        "A48": "reconversion",
        "A49": "professionnel",
        "A410": "autre",
    },
    "epargne": {
        "A61": "moins de 100 DM",
        "A62": "100 à 500 DM",
        "A63": "500 à 1 000 DM",
        "A64": "1 000 DM ou plus",
        "A65": "inconnue / aucune",
    },
    "anciennete_emploi": {
        "A71": "sans emploi",
        "A72": "moins d'1 an",
        "A73": "1 à 4 ans",
        "A74": "4 à 7 ans",
        "A75": "7 ans ou plus",
    },
    "situation_sexe": {
        "A91": "homme divorcé/séparé",
        "A92": "femme divorcée/séparée/mariée",
        "A93": "homme célibataire",
        "A94": "homme marié/veuf",
        "A95": "femme célibataire",
    },
    "autres_debiteurs": {"A101": "aucun", "A102": "co-emprunteur", "A103": "garant"},
    "patrimoine": {
        "A121": "immobilier",
        "A122": "épargne logement / assurance vie",
        "A123": "voiture ou autre",
        "A124": "aucun / inconnu",
    },
    "autres_credits": {"A141": "banque", "A142": "magasins", "A143": "aucun"},
    "logement": {"A151": "locataire", "A152": "propriétaire", "A153": "logé gratuitement"},
    "emploi": {
        "A171": "sans emploi / non qualifié non résident",
        "A172": "non qualifié",
        "A173": "qualifié / employé",
        "A174": "cadre / indépendant / très qualifié",
    },
    "telephone": {"A191": "non", "A192": "oui"},
    "travailleur_etranger": {"A201": "oui", "A202": "non"},
}

NUMERIC_FEATURES = [
    "duree_mois", "montant", "taux_effort", "anciennete_residence", "age", "nb_credits", "nb_personnes_a_charge",
]
CATEGORICAL_FEATURES = [
    "statut_compte", "historique_credit", "objet_credit", "epargne", "anciennete_emploi",
    "autres_debiteurs", "patrimoine", "autres_credits", "logement", "emploi", "telephone",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Exclues du modèle : décider d'un crédit selon le sexe, la situation familiale ou la nationalité
# serait discriminatoire. Elles servent uniquement à vérifier l'équité des décisions.
SENSITIVE = ["situation_sexe", "travailleur_etranger"]
TARGET = "defaut"  # 1 = mauvais payeur (défaut), 0 = bon payeur

# Matrice de coûts fournie avec le jeu de données
COST_FALSE_ACCEPT = 5  # accorder un crédit à un mauvais payeur
COST_FALSE_REJECT = 1  # refuser un bon payeur


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=RAW_COLUMNS)


def decode(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace les codes par les libellés et crée la cible binaire `defaut`."""
    out = df.copy()
    for col, mapping in CODES.items():
        unknown = set(out[col]) - set(mapping)
        if unknown:
            raise ValueError(f"Codes inconnus dans {col} : {sorted(unknown)}")
        out[col] = out[col].map(mapping)
    out[TARGET] = (out.pop("cible") == 2).astype(int)
    out["sexe"] = out["situation_sexe"].str.split().str[0]  # pour l'audit d'équité
    return out


def load_dataset(path: Path = RAW_PATH) -> pd.DataFrame:
    return decode(load_raw(path))


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES], df[TARGET]


def categories() -> dict[str, list[str]]:
    """Valeurs possibles de chaque variable catégorielle (utilisé par l'API)."""
    return {col: list(CODES[col].values()) for col in CATEGORICAL_FEATURES}
