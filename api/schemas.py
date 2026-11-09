"""
Schémas Pydantic : ce que l'API accepte et ce qu'elle renvoie.
Les valeurs autorisées des variables catégorielles viennent directement de src/scoring/data.py :
impossible d'envoyer une catégorie que le modèle ne connaît pas.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from scoring.data import CODES


def choices(col: str):
    return Literal[tuple(CODES[col].values())]  # type: ignore[valid-type]


EXAMPLE = {
    "statut_compte": "0 à 200 DM",
    "duree_mois": 24,
    "historique_credit": "crédits en cours remboursés à temps",
    "objet_credit": "voiture d'occasion",
    "montant": 3500,
    "epargne": "moins de 100 DM",
    "anciennete_emploi": "1 à 4 ans",
    "taux_effort": 3,
    "autres_debiteurs": "aucun",
    "anciennete_residence": 2,
    "patrimoine": "voiture ou autre",
    "age": 32,
    "autres_credits": "aucun",
    "logement": "propriétaire",
    "nb_credits": 1,
    "emploi": "qualifié / employé",
    "nb_personnes_a_charge": 1,
    "telephone": "oui",
}


class Demande(BaseModel):
    """Une demande de crédit. Les montants sont en Deutsche Marks (DM), comme dans les données d'origine."""

    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE})

    statut_compte: choices("statut_compte") = Field(description="Solde du compte courant")
    duree_mois: int = Field(ge=1, le=120, description="Durée du crédit en mois")
    historique_credit: choices("historique_credit") = Field(description="Historique de remboursement")
    objet_credit: choices("objet_credit") = Field(description="Objet du crédit")
    montant: float = Field(gt=0, le=100_000, description="Montant demandé (DM)")
    epargne: choices("epargne") = Field(description="Épargne disponible")
    anciennete_emploi: choices("anciennete_emploi") = Field(description="Ancienneté dans l'emploi actuel")
    taux_effort: int = Field(ge=1, le=4, description="Mensualité en % du revenu disponible, de 1 (faible) à 4 (élevé)")
    autres_debiteurs: choices("autres_debiteurs") = Field(description="Co-emprunteur ou garant")
    anciennete_residence: int = Field(ge=1, le=4, description="Ancienneté dans le logement, de 1 (récent) à 4 (ancien)")
    patrimoine: choices("patrimoine") = Field(description="Patrimoine le plus important")
    age: int = Field(ge=18, le=100, description="Âge en années")
    autres_credits: choices("autres_credits") = Field(description="Autres crédits en cours")
    logement: choices("logement") = Field(description="Situation de logement")
    nb_credits: int = Field(ge=1, le=10, description="Nombre de crédits en cours dans cette banque")
    emploi: choices("emploi") = Field(description="Catégorie d'emploi")
    nb_personnes_a_charge: int = Field(ge=0, le=10, description="Nombre de personnes à charge")
    telephone: choices("telephone") = Field(description="Téléphone à son nom")


class Facteur(BaseModel):
    variable: str
    libelle: str
    valeur: str
    effet: float = Field(description="Variation de la probabilité de défaut par rapport au client type (en points de %)")


class Decision(BaseModel):
    code: Literal["accorde", "etude", "refuse"]
    libelle: str


class Prediction(BaseModel):
    probabilite_defaut: float = Field(description="Probabilité estimée que le client ne rembourse pas (0 à 1)")
    score: int = Field(description="Score de 300 à 900 points : plus il est haut, moins le risque est élevé")
    decision: Decision
    seuil: float = Field(description="Probabilité au-delà de laquelle le dossier n'est pas accordé automatiquement")
    facteurs_risque: list[Facteur] = Field(description="Ce qui augmente le risque de ce client")
    facteurs_favorables: list[Facteur] = Field(description="Ce qui diminue le risque de ce client")


class DemandesLot(BaseModel):
    demandes: list[Demande] = Field(min_length=1, max_length=100)
