"""
Entraînement du modèle de scoring.

Lancer :  python -m scoring.train      (depuis le dossier src, ou via `make train`)

Étapes :
  1. Séparation apprentissage (80 %) / test (20 %), stratifiée sur la cible.
  2. Comparaison de 3 modèles par validation croisée à 5 plis (AUC).
  3. Choix du modèle : le meilleur AUC, sauf si la régression logistique est à moins de 0,01 :
     on la préfère alors, car elle est explicable (motifs de refus) et stable.
  4. Choix du seuil de décision qui minimise le coût métier (prédictions hors échantillon).
  5. Évaluation finale sur le jeu de test, jamais vu jusque-là.
  6. Audit d'équité, importance des variables, figures, model card, sauvegarde du modèle.
"""

from __future__ import annotations

import json
from datetime import date

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split

from .data import (
    COST_FALSE_ACCEPT, COST_FALSE_REJECT, FEATURES, ROOT, SENSITIVE, TARGET, load_dataset,
)
from .model import SEED, best_threshold, candidate_models, decide, expected_cost, proba_to_score, reference_profile

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
PREFERENCE_MARGIN = 0.01  # écart d'AUC toléré pour préférer la régression logistique

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
COLORS = {"Régression logistique": "#2e86ab", "Forêt aléatoire": "#3d9970", "Gradient boosting": "#edae49"}


def ks_statistic(y, p) -> float:
    """Kolmogorov-Smirnov : écart maximal entre les distributions de score des bons et des mauvais payeurs."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> dict:
    MODELS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=SEED)
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]
    log(f"Données : {len(df)} dossiers · apprentissage {len(train_df)} · test {len(test_df)} · défaut {y_train.mean():.0%}")

    # ── 1. Comparaison des modèles (validation croisée) ──────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    comparison, oof = [], {}
    for name, pipe in candidate_models().items():
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        oof[name] = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        thr, _ = best_threshold(y_train, oof[name])
        comparison.append({
            "modele": name,
            "auc_cv_moyenne": float(res["test_score"].mean()),
            "auc_cv_ecart_type": float(res["test_score"].std()),
            "seuil_optimal_cv": thr,
            "cout_cv": expected_cost(y_train, oof[name], thr),
        })
        log(f"  {name:22s} AUC CV = {res['test_score'].mean():.3f} ± {res['test_score'].std():.3f}")
    comp = pd.DataFrame(comparison).sort_values("auc_cv_moyenne", ascending=False).reset_index(drop=True)

    # ── 2. Choix du modèle ───────────────────────────────────────────
    best_auc = comp["auc_cv_moyenne"].max()
    lr_auc = comp.set_index("modele").loc["Régression logistique", "auc_cv_moyenne"]
    chosen = "Régression logistique" if best_auc - lr_auc <= PREFERENCE_MARGIN else comp.iloc[0]["modele"]
    reason = (
        "meilleur AUC" if chosen == comp.iloc[0]["modele"]
        else f"à moins de {PREFERENCE_MARGIN} d'AUC du meilleur, et explicable"
    )
    log(f"Modèle retenu : {chosen} ({reason})")

    # ── 3. Seuil de décision (sur les prédictions hors échantillon) ───
    threshold, cost_curve = best_threshold(y_train, oof[chosen])
    log(f"Seuil optimal (coût {COST_FALSE_ACCEPT}:{COST_FALSE_REJECT}) : {threshold:.2f}")

    # ── 4. Entraînement final et évaluation sur le test ──────────────
    final = candidate_models()[chosen].fit(X_train, y_train)
    p_test = final.predict_proba(X_test)[:, 1]
    refuse = p_test >= threshold
    tn, fp, fn, tp = confusion_matrix(y_test, refuse.astype(int)).ravel()
    test_auc = roc_auc_score(y_test, p_test)
    metrics = {
        "auc": float(test_auc),
        "gini": float(2 * test_auc - 1),
        "ks": ks_statistic(y_test, p_test),
        "seuil": threshold,
        "accuracy": float(accuracy_score(y_test, refuse)),
        "rappel_defauts": float(recall_score(y_test, refuse)),  # part des mauvais payeurs détectés
        "precision_refus": float(precision_score(y_test, refuse)),  # part des refus qui sont de vrais mauvais payeurs
        "taux_acceptation": float((~refuse).mean()),
        "cout_moyen_modele": expected_cost(y_test, p_test, threshold),
        "cout_moyen_tout_accepter": expected_cost(y_test, p_test, 1.01),
        "cout_moyen_tout_refuser": expected_cost(y_test, p_test, -0.01),
        "matrice_confusion": {"bons_acceptes": int(tn), "bons_refuses": int(fp),
                              "mauvais_acceptes": int(fn), "mauvais_refuses": int(tp)},
    }
    metrics["gain_vs_tout_accepter"] = 1 - metrics["cout_moyen_modele"] / metrics["cout_moyen_tout_accepter"]
    log(f"Test : AUC {metrics['auc']:.3f} · Gini {metrics['gini']:.3f} · KS {metrics['ks']:.3f} · "
        f"coût {metrics['cout_moyen_modele']:.3f} (tout accepter : {metrics['cout_moyen_tout_accepter']:.3f})")

    # ── 5. Audit d'équité (variables sensibles NON utilisées par le modèle) ──
    audit = test_df[SENSITIVE + ["sexe", "age", TARGET]].copy()
    audit["accorde"] = [decide(p, threshold).code == "accorde" for p in p_test]
    audit["tranche_age"] = np.where(audit["age"] < 25, "moins de 25 ans", "25 ans et plus")
    fairness = {}
    for col in ["sexe", "tranche_age", "travailleur_etranger"]:
        g = audit.groupby(col).agg(dossiers=("accorde", "size"), taux_accord=("accorde", "mean"),
                                   taux_defaut_reel=(TARGET, "mean"))
        rates = g["taux_accord"]
        fairness[col] = {
            "groupes": g.round(3).reset_index().to_dict(orient="records"),
            "ratio_impact_disparate": float(rates.min() / rates.max()) if rates.max() > 0 else None,
        }

    # ── 6. Importance des variables (permutation, sur le test) ──────
    perm = permutation_importance(final, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=SEED)
    importance = (pd.DataFrame({"variable": FEATURES, "importance": perm.importances_mean, "ecart_type": perm.importances_std})
                  .sort_values("importance", ascending=False).reset_index(drop=True))

    # ── 7. Figures ───────────────────────────────────────────────────
    make_figures(y_train, oof, chosen, cost_curve, threshold, y_test, p_test, importance)

    # ── 8. Sauvegarde ────────────────────────────────────────────────
    reference = reference_profile(X_train)
    bundle = {
        "pipeline": final,
        "modele": chosen,
        "seuil": threshold,
        "reference": reference,
        "features": FEATURES,
        "version": date.today().isoformat(),
        "sklearn": sklearn.__version__,
        "metrics": metrics,
    }
    joblib.dump(bundle, MODELS_DIR / "model.joblib")

    report = {
        "date": date.today().isoformat(),
        "donnees": {"dossiers": len(df), "apprentissage": len(train_df), "test": len(test_df),
                    "taux_defaut": float(df[TARGET].mean())},
        "comparaison": comp.round(4).to_dict(orient="records"),
        "modele_retenu": chosen,
        "raison": reason,
        "test": metrics,
        "equite": fairness,
        "importance": importance.head(10).round(4).to_dict(orient="records"),
        "exemples_scores": {f"{p:.0%}": proba_to_score(p) for p in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]},
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_model_card(report)
    log(f"Modèle enregistré : {MODELS_DIR / 'model.joblib'}")
    return report


def make_figures(y_train, oof, chosen, cost_curve, threshold, y_test, p_test, importance):
    # Courbes ROC (validation croisée)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p in oof.items():
        fpr, tpr, _ = roc_curve(y_train, p)
        ax.plot(fpr, tpr, color=COLORS[name], lw=2.5 if name == chosen else 1.5,
                label=f"{name} (AUC {roc_auc_score(y_train, p):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Hasard")
    ax.set_xlabel("Taux de faux positifs (bons payeurs refusés)")
    ax.set_ylabel("Taux de vrais positifs (mauvais payeurs détectés)")
    ax.set_title("Courbes ROC — validation croisée")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "roc.png"); plt.close(fig)

    # Coût selon le seuil
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cost_curve["seuil"], cost_curve["cout"], color="#2e86ab", lw=2)
    ax.axvline(threshold, color="#d1495b", ls="--")
    ax.annotate(f"seuil optimal = {threshold:.2f}", (threshold, cost_curve["cout"].min()),
                xytext=(12, 25), textcoords="offset points", color="#d1495b")
    ax.axhline(COST_FALSE_ACCEPT * y_train.mean(), color="grey", ls=":", lw=1)
    ax.text(0.95, COST_FALSE_ACCEPT * y_train.mean(), "tout accepter", ha="right", va="bottom", color="grey", fontsize=8)
    ax.set_xlabel("Seuil de probabilité de défaut au-delà duquel on refuse")
    ax.set_ylabel("Coût moyen par dossier")
    ax.set_title(f"Choix du seuil : refuser un bon payeur coûte {COST_FALSE_REJECT}, accepter un mauvais coûte {COST_FALSE_ACCEPT}")
    fig.tight_layout(); fig.savefig(FIG_DIR / "cout_seuil.png"); plt.close(fig)

    # Distribution des scores (test)
    scores = np.array([proba_to_score(p) for p in p_test])
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(300, 925, 25)
    ax.hist(scores[np.asarray(y_test) == 0], bins=bins, alpha=0.7, color="#3d9970", label="Bons payeurs")
    ax.hist(scores[np.asarray(y_test) == 1], bins=bins, alpha=0.7, color="#d1495b", label="Mauvais payeurs")
    ax.axvline(proba_to_score(threshold), color="black", ls="--", lw=1)
    ax.text(proba_to_score(threshold) + 5, ax.get_ylim()[1] * 0.92, "accord au-dessus", fontsize=8)
    ax.set_xlim(450, 900)
    ax.set_xlabel("Score (points)"); ax.set_ylabel("Nombre de dossiers")
    ax.set_title("Les mauvais payeurs ont des scores plus bas (jeu de test)")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "distribution_scores.png"); plt.close(fig)

    # Calibration
    frac, mean_pred = calibration_curve(y_test, p_test, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Calibration parfaite")
    ax.plot(mean_pred, frac, "o-", color="#2e86ab", lw=2, label=chosen)
    ax.set_xlabel("Probabilité de défaut prédite"); ax.set_ylabel("Taux de défaut observé")
    ax.set_title("Calibration — les probabilités sont-elles fiables ?")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "calibration.png"); plt.close(fig)

    # Importance des variables
    top = importance.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top["variable"], top["importance"], xerr=top["ecart_type"], color="#2e86ab", capsize=3)
    ax.set_xlabel("Baisse de l'AUC quand on mélange la variable")
    ax.set_title("Importance des variables (permutation, jeu de test)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "importance.png"); plt.close(fig)


def write_model_card(r: dict) -> None:
    t = r["test"]
    lines = [
        "# Fiche du modèle (model card)",
        "",
        f"*Générée automatiquement par `scoring.train` le {r['date']}.*",
        "",
        "## Usage prévu",
        "Estimer la probabilité qu'un demandeur de crédit à la consommation ne rembourse pas (défaut),",
        "pour **aider** un conseiller à décider. Projet pédagogique : le modèle n'est pas destiné à une utilisation réelle.",
        "",
        "## Données",
        f"- German Credit (UCI Statlog), {r['donnees']['dossiers']} dossiers, {r['donnees']['taux_defaut']:.0%} de défauts.",
        f"- Apprentissage : {r['donnees']['apprentissage']} · test : {r['donnees']['test']} (séparation stratifiée).",
        "- Variables **exclues** du modèle : sexe / situation familiale et statut de travailleur étranger.",
        "",
        "## Modèles comparés (validation croisée à 5 plis)",
        "| Modèle | AUC moyenne | Écart-type |",
        "|---|---|---|",
        *[f"| {c['modele']} | {c['auc_cv_moyenne']:.3f} | {c['auc_cv_ecart_type']:.3f} |" for c in r["comparaison"]],
        "",
        f"**Retenu : {r['modele_retenu']}** ({r['raison']}).",
        "",
        "## Performances sur le jeu de test",
        f"- AUC **{t['auc']:.3f}** · Gini **{t['gini']:.3f}** · KS **{t['ks']:.3f}**",
        f"- Seuil de refus : probabilité de défaut ≥ **{t['seuil']:.2f}** (minimise le coût {COST_FALSE_ACCEPT}:{COST_FALSE_REJECT})",
        f"- Mauvais payeurs détectés : **{t['rappel_defauts']:.0%}** · taux d'acceptation : {t['taux_acceptation']:.0%}",
        f"- Coût moyen par dossier : **{t['cout_moyen_modele']:.3f}**, contre {t['cout_moyen_tout_accepter']:.3f} "
        f"en acceptant tout le monde (−{t['gain_vs_tout_accepter']:.0%}).",
        "",
        "## Équité (taux d'accord automatique par groupe)",
    ]
    for col, f in r["equite"].items():
        groups = ", ".join(f"{g[col]} {g['taux_accord']:.0%} (n={g['dossiers']})" for g in f["groupes"])
        ratio = f["ratio_impact_disparate"]
        smallest = min(g["dossiers"] for g in f["groupes"])
        if smallest < 30:
            flag = f"groupe trop petit pour conclure (n={smallest})"
        else:
            flag = "⚠️ sous 0,8" if ratio is not None and ratio < 0.8 else "OK"
        lines.append(f"- **{col}** : {groups} → ratio {ratio:.2f} ({flag})")
    lines += [
        "",
        "Le ratio d'impact disparate compare le taux d'accord du groupe le moins favorisé à celui du plus favorisé ;",
        "sous 0,8 (règle des quatre cinquièmes), l'écart mérite une analyse.",
        "Les moins de 25 ans sont moins souvent acceptés : ils ont aussi un taux de défaut réel plus élevé,",
        "mais l'écart d'accord est plus fort que l'écart de risque. En production, il faudrait l'analyser",
        "(variables liées à l'âge comme l'ancienneté d'emploi) et décider d'une éventuelle correction.",
        "",
        "## Limites",
        "- Données anciennes (Allemagne, années 1970-1990), en Deutsche Marks, et seulement 1 000 dossiers.",
        "- Des variables corrélées à l'âge ou à la situation peuvent encore porter des biais indirects.",
        "- Certaines catégories sont rares : les motifs liés à ces catégories sont moins fiables.",
    ]
    (REPORTS_DIR / "model_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
