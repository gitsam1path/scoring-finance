# Fiche du modèle (model card)

*Générée automatiquement par `scoring.train` le 2026-11-09.*

## Usage prévu
Estimer la probabilité qu'un demandeur de crédit à la consommation ne rembourse pas (défaut),
pour **aider** un conseiller à décider. Projet pédagogique : le modèle n'est pas destiné à une utilisation réelle.

## Données
- German Credit (UCI Statlog), 1000 dossiers, 30% de défauts.
- Apprentissage : 800 · test : 200 (séparation stratifiée).
- Variables **exclues** du modèle : sexe / situation familiale et statut de travailleur étranger.

## Modèles comparés (validation croisée à 5 plis)
| Modèle | AUC moyenne | Écart-type |
|---|---|---|
| Forêt aléatoire | 0.790 | 0.043 |
| Gradient boosting | 0.786 | 0.046 |
| Régression logistique | 0.773 | 0.057 |

**Retenu : Forêt aléatoire** (meilleur AUC).

## Performances sur le jeu de test
- AUC **0.801** · Gini **0.602** · KS **0.510**
- Seuil de refus : probabilité de défaut ≥ **0.24** (minimise le coût 5:1)
- Mauvais payeurs détectés : **87%** · taux d'acceptation : 40%
- Coût moyen par dossier : **0.535**, contre 1.500 en acceptant tout le monde (−64%).

## Équité (taux d'accord automatique par groupe)
- **sexe** : femme 36% (n=61), homme 42% (n=139) → ratio 0.85 (OK)
- **tranche_age** : 25 ans et plus 45% (n=161), moins de 25 ans 20% (n=39) → ratio 0.45 (⚠️ sous 0,8)
- **travailleur_etranger** : non 83% (n=6), oui 39% (n=194) → ratio 0.47 (groupe trop petit pour conclure (n=6))

Le ratio d'impact disparate compare le taux d'accord du groupe le moins favorisé à celui du plus favorisé ;
sous 0,8 (règle des quatre cinquièmes), l'écart mérite une analyse.
Les moins de 25 ans sont moins souvent acceptés : ils ont aussi un taux de défaut réel plus élevé,
mais l'écart d'accord est plus fort que l'écart de risque. En production, il faudrait l'analyser
(variables liées à l'âge comme l'ancienneté d'emploi) et décider d'une éventuelle correction.

## Limites
- Données anciennes (Allemagne, années 1970-1990), en Deutsche Marks, et seulement 1 000 dossiers.
- Des variables corrélées à l'âge ou à la situation peuvent encore porter des biais indirects.
- Certaines catégories sont rares : les motifs liés à ces catégories sont moins fiables.
