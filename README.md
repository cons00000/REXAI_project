# REXAI Project

Projet d'analyse de données multimodales (tabulaires, images, textuelles) développé dans le cadre d'un cours de machine learning.

## Structure du projet

Le projet est organisé en trois notebooks thématiques, accompagnés d'un module Python centralisé :

| Fichier | Contenu |
|---|---|
| `projet_REXIA_1.ipynb` | Analyse et modélisation des **données tabulaires** |
| `projet_REXIA_2.ipynb` | Traitement et classification des **données images** |
| `projet_REXIA_3.ipynb` | Analyse et modélisation des **données textuelles** |
| `data_analysis.py` | Fonctions et classes partagées, importées dans les notebooks |

> Les fonctions et classes réutilisables sont centralisées dans `data_analysis.py` afin d'alléger les notebooks et d'éviter la duplication de code.

## Installation

Cloner le dépôt puis installer les dépendances :
```bash
git clone https://github.com/<utilisateur>/REXAI_project.git
cd REXAI_project
pip install -r requirements.txt
```