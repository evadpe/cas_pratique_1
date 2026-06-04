# Classification de panneaux routiers — CNN vs Vision Transformer

Comparaison de deux architectures de deep learning sur le dataset **GTSRB** (German Traffic Sign Recognition Benchmark) : un CNN convolutionnel et un Vision Transformer (ViT) implémentés from scratch en PyTorch.

---

## Dataset

**GTSRB** — 43 classes de panneaux de signalisation, images 32×32 RGB.

| Split | Exemples |
|---|---|
| Train | 86 989 |
| Validation | 4 410 |
| Test | 12 630 |

Le dataset n'est pas inclus dans ce dépôt (trop volumineux). Il doit être placé à la racine sous la forme `data3.pickle` avec les clés `x_train`, `x_validation`, `x_test`, `y_train`, `y_validation`, `y_test`.

---

## Architectures

### CNN

3 blocs Conv2d → BatchNorm → ReLU → MaxPool → Dropout, suivi d'un classificateur fully-connected.

| Hyperparamètre | Valeur |
|---|---|
| Paramètres | ~530 K |
| Optimiseur | Adam — lr 1e-3, wd 1e-4 |
| Scheduler | ReduceLROnPlateau |
| Epochs | 30, batch 128 |

### Vision Transformer (ViT)

Découpe l'image en patches 8×8, projette chaque patch dans un espace d'embedding, et passe la séquence dans des blocs Transformer (Multi-Head Self-Attention + MLP).

| Hyperparamètre | Valeur |
|---|---|
| Patch size | 8×8 → 16 patches |
| embed_dim | 64 |
| Profondeur | 4 blocs Transformer |
| Paramètres | ~885 K |
| Optimiseur | AdamW — lr 3e-4, wd 0.05 |
| Scheduler | Warmup linéaire (5 ep) + Cosine decay |
| Epochs | 30, batch 128 |

---

## Fichiers

```
.
├── cnn_traffic_signs.ipynb       # Notebook CNN (exploration, entraînement, évaluation)
├── vit_traffic_signs.ipynb       # Notebook ViT (architecture, entraînement, cartes d'attention)
├── transformer_traffic_signs.ipynb
├── app.py                        # Interface Streamlit
├── label_names.csv               # Noms des 43 classes
├── mean_image_rgb.pickle         # Moyenne RGB pour dénormalisation
├── std_rgb.pickle                # Ecart-type RGB pour dénormalisation
├── best_model.pth                # Poids CNN (meilleure val_acc)
├── best_vit_model.pth            # Poids ViT (meilleure val_acc)
├── training_curves.png           # Courbes loss/accuracy CNN
├── confusion_matrix.png          # Matrice de confusion CNN
├── predictions.png               # Exemples de prédictions CNN
├── vit_training_curves.png       # Courbes loss/accuracy/lr ViT
├── vit_confusion_matrix.png      # Matrice de confusion ViT
└── vit_attention_maps.png        # Cartes d'attention ViT
```

---

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install torch torchvision numpy matplotlib seaborn scikit-learn scipy streamlit pillow
```

## Lancer l'interface

```bash
streamlit run app.py
```

L'application propose cinq sections :

- **Vue d'ensemble** — tableau comparatif des deux modèles et recall par classe
- **Courbes d'apprentissage** — loss et accuracy sur train/validation
- **Matrices de confusion** — matrices normalisées et classes les plus difficiles
- **Predictions & Attention** — exemples de prédictions CNN et cartes d'attention ViT
- **Tester une image** — inférence sur une image du test set ou une image importée, avec carte d'attention interactive

---

## Résultats

Les courbes d'entraînement et matrices de confusion sont disponibles dans les fichiers PNG inclus dans le dépôt.

Le ViT charge automatiquement ses hyperparamètres depuis le checkpoint (`best_vit_model.pth`) — aucune modification manuelle nécessaire si les poids sont remplacés.
