# Classification de panneaux routiers — CNN vs Vision Transformer

Comparaison de deux architectures de deep learning sur le dataset **GTSRB** (German Traffic Sign Recognition Benchmark) : un CNN convolutionnel et un Vision Transformer (ViT) implémentés from scratch en PyTorch.


## Dataset

GTSRB contient 43 classes de panneaux de signalisation en images 32×32 RGB, réparties en 86 989 exemples d'entraînement, 4 410 de validation et 12 630 de test.

Le dataset n'est pas inclus dans ce dépôt (trop volumineux). Il doit être placé à la racine sous la forme `data3.pickle` avec les clés `x_train`, `x_validation`, `x_test`, `y_train`, `y_validation`, `y_test`.

## Architectures

### CNN

3 blocs Conv2d / BatchNorm / ReLU / MaxPool / Dropout, suivis d'un classificateur fully-connected (~530 K paramètres). Optimiseur Adam avec lr 1e-3 et weight decay 1e-4, scheduler ReduceLROnPlateau, entraîné sur 30 epochs avec des batches de 128.

### Vision Transformer

L'image est découpée en patches 8×8 (16 patches au total), chacun projeté dans un espace d'embedding de dimension 64. La séquence passe ensuite dans 4 blocs Transformer (Multi-Head Self-Attention + MLP). Optimiseur AdamW avec lr 3e-4 et weight decay 0.05, warmup linéaire sur 5 epochs puis cosine decay, 30 epochs, batches de 128.


## Fichiers


cnn_traffic_signs.ipynb       exploration, entraînement et évaluation du CNN
vit_traffic_signs.ipynb       architecture ViT, entraînement et cartes d'attention
transformer_traffic_signs.ipynb
app.py                        interface Streamlit
label_names.csv               noms des 43 classes
mean_image_rgb.pickle         moyenne RGB pour dénormalisation
std_rgb.pickle                écart-type RGB pour dénormalisation
best_model.pth                poids CNN (meilleure val_acc)
best_vit_model.pth            poids ViT (meilleure val_acc)
training_curves.png           courbes loss/accuracy CNN
confusion_matrix.png          matrice de confusion CNN
predictions.png               exemples de prédictions CNN
vit_training_curves.png       courbes loss/accuracy/lr ViT
vit_confusion_matrix.png      matrice de confusion ViT
vit_attention_maps.png        cartes d'attention ViT


---

## Installation

bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS

pip install torch torchvision numpy matplotlib seaborn scikit-learn scipy streamlit pillow


## Lancer l'interface

bash
streamlit run app.py


L'application propose cinq sections : vue d'ensemble avec comparaison des métriques, courbes d'apprentissage, matrices de confusion, exemples de prédictions et cartes d'attention ViT, et une page d'inférence interactive permettant de tester n'importe quelle image.
