import streamlit as st
import pickle
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from PIL import Image
import os
from scipy.ndimage import zoom as ndimage_zoom

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CNN vs ViT — GTSRB",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts & base ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default chrome ──────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* ── Sidebar ──────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.85rem;
    padding: 0.35rem 0;
    letter-spacing: 0.01em;
}

/* ── Page background ──────────────────────────────────────── */
.stApp { background: #0d1117; color: #e6edf3; }

/* ── Typography ───────────────────────────────────────────── */
h1 { font-size: 1.6rem !important; font-weight: 600 !important;
     color: #e6edf3 !important; letter-spacing: -0.02em; margin-bottom: 0.25rem !important; }
h2 { font-size: 1.1rem !important; font-weight: 500 !important;
     color: #c9d1d9 !important; letter-spacing: -0.01em; }
h3 { font-size: 0.95rem !important; font-weight: 500 !important; color: #8b949e !important; }
p, li, .stMarkdown { color: #c9d1d9; font-size: 0.88rem; line-height: 1.6; }

/* ── Metric cards ─────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1rem 1.25rem;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.78rem !important;
    text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 1.6rem !important;
    font-weight: 600 !important; font-family: 'JetBrains Mono', monospace; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── Tables ───────────────────────────────────────────────── */
table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
th { background: #161b22; color: #8b949e; font-weight: 500;
     text-transform: uppercase; letter-spacing: 0.05em;
     padding: 0.5rem 0.75rem; border-bottom: 1px solid #21262d; }
td { padding: 0.45rem 0.75rem; border-bottom: 1px solid #21262d; color: #c9d1d9; }
tr:last-child td { border-bottom: none; }

/* ── Dividers ─────────────────────────────────────────────── */
hr { border: none; border-top: 1px solid #21262d; margin: 1.5rem 0; }

/* ── Tabs ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid #21262d; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #8b949e;
    font-size: 0.83rem; font-weight: 500; letter-spacing: 0.01em;
    padding: 0.5rem 1.25rem; border: none; border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff !important; }

/* ── Alerts ───────────────────────────────────────────────── */
.stAlert { border-radius: 6px; font-size: 0.83rem; }

/* ── Buttons / radio ──────────────────────────────────────── */
.stRadio label { font-size: 0.85rem !important; color: #c9d1d9 !important; }

/* ── Slider ───────────────────────────────────────────────── */
.stSlider { padding: 0.25rem 0; }

/* ── File uploader ────────────────────────────────────────── */
[data-testid="stFileUploader"] { border: 1px dashed #30363d; border-radius: 8px; padding: 0.5rem; }

/* ── Spinner ──────────────────────────────────────────────── */
.stSpinner { color: #58a6ff !important; }

/* ── Caption / small label ────────────────────────────────── */
.caption { font-size: 0.75rem; color: #8b949e; margin-top: 0.25rem; }

/* ── Section header ───────────────────────────────────────── */
.section-label {
    font-size: 0.7rem; font-weight: 500; letter-spacing: 0.1em;
    text-transform: uppercase; color: #58a6ff;
    margin-bottom: 0.5rem; display: block;
}

/* ── Model card ───────────────────────────────────────────── */
.model-card {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 10px; padding: 1.25rem 1.5rem;
}
.model-card h2 { margin-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib theme ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#0d1117",
    "axes.edgecolor":    "#21262d",
    "axes.labelcolor":   "#8b949e",
    "axes.titlecolor":   "#c9d1d9",
    "axes.titlesize":    10,
    "axes.labelsize":    8,
    "axes.grid":         True,
    "grid.color":        "#21262d",
    "grid.linewidth":    0.6,
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#21262d",
    "legend.labelcolor": "#c9d1d9",
    "legend.fontsize":   8,
    "text.color":        "#c9d1d9",
    "lines.linewidth":   1.5,
})

DEVICE      = torch.device("cpu")
NUM_CLASSES = 43
C_CNN = "#58a6ff"
C_VIT = "#f78166"

# ── Data & labels ─────────────────────────────────────────────────────────────
@st.cache_data
def load_label_names():
    names = []
    with open("label_names.csv", "r") as f:
        for row in csv.reader(f):
            names.append(row[1])
    return names[1:]

@st.cache_data
def load_data():
    with open("data3.pickle", "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data["x_test"].astype(np.float32), data["y_test"].astype(np.int64)

@st.cache_data
def load_norm_params():
    with open("mean_image_rgb.pickle", "rb") as f:
        mean_rgb = pickle.load(f, encoding="latin1")["mean_image_rgb"]
    with open("std_rgb.pickle", "rb") as f:
        std_rgb = pickle.load(f, encoding="latin1")["std_rgb"]
    return mean_rgb, std_rgb

def denormalize(img, mean_rgb, std_rgb):
    img = img * std_rgb + mean_rgb
    return np.clip(img * 255.0, 0, 255).astype(np.uint8).transpose(1, 2, 0)

# ── Model architectures ───────────────────────────────────────────────────────
class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_drop=0.0, proj_drop=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_drop, batch_first=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_weights=False):
        x_norm = self.norm(x)
        out, w = self.attn(x_norm, x_norm, x_norm)
        out = x + self.proj_drop(out)
        return (out, w) if return_weights else out

class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        h = int(embed_dim * mlp_ratio)
        self.norm  = nn.LayerNorm(embed_dim)
        self.fc1   = nn.Linear(embed_dim, h)
        self.act   = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2   = nn.Linear(h, embed_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop2(self.fc2(self.drop1(self.act(self.fc1(self.norm(x))))))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, attn_drop=0.0, proj_drop=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_drop, proj_drop)
        self.mlp  = MLP(embed_dim, mlp_ratio, proj_drop)

    def forward(self, x, return_weights=False):
        if return_weights:
            x, w = self.attn(x, return_weights=True)
            return self.mlp(x), w
        return self.mlp(self.attn(x))

class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=43,
                 embed_dim=128, depth=6, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, proj_drop=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, return_attn=False):
        B = x.size(0)
        x = self.pos_drop(torch.cat([self.cls_token.expand(B, -1, -1), self.patch_embed(x)], 1) + self.pos_embed)
        attn_list = []
        for i, blk in enumerate(self.blocks):
            if return_attn and i == len(self.blocks) - 1:
                x, w = blk(x, return_weights=True)
                attn_list.append(w)
            else:
                x = blk(x)
        logits = self.head(self.norm(x)[:, 0])
        return (logits, attn_list) if return_attn else logits

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_cnn():
    m = TrafficSignCNN(NUM_CLASSES).to(DEVICE)
    if os.path.exists("best_model.pth"):
        m.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    return m.eval()

def _infer_vit_config(sd):
    proj_w    = sd["patch_embed.proj.weight"]        # [embed_dim, 3, patch_size, patch_size]
    embed_dim = proj_w.shape[0]
    patch_size = proj_w.shape[2]
    num_patches = sd["pos_embed"].shape[1] - 1       # pos_embed: [1, n_patches+1, embed_dim]
    img_size   = int(math.sqrt(num_patches)) * patch_size
    depth      = sum(1 for k in sd if k.startswith("blocks.") and k.endswith(".mlp.fc1.weight"))
    mlp_ratio  = sd["blocks.0.mlp.fc1.weight"].shape[0] // embed_dim
    num_heads  = max(1, embed_dim // 16)             # heuristic: 16 dims per head
    return dict(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
                depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio)

@st.cache_resource
def load_vit():
    if not os.path.exists("best_vit_model.pth"):
        return VisionTransformer(num_classes=NUM_CLASSES).to(DEVICE).eval()
    sd  = torch.load("best_vit_model.pth", map_location=DEVICE)
    cfg = _infer_vit_config(sd)
    m   = VisionTransformer(num_classes=NUM_CLASSES, **cfg).to(DEVICE)
    m.load_state_dict(sd)
    return m.eval()

@st.cache_data
def run_test_inference(_cnn, _vit, _x_test, _y_test):
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(torch.from_numpy(_x_test), torch.from_numpy(_y_test)),
                        batch_size=256, shuffle=False)
    cp, vp, lb = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            cp.extend(_cnn(xb).argmax(1).cpu().numpy())
            vp.extend(_vit(xb).argmax(1).cpu().numpy())
            lb.extend(yb.numpy())
    return np.array(cp), np.array(vp), np.array(lb)

def preprocess_uploaded(img, mean_rgb, std_rgb):
    arr = np.array(img.convert("RGB").resize((32, 32))).astype(np.float32) / 255.0
    return (arr.transpose(2, 0, 1) - mean_rgb) / (std_rgb + 1e-8)

def get_attention_map(vit, img_tensor):
    with torch.no_grad():
        _, attn_list = vit(img_tensor.unsqueeze(0).to(DEVICE), return_attn=True)
    cls_attn = attn_list[-1][0][0, 1:].cpu().numpy()
    n = int(math.sqrt(cls_attn.shape[0]))
    return cls_attn.reshape(n, n)

models_available = os.path.exists("best_model.pth") and os.path.exists("best_vit_model.pth")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<span class='section-label'>Navigation</span>", unsafe_allow_html=True)
    page = st.radio(
        label="",
        options=["Vue d'ensemble", "Courbes d'apprentissage",
                 "Matrices de confusion", "Predictions & Attention", "Tester une image"],
        label_visibility="collapsed",
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.72rem;color:#484f58;'>GTSRB · 43 classes · 32×32</p>",
                unsafe_allow_html=True)

label_names = load_label_names()

# ── PAGE : Vue d'ensemble ─────────────────────────────────────────────────────
if page == "Vue d'ensemble":
    st.markdown("<span class='section-label'>Classification de panneaux routiers</span>", unsafe_allow_html=True)
    st.title("CNN vs Vision Transformer")
    st.markdown(
        "<p style='color:#8b949e;margin-top:-0.5rem;margin-bottom:1.5rem;'>"
        "GTSRB &mdash; 43 classes &middot; images 32&times;32 &middot; "
        "86 989 train &middot; 4 410 val &middot; 12 630 test</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("""
        <div class='model-card'>
        <h2>CNN</h2>
        <table>
          <tr><th>Parametre</th><th>Valeur</th></tr>
          <tr><td>Architecture</td><td>3 blocs Conv · BN · ReLU · MaxPool</td></tr>
          <tr><td>Parametres</td><td>~530 K</td></tr>
          <tr><td>Optimiseur</td><td>Adam — lr 1e-3 · wd 1e-4</td></tr>
          <tr><td>Scheduler</td><td>ReduceLROnPlateau</td></tr>
          <tr><td>Epochs</td><td>30 · batch 128</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='model-card'>
        <h2>Vision Transformer</h2>
        <table>
          <tr><th>Parametre</th><th>Valeur</th></tr>
          <tr><td>Architecture</td><td>Patch 4×4 &rarr; 6 Transformer blocks</td></tr>
          <tr><td>Parametres</td><td>~1.21 M</td></tr>
          <tr><td>Optimiseur</td><td>AdamW — lr 3e-4 · wd 0.05</td></tr>
          <tr><td>Scheduler</td><td>Warmup 5 ep + Cosine decay</td></tr>
          <tr><td>Epochs</td><td>30 · batch 128</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    if models_available:
        with st.spinner("Evaluation sur le test set…"):
            cnn_m = load_cnn()
            vit_m = load_vit()
            x_test, y_test = load_data()
            cnn_preds, vit_preds, labels = run_test_inference(cnn_m, vit_m, x_test, y_test)

        cnn_acc = accuracy_score(labels, cnn_preds)
        vit_acc = accuracy_score(labels, vit_preds)
        delta   = (vit_acc - cnn_acc) * 100

        m1, m2, m3 = st.columns(3)
        m1.metric("CNN — Accuracy", f"{cnn_acc*100:.2f}%")
        m2.metric("ViT — Accuracy", f"{vit_acc*100:.2f}%")
        m3.metric("Delta (ViT − CNN)", f"{delta:+.2f}%")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<span class='section-label'>Recall par classe</span>", unsafe_allow_html=True)

        cnn_cm  = confusion_matrix(labels, cnn_preds)
        vit_cm  = confusion_matrix(labels, vit_preds)
        cnn_rec = cnn_cm.diagonal() / cnn_cm.sum(axis=1)
        vit_rec = vit_cm.diagonal() / vit_cm.sum(axis=1)

        fig, ax = plt.subplots(figsize=(13, 3.2))
        x = np.arange(NUM_CLASSES)
        w = 0.38
        ax.bar(x - w/2, cnn_rec, w, label="CNN", color=C_CNN, alpha=0.85)
        ax.bar(x + w/2, vit_rec, w, label="ViT", color=C_VIT, alpha=0.85)
        ax.set_xlabel("Classe")
        ax.set_ylabel("Recall")
        ax.set_xticks(x)
        ax.set_xticklabels(x, fontsize=6.5)
        ax.set_ylim(0, 1.05)
        ax.legend(framealpha=0.8)
        fig.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.warning("Fichiers de poids introuvables. Lancez d'abord les notebooks d'entrainement.")

# ── PAGE : Courbes d'apprentissage ────────────────────────────────────────────
elif page == "Courbes d'apprentissage":
    st.markdown("<span class='section-label'>Entrainement</span>", unsafe_allow_html=True)
    st.title("Courbes d'apprentissage")

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("<h2>CNN</h2>", unsafe_allow_html=True)
        if os.path.exists("training_curves.png"):
            st.image("training_curves.png", use_container_width=True)
        else:
            st.warning("training_curves.png introuvable.")

    with col2:
        st.markdown("<h2>Vision Transformer</h2>", unsafe_allow_html=True)
        if os.path.exists("vit_training_curves.png"):
            st.image("vit_training_curves.png", use_container_width=True)
        else:
            st.warning("vit_training_curves.png introuvable.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <ul style='font-size:0.83rem;color:#8b949e;line-height:2;'>
      <li>Le CNN converge rapidement grace a ses biais inductifs (localite, invariance par translation).</li>
      <li>Le ViT utilise un warmup lineaire sur 5 epochs puis un cosine decay pour stabiliser l'entrainement.</li>
      <li>Le label smoothing (0.1) applique au ViT rend les valeurs de loss non comparables directement avec le CNN.</li>
    </ul>
    """, unsafe_allow_html=True)

# ── PAGE : Matrices de confusion ──────────────────────────────────────────────
elif page == "Matrices de confusion":
    st.markdown("<span class='section-label'>Evaluation</span>", unsafe_allow_html=True)
    st.title("Matrices de confusion")

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("<h2>CNN</h2>", unsafe_allow_html=True)
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_container_width=True)
        else:
            st.warning("confusion_matrix.png introuvable.")

    with col2:
        st.markdown("<h2>Vision Transformer</h2>", unsafe_allow_html=True)
        if os.path.exists("vit_confusion_matrix.png"):
            st.image("vit_confusion_matrix.png", use_container_width=True)
        else:
            st.warning("vit_confusion_matrix.png introuvable.")

    if models_available:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<span class='section-label'>Analyse par classe</span>", unsafe_allow_html=True)

        with st.spinner("Calcul des metriques…"):
            cnn_m = load_cnn()
            vit_m = load_vit()
            x_test, y_test = load_data()
            cnn_preds, vit_preds, labels = run_test_inference(cnn_m, vit_m, x_test, y_test)

        cnn_cm  = confusion_matrix(labels, cnn_preds)
        vit_cm  = confusion_matrix(labels, vit_preds)
        cnn_rec = cnn_cm.diagonal() / cnn_cm.sum(axis=1)
        vit_rec = vit_cm.diagonal() / vit_cm.sum(axis=1)

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown("<h2>CNN — classes les plus difficiles</h2>", unsafe_allow_html=True)
            rows = "".join(
                f"<tr><td style='font-family:JetBrains Mono,monospace;'>{c:02d}</td>"
                f"<td>{label_names[c]}</td>"
                f"<td style='color:#f78166;text-align:right;font-family:JetBrains Mono,monospace;'>{cnn_rec[c]:.3f}</td></tr>"
                for c in np.argsort(cnn_rec)[:5]
            )
            st.markdown(f"<table><tr><th>#</th><th>Classe</th><th>Recall</th></tr>{rows}</table>",
                        unsafe_allow_html=True)

        with c2:
            st.markdown("<h2>ViT — classes les plus difficiles</h2>", unsafe_allow_html=True)
            rows = "".join(
                f"<tr><td style='font-family:JetBrains Mono,monospace;'>{c:02d}</td>"
                f"<td>{label_names[c]}</td>"
                f"<td style='color:#f78166;text-align:right;font-family:JetBrains Mono,monospace;'>{vit_rec[c]:.3f}</td></tr>"
                for c in np.argsort(vit_rec)[:5]
            )
            st.markdown(f"<table><tr><th>#</th><th>Classe</th><th>Recall</th></tr>{rows}</table>",
                        unsafe_allow_html=True)

# ── PAGE : Predictions & Attention ────────────────────────────────────────────
elif page == "Predictions & Attention":
    st.markdown("<span class='section-label'>Visualisation</span>", unsafe_allow_html=True)
    st.title("Predictions & Cartes d'attention")

    tab1, tab2 = st.tabs(["CNN — Predictions", "ViT — Attention"])

    with tab1:
        if os.path.exists("predictions.png"):
            st.image("predictions.png", use_container_width=True)
            st.markdown("<p class='caption'>Vert = prediction correcte &nbsp;&middot;&nbsp; Rouge = erreur</p>",
                        unsafe_allow_html=True)
        else:
            st.warning("predictions.png introuvable.")

    with tab2:
        if os.path.exists("vit_attention_maps.png"):
            st.image("vit_attention_maps.png", use_container_width=True)
        else:
            st.warning("vit_attention_maps.png introuvable.")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <p style='font-size:0.83rem;color:#8b949e;'>
        Chaque ligne presente une image. De gauche a droite&nbsp;: image originale,
        carte d'attention du CLS token vers les 64 patches, superposition.
        Les zones claires indiquent les regions ayant le plus influence la decision du modele.
        </p>
        """, unsafe_allow_html=True)

# ── PAGE : Tester une image ───────────────────────────────────────────────────
elif page == "Tester une image":
    st.markdown("<span class='section-label'>Inference</span>", unsafe_allow_html=True)
    st.title("Tester une image")

    mean_rgb, std_rgb = load_norm_params()
    cnn_model = load_cnn()
    vit_model = load_vit()

    mode = st.radio(
        "Source",
        ["Image du test set", "Image importee"],
        horizontal=True,
        label_visibility="collapsed",
    )

    img_tensor  = None
    img_display = None
    true_label  = None

    if mode == "Image du test set":
        x_test, y_test = load_data()
        idx = st.slider("Index", 0, len(x_test) - 1, 0, label_visibility="collapsed")
        img_tensor  = torch.from_numpy(x_test[idx])
        img_display = denormalize(x_test[idx], mean_rgb, std_rgb)
        true_label  = int(y_test[idx])
    else:
        uploaded = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        if uploaded:
            pil_img     = Image.open(uploaded)
            arr         = preprocess_uploaded(pil_img, mean_rgb, std_rgb)
            img_tensor  = torch.from_numpy(arr)
            img_display = np.array(pil_img.convert("RGB").resize((32, 32)))

    if img_tensor is not None:
        st.markdown("<hr>", unsafe_allow_html=True)

        with torch.no_grad():
            x           = img_tensor.unsqueeze(0).to(DEVICE)
            cnn_probs   = F.softmax(cnn_model(x)[0], dim=0).cpu().numpy()
            vit_probs   = F.softmax(vit_model(x)[0], dim=0).cpu().numpy()

        cnn_top = np.argsort(cnn_probs)[::-1][:5]
        vit_top = np.argsort(vit_probs)[::-1][:5]

        col_img, col_cnn, col_vit = st.columns([1, 2, 2], gap="medium")

        with col_img:
            st.markdown("<span class='section-label'>Image</span>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(2.5, 2.5))
            fig.patch.set_facecolor("#0d1117")
            ax.imshow(img_display)
            ax.axis("off")
            fig.tight_layout(pad=0)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            if true_label is not None:
                st.markdown(
                    f"<p class='caption'>Vraie classe&nbsp;: "
                    f"<span style='color:#e6edf3;font-family:JetBrains Mono,monospace;'>"
                    f"{true_label:02d}</span> {label_names[true_label]}</p>",
                    unsafe_allow_html=True,
                )

        with col_cnn:
            st.markdown("<span class='section-label'>CNN</span>", unsafe_allow_html=True)
            pred_cls = cnn_top[0]
            correct  = true_label is not None and pred_cls == true_label
            color    = "#3fb950" if correct else "#e6edf3"
            st.markdown(
                f"<p style='font-size:1rem;font-weight:600;color:{color};margin-bottom:0.75rem;'>"
                f"{label_names[pred_cls]}"
                f"<span style='font-size:0.8rem;font-weight:400;color:#8b949e;margin-left:0.5rem;'>"
                f"{cnn_probs[pred_cls]*100:.1f}%</span></p>",
                unsafe_allow_html=True,
            )
            rows = "".join(
                f"<tr><td style='font-family:JetBrains Mono,monospace;color:#8b949e;'>{c:02d}</td>"
                f"<td>{label_names[c]}</td>"
                f"<td style='text-align:right;font-family:JetBrains Mono,monospace;color:{C_CNN};'>"
                f"{cnn_probs[c]*100:.1f}%</td></tr>"
                for c in cnn_top
            )
            st.markdown(f"<table><tr><th>#</th><th>Classe</th><th>Proba</th></tr>{rows}</table>",
                        unsafe_allow_html=True)

        with col_vit:
            st.markdown("<span class='section-label'>Vision Transformer</span>", unsafe_allow_html=True)
            pred_cls = vit_top[0]
            correct  = true_label is not None and pred_cls == true_label
            color    = "#3fb950" if correct else "#e6edf3"
            st.markdown(
                f"<p style='font-size:1rem;font-weight:600;color:{color};margin-bottom:0.75rem;'>"
                f"{label_names[pred_cls]}"
                f"<span style='font-size:0.8rem;font-weight:400;color:#8b949e;margin-left:0.5rem;'>"
                f"{vit_probs[pred_cls]*100:.1f}%</span></p>",
                unsafe_allow_html=True,
            )
            rows = "".join(
                f"<tr><td style='font-family:JetBrains Mono,monospace;color:#8b949e;'>{c:02d}</td>"
                f"<td>{label_names[c]}</td>"
                f"<td style='text-align:right;font-family:JetBrains Mono,monospace;color:{C_VIT};'>"
                f"{vit_probs[c]*100:.1f}%</td></tr>"
                for c in vit_top
            )
            st.markdown(f"<table><tr><th>#</th><th>Classe</th><th>Proba</th></tr>{rows}</table>",
                        unsafe_allow_html=True)

        # ── Attention map ─────────────────────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<span class='section-label'>Carte d'attention — ViT</span>", unsafe_allow_html=True)

        attn_map = get_attention_map(vit_model, img_tensor)
        attn_up  = ndimage_zoom(attn_map, 32 / attn_map.shape[0], order=1)

        fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
        fig.patch.set_facecolor("#0d1117")
        for ax in axes:
            ax.set_facecolor("#0d1117")
        axes[0].imshow(img_display)
        axes[0].set_title("Image originale")
        axes[1].imshow(attn_up, cmap="hot", vmin=0)
        axes[1].set_title("Attention CLS → patches")
        axes[2].imshow(img_display)
        axes[2].imshow(attn_up, cmap="hot", alpha=0.5, vmin=0)
        axes[2].set_title("Superposition")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ── Confidence bar chart ──────────────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<span class='section-label'>Distribution de confiance — Top 10</span>", unsafe_allow_html=True)

        top10 = np.argsort(cnn_probs)[::-1][:10]
        lbls  = [f"{i} · {label_names[i][:20]}" for i in top10]

        fig, ax = plt.subplots(figsize=(11, 3))
        fig.patch.set_facecolor("#0d1117")
        ax.set_facecolor("#0d1117")
        xp = np.arange(10)
        ax.bar(xp - 0.2, cnn_probs[top10] * 100, 0.38, label="CNN", color=C_CNN, alpha=0.85)
        ax.bar(xp + 0.2, vit_probs[top10] * 100, 0.38, label="ViT", color=C_VIT, alpha=0.85)
        ax.set_xticks(xp)
        ax.set_xticklabels(lbls, rotation=28, ha="right", fontsize=7.5)
        ax.set_ylabel("Probabilite (%)")
        ax.legend(framealpha=0.8)
        fig.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
