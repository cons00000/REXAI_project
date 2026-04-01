import copy
from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import shap
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from scipy.special import expit  
from torch import nn
import torchvision.transforms as T
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from xplique.attributions import Lime, KernelShap

# ---------------------------   DONNEES TABULAIRES ---------------------------

class Analyzer:
    def __init__(self, data, sep=';', encoding='utf-8'):
        """
        data: can be a string (file path) or a pandas DataFrame.
        """
        if isinstance(data, str):
            try:
                self.df = pd.read_csv(data, sep=sep, encoding=encoding)
            except UnicodeDecodeError:
                self.df = pd.read_csv(data, sep=sep, encoding='latin-1')
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            raise ValueError("Data must be a file path (str) or a pandas DataFrame.")
            
        print(f"Dataset ready: {self.df.shape[0]} rows and {self.df.shape[1]} columns.")

    def get_inventory(self):
        """
        Explores types, cardinality, missing values, and ranges for all columns.
        """
        inventory = []
        for col in self.df.columns:
            col_data = self.df[col]
            dtype = col_data.dtype
            nunique = col_data.nunique()
            nulls = col_data.isnull().sum()
            
            if nunique < 10: 
                val_range = f"Unique values: {col_data.unique().tolist()}"
            elif is_numeric_dtype(col_data):
                val_range = f"Range: [{col_data.min()}, {col_data.max()}]"
            else:
                val_range = f"Examples: {col_data.unique()[:1].tolist()}"

            inventory.append({
                "Column": col,
                "Type": dtype,
                "Cardinality": nunique,
                "Missing": nulls,
                "Details": val_range
            })
        
        inventory_df = pd.DataFrame(inventory)
        return inventory_df

    def plot_distributions(self, cols=None, cols_per_row=2):
        all_cols = cols if cols is not None else self.df.columns
        # Skip ID columns
        target_cols = [col for col in all_cols if "id" not in col.lower()]
        
        n_cols = len(target_cols)
        n_rows = (n_cols + cols_per_row - 1) // cols_per_row
        
        plt.figure(figsize=(16, 4 * n_rows)) 

        for i, col in enumerate(target_cols, 1):
            ax = plt.subplot(n_rows, cols_per_row, i)
            
            # 1. Boolean / Low Cardinality
            if self.df[col].dtype == 'bool' or self.df[col].nunique() == 2:
                sns.countplot(x=self.df[col], hue=self.df[col], 
                            palette=["#E74C3C", "#2ECC71"], legend=False)
                plt.title(f"Flag: {col}")
            
            # 2. Continuous Numeric
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                sns.histplot(self.df[col], kde=True, color="#3498DB", edgecolor="white")
                plt.title(f"Numeric: {col}")
            
            # 3. Categorical
            else:
                counts = self.df[col].value_counts().iloc[:10]
                sns.barplot(y=counts.index, x=counts.values, hue=counts.index, 
                            palette="Set2", legend=False)
                plt.title(f"Categorical: {col}")

            plt.xlabel("")

        plt.tight_layout()
        plt.show()

    def target_correlation(self, target_column):
        """
        Calculates and plots correlation of all numeric features against a specific target.
        """
        if target_column not in self.df.columns:
            print(f"Error: {target_column} not found.")
            return

        bool_cols = self.df.select_dtypes(include=[bool]).columns
        numeric_df = self.df.select_dtypes(include=[np.number]).copy()

        for col in bool_cols:
            if col not in numeric_df.columns:
                numeric_df[col] = self.df[col].astype(int)

        if target_column not in numeric_df.columns:
            print(f"Error: {target_column} must be numeric or boolean.")
            return

        correlations = numeric_df.corr()[target_column].sort_values(ascending=False)
        
        # Remove the target's correlation with itself
        correlations = correlations.drop(target_column, errors="ignore")

        if correlations.empty:
            print(f"No numeric correlations available for {target_column}.")
            return correlations

        plt.figure(figsize=(10, 6))
        sns.barplot(x=correlations.values, y=correlations.index, hue=correlations.index, palette="RdBu_r", legend=False)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        plt.title(f"Correlation of Features with '{target_column}'")
        plt.xlabel("Pearson Correlation Coefficient")
        plt.show()

        return correlations
    
    def plot_matrix_correlation(self, threshold: float = 0.3):
        """
        Plots a correlation matrix for all numeric and boolean features,
        filtering out variables with no correlation above the given threshold.
        """
        bool_cols = self.df.select_dtypes(include=[bool]).columns
        numeric_df = self.df.select_dtypes(include=[np.number]).copy()

        for col in bool_cols:
            if col not in numeric_df.columns:
                numeric_df[col] = self.df[col].astype(int)

        corr_matrix = numeric_df.corr()

        mask = corr_matrix.abs() >= threshold
        for i in range(len(mask)):
            mask.iat[i, i] = False

        cols_to_keep = mask.any(axis=1)
        corr_filtered = corr_matrix.loc[cols_to_keep, cols_to_keep]

        if corr_filtered.empty:
            print(f"No numeric/boolean correlations found above the threshold {threshold}.")
            return corr_filtered

        size = max(10, len(corr_filtered) * 0.6)
        plt.figure(figsize=(size, size * 0.85))
        sns.heatmap(corr_filtered, annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title(f"Correlation Matrix (|r| ≥ {threshold})")
        plt.tight_layout()
        plt.show()

class DataPreprocessor:
    def __init__(self, df):
        # We work on a copy to keep the original data intact
        self.df = df.copy()
        self.original_columns = df.columns.tolist()

    def handle_encoding(self):
            """
            Automatically identifies 'object' columns, applies One-Hot Encoding,
            and sanitizes column names to prevent SyntaxErrors.
            """
            # Identify columns to encode
            categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
            
            if not categorical_cols:
                print("No categorical columns (type 'object') found to encode.")
                return self.df

            print(f"Encoding columns: {categorical_cols}")

            # Perform One-Hot Encoding
            self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=False)
            
            # Column Name Sanitization
            self.df.columns = [
                col.replace("'", "")
                .replace(" ", "_")
                .replace("/", "_")
                .replace("(", "")
                .replace(")", "") 
                for col in self.df.columns
            ]
            
            print(f"Encoding complete. New shape: {self.df.shape}")
            return self.df

def _coerce_binary_series(series, column_name):
    """
    Converts binary columns encoded as 0/1, -1/1, bool or numeric strings to 0/1.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    invalid_mask = series.notna() & numeric.isna()
    if invalid_mask.any():
        invalid_values = series[invalid_mask].astype(str).unique()[:5].tolist()
        raise ValueError(
            f"Column '{column_name}' must be binary. Invalid values found: {invalid_values}"
        )

    unique_values = set(pd.unique(numeric.dropna()))
    if not unique_values:
        return numeric.astype(float)
    if unique_values <= {0.0, 1.0}:
        return numeric.astype(float)
    if unique_values <= {-1.0, 1.0}:
        return numeric.map({-1.0: 0.0, 1.0: 1.0}).astype(float)

    preview = sorted(unique_values)[:5]
    raise ValueError(
        f"Column '{column_name}' must be binary and encoded as 0/1 or -1/1. "
        f"Found values like {preview}."
    )

def _positive_rates_by_group(df, target_column, sensitive_column):
    if target_column not in df.columns or sensitive_column not in df.columns:
        raise KeyError(f"Columns '{target_column}' and/or '{sensitive_column}' are missing.")

    prepared = pd.DataFrame({
        "target": _coerce_binary_series(df[target_column], target_column),
        "sensitive": _coerce_binary_series(df[sensitive_column], sensitive_column),
    }).dropna()

    if prepared.empty:
        raise ValueError("No valid rows available after binary coercion.")

    group_1 = prepared.loc[prepared["sensitive"] == 1.0, "target"]
    group_0 = prepared.loc[prepared["sensitive"] == 0.0, "target"]

    if group_1.empty or group_0.empty:
        raise ValueError(
            f"Sensitive column '{sensitive_column}' must contain both groups after coercion."
        )

    return group_1.mean(), group_0.mean()

def get_parcours(df, matricule):
    """
    Filtre et prépare les données pour un matricule.
    """
    data = df[df['matricule'] == matricule].copy()
    
    print(f"Nombre de lignes pour le matricule {matricule} :", len(data))
    
    if data.empty:
        return data
    
    # Conversion des colonnes utiles
    data['Début'] = pd.to_numeric(data['Début de contrat (années)'], errors='coerce')
    data['Ancienneté'] = pd.to_numeric(data['Ancienneté groupe (années)'], errors='coerce')
    data['Niveau hiérarchique'] = pd.to_numeric(data['Niveau hiérarchique'], errors='coerce')
    
    # Reconstruction du temps
    data['temps'] = data['Début'] + data['Ancienneté']
    
    # Trier
    parcours = data.sort_values('temps')
    
    return parcours

def plot_parcours(parcours, title="Parcours pro"):
    if parcours.empty:
        print("Aucune donnée à afficher.")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Ligne principale
    ax.plot(parcours['temps'], parcours['Niveau hiérarchique'], marker='o')
    
    # Détection des changements de poste
    parcours['changement'] = parcours['Famille d\'emploi'].ne(parcours['Famille d\'emploi'].shift())
    
    # Annotations uniquement si changement (évite surcharge)
    for _, row in parcours[parcours['changement']].iterrows():
        ax.annotate(
            str(row['Famille d\'emploi']),
            (row['temps'], row['Niveau hiérarchique']),
            textcoords="offset points",
            xytext=(0,15),
            ha='center',
            fontsize=9,
            fontweight='bold'
        )
    
    # Promotions (taille des points)
    promo = pd.to_numeric(parcours['Dernière promotion (mois)'], errors='coerce').fillna(0)
    ax.scatter(
        parcours['temps'],
        parcours['Niveau hiérarchique'],
        s=promo*3 + 30,
        alpha=0.4
    )
    
    # Mettre en évidence les changements de niveau
    changement_niveau = parcours['Niveau hiérarchique'].diff() != 0
    ax.scatter(
        parcours.loc[changement_niveau, 'temps'],
        parcours.loc[changement_niveau, 'Niveau hiérarchique'],
        marker='D',
        s=80,
        label='Changement de niveau'
    )
    
    # Labels
    ax.set_title(title)
    ax.set_xlabel("Temps (années reconstituées)")
    ax.set_ylabel("Niveau hiérarchique")
    
    ax.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# ---------------------------   DONNEES IMAGES ---------------------------

# Chargement des données
class Celeb_Faces:
    def __init__(self, dataset_path):
        self.path = dataset_path
        self.data = {"attr_df" : None,
                    "partition_df" : None,
                    "bbox_df" : None,
                    "landmarks_df" : None}
        
    def load_data(self):
        """Charge les fichiers CSV principaux."""
        self.data["attr_df"] = pd.read_csv(os.path.join(self.path, 'list_attr_celeba.csv'))
        self.data["partition_df"] = pd.read_csv(os.path.join(self.path, 'list_eval_partition.csv'))
        self.data["bbox_df"] = pd.read_csv(os.path.join(self.path, 'list_bbox_celeba.csv'))
        self.data["landmarks_df"] = pd.read_csv(os.path.join(self.path, 'list_landmarks_align_celeba.csv'))

def resolve_celeba_image_dir(dataset_path):
    """
    Resolves the actual directory containing the CelebA image files.
    """
    dataset_path = Path(dataset_path)
    candidates = [
        dataset_path,
        dataset_path / "img_align_celeba",
        dataset_path / "img_align_celeba" / "img_align_celeba",
    ]

    for candidate in candidates:
        if candidate.is_dir() and next(candidate.glob("*.jpg"), None) is not None:
            return str(candidate)

    raise FileNotFoundError(
        f"Could not find the CelebA image directory from '{dataset_path}'."
    )

# Fonctions présentées dans la consigne
def demographic_parity(df, Y, S):
    p_y1_given_s1, p_y1_given_s0 = _positive_rates_by_group(df, Y, S)
    return p_y1_given_s1 - p_y1_given_s0

def disparate_impact(df,Y,S):
    p_y1_given_s1, p_y1_given_s0 = _positive_rates_by_group(df, Y, S)
    if p_y1_given_s0 != 0:
        return p_y1_given_s1 / p_y1_given_s0
    return np.nan
    
# Visualiser les proportions après groupement de données
def plot_table_attr(df: pd.DataFrame, attrs: list, figsize=None) -> None:
    nr, nc = df.shape
    fig, ax = plt.subplots(figsize=figsize or (nc * 1.4, nr * 0.55 + 1))
    ax.axis("off")
    cmap = mpl.colormaps["YlOrRd"]

    for j, a in enumerate(attrs):
        ax.text(j+1, nr, a.replace("_"," "), ha="center", va="bottom",
                fontsize=9, fontweight="bold", rotation=25)

    for i, p in enumerate(df.index):
        ax.text(0, nr-1-i, str(p), ha="right", va="center",
                fontsize=10, fontweight="bold")
        for j, a in enumerate(attrs):
            v = df.loc[p, a]
            bg = cmap(v / 100)
            fg = "white" if (0.299*bg[0] + 0.587*bg[1] + 0.114*bg[2]) < 0.5 else "#1a1a1a"
            ax.add_patch(mpl.patches.FancyBboxPatch(
                (j+0.52, nr-1-i-0.38), 0.92, 0.76,
                boxstyle="round,pad=0.02", linewidth=0, facecolor=bg))
            ax.text(j+1, nr-1-i, f"{v:.0f}%", ha="center", va="center",
                    fontsize=10, color=fg)

    ax.set(xlim=(-0.3, nc+0.7), ylim=(-0.6, nr+0.8))
    plt.tight_layout()
    plt.show()

# Mettre en évidence des biais
def bias_report(df, attrs=None, threshold=0.7):
    data = (df[attrs] if attrs else df).copy()
    t = threshold * 100

    biased = {
        p: sorted([(a, data.at[p, a]) for a in data.columns if data.at[p, a] > t],
                  key=lambda x: x[1], reverse=True)
        for p in data.index
    }
    biased = dict(sorted(
        {p: v for p, v in biased.items() if len(v) >= 2}.items(),
        key=lambda x: len(x[1]), reverse=True
    ))

    if not biased:
        print(f"Aucun persona ne cumule plusieurs attributs > {threshold:.0%}.")
        return

    print(f"── Personas biaisés (attributs > {threshold:.0%}) ──\n")
    for persona, flagged in biased.items():
        icon = "🔴" if len(flagged) >= 3 else "🟡"
        print(f"{icon} {persona} ({len(flagged)} attributs forts)")
        for attr, val in flagged:
            print(f"   {attr:<22} {val:5.1f}%  {'█' * int(val // 10)}")
        print()

# Structure pour pouvoir charger les images dans un dataloader avant de les embedder via ResNet
class CelebADataset(Dataset):
    def __init__(self, image_ids: pd.Series, labels: pd.Series,
                 path_image: str = None, transform=None):
        self.records = [
            (img_id, label)
            for img_id, label in zip(image_ids, labels)
            if path_image is None or os.path.exists(os.path.join(path_image, img_id))
        ]
        self.path_image = path_image
        self.transform  = transform
        self.features   = None                                 # None = mode image

    def load_features(self, X: np.ndarray):
        """Bascule en mode feature : plus besoin des images."""
        assert len(X) == len(self.records), "Taille incompatible"
        self.features = torch.from_numpy(X).float()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_id, label = self.records[idx]
        label = 1 if int(label) == 1 else 0

        if self.features is not None:                          # mode feature
            return self.features[idx], int(label)

        img = Image.open(os.path.join(self.path_image, img_id)).convert("RGB")
        return self.transform(img), int(label), img_id

def _unpack_batch(batch):
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("Expected a batch shaped as (inputs, labels) or (inputs, labels, ...).")
    return batch[0], batch[1], batch[2:]

@torch.no_grad()
def extract_embeddings(loader, backbone, device):
    all_embs, all_targets = [], []
    for batch in loader:
        tensors, targets, _ = _unpack_batch(batch)
        all_embs.append(backbone(tensors.to(device)).cpu().numpy())
        all_targets.append(targets.numpy())
    return np.vstack(all_embs), np.concatenate(all_targets)

def load_or_compute_embeddings(dataset, backbone, device,
                                cache_path, batch_size=64, num_workers=4):
    current_image_ids = np.array([img_id for img_id, _ in dataset.records], dtype=str)
    current_labels = np.array(
        [1 if int(label) == 1 else 0 for _, label in dataset.records],
        dtype=np.int8,
    )

    if Path(cache_path).exists():
        data = np.load(cache_path, allow_pickle=False)
        if {"X", "y", "image_ids", "labels"}.issubset(data.files):
            if (
                np.array_equal(data["image_ids"], current_image_ids)
                and np.array_equal(data["labels"], current_labels)
            ):
                return data["X"], data["y"]

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers,
                        pin_memory=(device == "cuda"))
    X, y = extract_embeddings(loader, backbone, device)

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X=X,
        y=y,
        image_ids=current_image_ids,
        labels=current_labels,
    )
    return X, y

def run_epoch(head, criterion, optimizer,loader, device, train=True):
    head.train(train)
    total_loss, correct, n = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            features, labels, _ = _unpack_batch(batch)
            features = features.to(device)
            labels   = labels.float().unsqueeze(1).to(device)

            logits = head(features)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(features)
            correct    += ((logits.sigmoid() > 0.5) == labels.bool()).sum().item()
            n          += len(features)

    return total_loss / n, correct / n

class FullModel2Class(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone.eval()
        self.head = head.eval()
        
    def forward(self, x):
        features = self.backbone(x)
        
        logit = self.head(features) 
        
        prob1 = torch.sigmoid(logit)          
        prob0 = 1 - prob1                     
        return torch.cat([prob0, prob1], dim=1)

class FullModelLogits2Class(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone.eval()
        self.head = head.eval()

    def forward(self, x):
        features = self.backbone(x)
        logit = self.head(features)
        return torch.cat([-logit, logit], dim=1)

def fairness_report_dataloader_streaming_imgid(model, dataloader, sensitive_df, threshold=0.5, device="cpu", img_id_col="img_id"):
    """
    Fairness report streaming avec DataLoader qui renvoie img_id.

    model        : FullModel2Class (renvoie [prob0, prob1])
    dataloader   : DataLoader renvoyant (X_batch, y_batch, img_id_batch)
    sensitive_df : DataFrame avec les attributs sensibles + colonne img_id_col
    threshold    : Seuil pour convertir probabilités en labels
    img_id_col   : nom de la colonne contenant les img_id dans sensitive_df
    """
    model.eval()
    model = model.to(device)

    if img_id_col not in sensitive_df.columns:
        raise KeyError(f"Column '{img_id_col}' is missing from sensitive_df.")

    sensitive_cols = [col for col in sensitive_df.columns if col != img_id_col]
    prepared_sensitive = sensitive_df[[img_id_col]].copy()
    for attr in sensitive_cols:
        prepared_sensitive[attr] = _coerce_binary_series(sensitive_df[attr], attr)

    # Création du mapping img_id -> ligne pour accès rapide aux attributs sensibles
    imgid_to_row = prepared_sensitive.set_index(img_id_col).to_dict(orient='index')

    # Initialisation des compteurs pour chaque attribut et valeur
    group_stats = {
        attr: {val: {"TN":0, "FP":0, "FN":0, "TP":0, "N":0} for val in [0,1]}
        for attr in sensitive_cols
    }

    with torch.no_grad():
        for batch in dataloader:
            X_batch, y_batch, extras = _unpack_batch(batch)
            if not extras:
                raise ValueError("DataLoader doit renvoyer (X, y, img_id)")
            img_ids_batch = extras[0]

            X_batch = X_batch.to(device)
            y_batch = y_batch.cpu().numpy()
            img_ids_batch = list(img_ids_batch) if isinstance(img_ids_batch, torch.Tensor) else img_ids_batch

            # Prédictions
            probs = model(X_batch)
            y_pred_batch = (probs[:,1] >= threshold).cpu().numpy().astype(int)

            valid_positions = [
                idx for idx, img_id in enumerate(img_ids_batch)
                if img_id in imgid_to_row
            ]
            if not valid_positions:
                continue

            y_true_valid = y_batch[valid_positions]
            y_pred_valid = y_pred_batch[valid_positions]
            rows_valid = [imgid_to_row[img_ids_batch[idx]] for idx in valid_positions]

            # Mise à jour des compteurs pour chaque attribut sensible
            for attr, stats in group_stats.items():
                attr_vals = np.array([row[attr] for row in rows_valid], dtype=float)
                for val in [0,1]:
                    mask = (attr_vals == float(val))
                    if mask.sum() == 0:
                        continue

                    yt, yp = y_true_valid[mask], y_pred_valid[mask]
                    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()

                    stats[val]["TN"] += tn
                    stats[val]["FP"] += fp
                    stats[val]["FN"] += fn
                    stats[val]["TP"] += tp
                    stats[val]["N"] += mask.sum()

    # Calcul des métriques finales
    rows = []
    for attr, stats in group_stats.items():
        for val in [0,1]:
            m = stats[val]
            o = stats[1-val]

            def safe_div(a, b):
                return a/b if b > 0 else np.nan

            acc = safe_div(m["TP"] + m["TN"], m["N"])
            fpr = safe_div(m["FP"], m["FP"] + m["TN"])
            fnr = safe_div(m["FN"], m["FN"] + m["TP"])

            o_acc = safe_div(o["TP"] + o["TN"], o["N"])
            o_fpr = safe_div(o["FP"], o["FP"] + o["TN"])
            o_fnr = safe_div(o["FN"], o["FN"] + o["TP"])

            rows.append({
                "Group": f"{attr}={val}",
                "N": m["N"],
                "Accuracy": acc,
                "FPR": fpr,
                "FNR": fnr,
                "Δ_Acc": abs(acc - o_acc) if pd.notna(acc) and pd.notna(o_acc) else np.nan,
                "Δ_FPR": abs(fpr - o_fpr) if pd.notna(fpr) and pd.notna(o_fpr) else np.nan,
                "Δ_FNR": abs(fnr - o_fnr) if pd.notna(fnr) and pd.notna(o_fnr) else np.nan,
            })

    return pd.DataFrame(rows).set_index("Group").round(3)

@dataclass
class ImageExplanationContext:
    explain_model: nn.Module
    head_model: nn.Module
    gradcam_model: nn.Module
    cam_extractor: GradCAM
    explain_device: str
    path_image: str
    label_names: list
    display_transform: object
    mean: np.ndarray
    std: np.ndarray
    mean_t: torch.Tensor
    std_t: torch.Tensor
    shap_explainer: object = None

def prepare_image_explanation_context(
    backbone,
    head,
    device,
    path_image,
    label_names=None,
    image_size=224,
):
    """
    Prepares the models and transforms needed for Grad-CAM and SHAP explanations.
    """
    label_names = label_names or ["Not Smiling", "Smiling"]
    path_image = str(path_image)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    display_transform = T.Compose([T.Resize(256), T.CenterCrop(image_size)])

    if device == "mps":
        explain_device = "cpu"
        backbone_explain = copy.deepcopy(backbone).to(explain_device).eval()
        head_explain = copy.deepcopy(head).to(explain_device).eval()
    else:
        explain_device = device
        backbone_explain = backbone
        head_explain = head

    explain_model = FullModel2Class(backbone_explain, head_explain).to(explain_device).eval()
    gradcam_model = FullModelLogits2Class(backbone_explain, head_explain).to(explain_device).eval()
    cam_extractor = GradCAM(model=gradcam_model, target_layers=[gradcam_model.backbone.layer4[-1]])

    return ImageExplanationContext(
        explain_model=explain_model,
        head_model=head_explain,
        gradcam_model=gradcam_model,
        cam_extractor=cam_extractor,
        explain_device=explain_device,
        path_image=path_image,
        label_names=label_names,
        display_transform=display_transform,
        mean=mean,
        std=std,
        mean_t=mean_t,
        std_t=std_t,
    )

def load_explanation_image(context, image_id):
    """
    Loads an image twice: as displayable RGB and as a normalized tensor for the model.
    """
    pil_image = Image.open(os.path.join(context.path_image, image_id)).convert("RGB")
    display_img = np.asarray(context.display_transform(pil_image), dtype=np.float32) / 255.0
    input_tensor = torch.from_numpy(display_img.transpose(2, 0, 1)).unsqueeze(0)
    input_tensor = (input_tensor - context.mean_t) / context.std_t
    return display_img, input_tensor

def predict_proba_from_rgb(context, batch_rgb):
    """
    Applies the smiling classifier on RGB images in [0, 1] with shape (N, H, W, C).
    """
    batch_rgb = np.asarray(batch_rgb, dtype=np.float32)
    batch_tensor = torch.from_numpy(batch_rgb.transpose(0, 3, 1, 2))
    batch_tensor = (batch_tensor - context.mean_t) / context.std_t
    with torch.no_grad():
        probs = context.explain_model(batch_tensor.to(context.explain_device)).cpu().numpy()
    return probs

def collect_feature_predictions(
    context,
    feature_dataset,
    sensitive_df,
    sensitive_attrs,
    batch_size=256,
    threshold=0.5,
    img_id_col="image_id",
):
    """
    Collects predictions quickly from a dataset already loaded with cached embeddings.
    """
    if feature_dataset.features is None:
        raise ValueError("feature_dataset must already contain cached embeddings via load_features().")
    if img_id_col not in sensitive_df.columns:
        raise KeyError(f"Column '{img_id_col}' is missing from sensitive_df.")

    prepared_sensitive = sensitive_df[[img_id_col]].copy()
    for attr in sensitive_attrs:
        prepared_sensitive[attr] = _coerce_binary_series(sensitive_df[attr], attr)
    sensitive_lookup = prepared_sensitive.set_index(img_id_col)

    loader = DataLoader(feature_dataset, batch_size=batch_size, shuffle=False)
    rows = []
    offset = 0

    with torch.no_grad():
        for batch in loader:
            features, labels, _ = _unpack_batch(batch)
            logits = context.head_model(features.to(context.explain_device))
            prob_smiling = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            preds = (prob_smiling >= threshold).astype(int)
            labels_np = labels.numpy()

            img_ids_batch = [
                img_id for img_id, _ in feature_dataset.records[offset:offset + len(labels_np)]
            ]
            offset += len(labels_np)

            for img_id, y_true, y_pred, proba in zip(img_ids_batch, labels_np, preds, prob_smiling):
                row = {
                    img_id_col: img_id,
                    "y_true": int(y_true),
                    "y_pred": int(y_pred),
                    "prob_smiling": float(proba),
                    "correct": bool(int(y_true) == int(y_pred)),
                }
                if img_id in sensitive_lookup.index:
                    for attr in sensitive_attrs:
                        value = sensitive_lookup.at[img_id, attr]
                        if pd.notna(value):
                            row[attr] = int(value)
                rows.append(row)

    return pd.DataFrame(rows)

def prepare_image_explanation_cases(
    context,
    feature_dataset,
    sensitive_df,
    sensitive_attrs,
    batch_size=256,
    threshold=0.5,
    img_id_col="image_id",
):
    """
    Builds a compact explanation payload: predictions, group counts, chosen cases and summary.
    """
    predictions = collect_feature_predictions(
        context=context,
        feature_dataset=feature_dataset,
        sensitive_df=sensitive_df,
        sensitive_attrs=sensitive_attrs,
        batch_size=batch_size,
        threshold=threshold,
        img_id_col=img_id_col,
    )

    if predictions.empty:
        raise ValueError("No predictions available to prepare explanation cases.")

    group_counts = pd.concat([
        predictions[attr]
        .value_counts()
        .rename_axis("value")
        .reset_index(name="count")
        .assign(attribute=attr)
        for attr in sensitive_attrs
    ], ignore_index=True).sort_values(["count", "attribute", "value"]).reset_index(drop=True)

    if group_counts.empty:
        raise ValueError("No sensitive-group counts available for explanation case selection.")

    minority_attr = group_counts.loc[0, "attribute"]
    minority_value = int(group_counts.loc[0, "value"])
    minority_count = int(group_counts.loc[0, "count"])

    correct_candidates = predictions[predictions["correct"]]
    if correct_candidates.empty:
        raise ValueError("No correct prediction available for explanation.")
    correct_case = correct_candidates.iloc[0]

    used_ids = {correct_case[img_id_col]}
    incorrect_candidates = predictions[
        (~predictions["correct"]) & (~predictions[img_id_col].isin(used_ids))
    ]
    if incorrect_candidates.empty:
        incorrect_candidates = predictions[~predictions["correct"]]
    if incorrect_candidates.empty:
        raise ValueError("No incorrect prediction available for explanation.")
    incorrect_case = incorrect_candidates.iloc[0]

    used_ids.add(incorrect_case[img_id_col])
    minority_pool = predictions[predictions[minority_attr] == minority_value]
    if minority_pool.empty:
        raise ValueError(f"No sample found for minority group {minority_attr}={minority_value}.")
    minority_candidates = minority_pool[~minority_pool[img_id_col].isin(used_ids)]
    if minority_candidates.empty:
        minority_candidates = minority_pool
    minority_case = minority_candidates.iloc[0]

    cases = {
        "correct": correct_case,
        "incorrect": incorrect_case,
        "minority": minority_case,
    }

    case_summary = pd.DataFrame({
        name: {
            "image_id": row[img_id_col],
            "y_true": row["y_true"],
            "y_pred": row["y_pred"],
            "prob_smiling": round(row["prob_smiling"], 3),
            "correct": row["correct"],
        }
        for name, row in cases.items()
    }).T
    case_summary.loc["minority", "minority_group"] = (
        f"{minority_attr}={minority_value} (n={minority_count})"
    )

    return {
        "predictions": predictions,
        "group_counts": group_counts,
        "minority_attr": minority_attr,
        "minority_value": minority_value,
        "minority_count": minority_count,
        "cases": cases,
        "case_summary": case_summary,
    }

def show_gradcam_explanations(context, cases, target_class=1):
    """
    Displays Grad-CAM visualizations for a dictionary of named cases.
    """
    for case_name, row in cases.items():
        image_id = row["image_id"]
        display_img, input_tensor = load_explanation_image(context, image_id)
        grayscale_cam = context.cam_extractor(
            input_tensor=input_tensor.to(context.explain_device),
            targets=[ClassifierOutputTarget(target_class)],
        )[0]
        cam_overlay = show_cam_on_image(display_img, grayscale_cam, use_rgb=True)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(display_img)
        axes[0].set_title(
            f"{case_name}\ntrue={row['y_true']} | pred={row['y_pred']} | p={row['prob_smiling']:.3f}"
        )
        axes[1].imshow(cam_overlay)
        axes[1].set_title(f"Grad-CAM sur la classe {context.label_names[target_class]}")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

def _get_shap_explainer(context, blur="blur(32,32)"):
    if context.shap_explainer is None:
        context.shap_explainer = shap.Explainer(
            lambda batch_rgb: predict_proba_from_rgb(context, batch_rgb),
            shap.maskers.Image(blur, (224, 224, 3)),
            output_names=context.label_names,
        )
    return context.shap_explainer

def show_shap_explanations(
    context,
    cases,
    max_evals=300,
    batch_size=16,
    output_index=1,
    blur="blur(32,32)",
):
    """
    Displays SHAP image explanations for a dictionary of named cases.
    """
    shap_explainer = _get_shap_explainer(context, blur=blur)

    for case_name, row in cases.items():
        image_id = row["image_id"]
        display_img, _ = load_explanation_image(context, image_id)
        batch_rgb = display_img[None, ...]
        shap_values = shap_explainer(
            batch_rgb,
            outputs=[output_index],
            max_evals=max_evals,
            batch_size=batch_size,
        )
        print(
            f"{case_name}: {image_id} | true={row['y_true']} | pred={row['y_pred']} | p={row['prob_smiling']:.3f}"
        )
        shap.image_plot(
            shap_values,
            pixel_values=batch_rgb,
            labels=np.array([[f"{case_name} | {context.label_names[output_index]}"]]),
        )
