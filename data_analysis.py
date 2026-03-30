import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from scipy.special import expit  

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

        # Calculate correlations
        bool_cols = self.df.select_dtypes(include=[bool]).columns
        numeric_df = self.df.select_dtypes(include=[np.number]).copy()

        for col in bool_cols:
            if col not in numeric_df.columns:
                numeric_df[col] = self.df[col].astype(int)

        correlations = numeric_df.corr()[target_column].sort_values(ascending=False)
        
        # Remove the target's correlation with itself
        correlations = correlations.drop(target_column)

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

        corr_matrix = self.df.corr(numeric_only=True)

        mask = corr_matrix.abs() >= threshold
        for i in range(len(mask)):
            mask.iat[i, i] = False

        cols_to_keep = mask.any(axis=1)
        corr_filtered = corr_matrix.loc[cols_to_keep, cols_to_keep]

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

# Fonctions présentées dans la consigne
def demographic_parity(df, Y, S):
    total = df.shape[0]
    p_y1_s1 = len(df[(df[Y]==1) & (df[S]==1)]) / total
    p_y1_s_1 = len(df[(df[Y]==1) & (df[S]==-1)]) / total
    
    return p_y1_s1 - p_y1_s_1

def disparate_impact(df,Y,S):
    total = df.shape[0]
    p_y1_s1 = len(df[(df[Y]==1) & (df[S]==1)]) / total
    p_y1_s_1 = len(df[(df[Y]==1) & (df[S]==-1)]) / total
    
    if p_y1_s_1 !=0:
        return p_y1_s1 / p_y1_s_1 
    else : 
        return "p_y1_s_1 vaut 0"
    
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
class ImageDataset(Dataset):
    def __init__(self, table: pd.DataFrame, path_image: str, transform):
        self.records = [                                          # éviter d'utiliser iloc sur un dataframe (opération bien plus longue)
            row for _, row in table.iterrows()
            if os.path.exists(f"{path_image}/{row['image_id']}")
        ]
        self.path_image = path_image
        self.transform  = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row  = self.records[idx]
        path = f"{self.path_image}/{row['image_id']}"
        img    = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        return tensor, int(row["Smiling"])
    
@torch.no_grad()
def extract_embeddings(table, path_image, backbone, transform, device,
                       batch_size=64, num_workers=4):
    dataset = ImageDataset(table, path_image, transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=(device == "cuda"))

    all_embs, all_targets = [], []
    for tensors, targets in loader:
        all_embs.append(backbone(tensors.to(device)).cpu().numpy())
        all_targets.extend(targets.numpy())

    return np.vstack(all_embs), np.array(all_targets)

def load_or_compute_embeddings(table, path_image, backbone, transform, device,
                                cache_path="cache/embeddings.npz", **kwargs):
    if Path(cache_path).exists():
        data = np.load(cache_path)
        return data["X"], data["y"]

    X, y = extract_embeddings(table, path_image, backbone, transform, device, **kwargs)

    Path(cache_path).parent.mkdir(exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y)
    return X, y

def fairness_report(model, X_test, y_true, sensitive_df):
    """
    model        : tout objet avec une méthode .predict()
    X_test       : features de test
    y_true       : array-like des vrais labels (-1/1)
    sensitive_df : DataFrame dont chaque colonne est un attribut sensible (-1/1)
    """
    y_pred = model.predict(X_test)
    y_true = np.array(y_true)
    rows = []

    for attr in sensitive_df.columns:
        attr_vals = np.array(sensitive_df[attr])
        metrics = {}

        for val in [-1, 1]:
            mask = attr_vals == val
            yt, yp = y_true[mask], y_pred[mask]
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[-1, 1]).ravel()
            metrics[val] = {
                "Group":    f"{attr}={'+' if val == 1 else ''}{val}",
                "N":        int(mask.sum()),
                "Accuracy": accuracy_score(yt, yp),
                "FPR":      fp / (fp + tn) if (fp + tn) else np.nan,
                "FNR":      fn / (fn + tp) if (fn + tp) else np.nan,
            }

        for val in [-1, 1]:
            other = metrics[-val]
            metrics[val]["Delta_Acc"] = abs(metrics[val]["Accuracy"] - other["Accuracy"])
            metrics[val]["Delta_FPR"] = abs(metrics[val]["FPR"]      - other["FPR"])
            metrics[val]["Delta_FNR"] = abs(metrics[val]["FNR"]      - other["FNR"])
            rows.append(metrics[val])

    return pd.DataFrame(rows).set_index("Group").round(3)

def predict(images_np, backbone, model, transform, device):
    """
    images_np : np.array (N, H, W, C)
    backbone : PyTorch model
    model    : sklearn pipeline/classifier
    """
    backbone.eval()  

    imgs = [transform(Image.fromarray(img.astype("uint8")).convert("RGB")) for img in images_np]
    batch = torch.stack(imgs).to(device)

    with torch.no_grad():
        embeddings = backbone(batch).cpu().numpy()  

    outputs = model.predict(embeddings) 

    return outputs

def predict_lime(images_np, backbone, pipe, transform, device): # on simule une fonction de prédiction pour LIME qui retourne des probabilités
    backbone.eval()
    imgs = [transform(Image.fromarray(img.astype("uint8")).convert("RGB")) for img in images_np]
    batch = torch.stack(imgs).to(device)
    
    with torch.no_grad():
        embeddings = backbone(batch).cpu().numpy()
    
    scores = pipe.decision_function(embeddings)  # shape (N,)
    probs = np.vstack([1 - expit(scores), expit(scores)]).T  # shape (N,2)
    return probs