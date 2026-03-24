import pandas as pd
from pandas.api.types import is_numeric_dtype
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as mtick

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
                val_range = f"Examples: {col_data.unique()[:3].tolist()}"

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

def plot_fairness_attribute(fairness_by_group, attribute, title_suffix=""):
    """Bar charts FPR / FNR / DI par sous-groupe, un subplot-row par modèle."""
    subset = fairness_by_group[fairness_by_group["attribute"] == attribute].copy()
    model_names = subset["model"].unique()
    n_models = len(model_names)

    fig, axes = plt.subplots(n_models, 3, figsize=(17, 4.5 * n_models), sharey=False)
    if n_models == 1:
        axes = [axes]

    fig.suptitle(f"FPR / FNR / Disparate Impact — {title_suffix}",
                 fontsize=13, fontweight="bold", y=1.01)

    for row_idx, model_name in enumerate(model_names):
        m = subset[subset["model"] == model_name].sort_values("group").reset_index(drop=True)
        groups = m["group"].astype(str).tolist()
        x = list(range(len(groups)))

        def annotate(ax, bars, values, fmt=".2%"):
            for bar, val in zip(bars, values):
                if pd.notna(val):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.003,
                            f"{val:{fmt}}", ha="center", va="bottom", fontsize=7.5)

        ax = axes[row_idx][0]
        bars = ax.bar(x, m["fpr"], color="#4472C4", edgecolor="white")
        ax.axhline(m["fpr"].mean(), color="crimson", linestyle="--", lw=1.2, label="moyenne")
        ax.set_title(f"{model_name} — FPR", fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(groups, rotation=35, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
        ax.set_ylabel("False Positive Rate"); ax.legend(fontsize=8)
        annotate(ax, bars, m["fpr"])

        ax = axes[row_idx][1]
        bars = ax.bar(x, m["fnr"], color="#ED7D31", edgecolor="white")
        ax.axhline(m["fnr"].mean(), color="crimson", linestyle="--", lw=1.2, label="moyenne")
        ax.set_title(f"{model_name} — FNR", fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(groups, rotation=35, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
        ax.set_ylabel("False Negative Rate"); ax.legend(fontsize=8)
        annotate(ax, bars, m["fnr"])

        ax = axes[row_idx][2]
        max_rate = m["selection_rate"].max()
        di = (m["selection_rate"] / max_rate) if max_rate > 0 else m["selection_rate"]
        colors = ["#C00000" if v < 0.80 else "#70AD47" for v in di.fillna(0)]
        bars = ax.bar(x, di, color=colors, edgecolor="white")
        ax.axhline(0.80, color="crimson", linestyle="--", lw=1.2, label="seuil 80%")
        ax.set_title(f"{model_name} — DI (vs groupe le + prédit)", fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(groups, rotation=35, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylabel("Disparate Impact"); ax.legend(fontsize=8)
        annotate(ax, bars, di, fmt=".2f")

    plt.tight_layout()
    plt.show()


def print_fairness_table(fairness_by_group, attribute):
    cols = ["model", "group", "n", "prevalence", "selection_rate", "fpr", "fnr"]
    sub = (fairness_by_group[fairness_by_group["attribute"] == attribute][cols]
           .sort_values(["model", "group"]).copy())
    for col in ["prevalence", "selection_rate", "fpr", "fnr"]:
        sub[col] = sub[col].map(lambda v: f"{v:.2%}" if pd.notna(v) else "—")
    print(sub.to_string(index=False))