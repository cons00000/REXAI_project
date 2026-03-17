import pandas as pd
from pandas.api.types import is_numeric_dtype
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

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
        target_cols = cols if cols else self.df.columns
        n_cols = len(target_cols)
        n_rows = (n_cols + cols_per_row - 1) // cols_per_row
        
        plt.figure(figsize=(16, 4 * n_rows)) 

        for i, col in enumerate(target_cols, 1):
            ax = plt.subplot(n_rows, cols_per_row, i)
            
            # 1. Handle One-Hot Encoded / Boolean / Low Cardinality
            if self.df[col].dtype == 'bool' or self.df[col].nunique() == 2:
                # FIX: Assign x to hue and set legend=False
                sns.countplot(x=self.df[col], hue=self.df[col], palette="Blues_r", legend=False)
                plt.title(f"Flag: {col}")
            
            # 2. Handle Continuous Numeric data
            elif np.issubdtype(self.df[col].dtype, np.number):
                sns.histplot(self.df[col], kde=True, color="teal")
                plt.title(f"Numeric: {col}")
            
            # 3. Handle Categorical data
            else:
                counts = self.df[col].value_counts().iloc[:10]
                # FIX: Assign y to hue and set legend=False
                sns.barplot(y=counts.index, x=counts.values, hue=counts.index, palette="viridis", legend=False)
                plt.title(f"Categorical: {col}")

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