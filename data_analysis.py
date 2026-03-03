import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class Analyzer:
    def __init__(self, file_path, sep=';', encoding='utf-8'):
        self.df = pd.read_csv(file_path, sep=sep, encoding=encoding)
        print(f"Loaded dataset with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")

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
            
            # Identify range or modalities
            if np.issubdtype(dtype, np.number):
                val_range = f"[{col_data.min()}, {col_data.max()}]"
                modalities = "N/A (Numeric)"
            else:
                val_range = "N/A (Categorical)"
                # Show up to 5 unique examples
                modalities = col_data.unique()[:5].tolist()

            inventory.append({
                "Column": col,
                "Type": dtype,
                "Cardinality": nunique,
                "Missing": nulls,
                "Range/Examples": val_range if "Numeric" in str(val_range) else modalities
            })
        
        inventory_df = pd.DataFrame(inventory)
        return inventory_df

    def plot_distributions(self, cols=None):
        """Plots histograms for numerical or countplots for categorical columns."""
        target_cols = cols if cols else self.df.columns[:6] # Limit to 6 by default
        
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(target_cols, 1):
            plt.subplot(int(len(target_cols)/2) + 1, 2, i)
            if np.issubdtype(self.df[col].dtype, np.number):
                sns.histplot(self.df[col], kde=True, color="teal")
            else:
                # Top 10 most frequent categories to avoid clutter
                sns.countplot(y=self.df[col], order=self.df[col].value_counts().iloc[:10].index)
            plt.title(f"Distribution: {col}")
        
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
        correlations = self.df.select_dtypes(include=[np.number]).corr()[target_column].sort_values(ascending=False)
        
        # Remove the target's correlation with itself
        correlations = correlations.drop(target_column)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=correlations.values, y=correlations.index, palette="RdBu_r")
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        plt.title(f"Correlation of Features with '{target_column}'")
        plt.xlabel("Pearson Correlation Coefficient")
        plt.show()

        return correlations