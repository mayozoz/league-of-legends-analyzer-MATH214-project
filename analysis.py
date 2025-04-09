import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the champion data
def load_champion_data(file_path='champion_stats.csv'):
    """Load champion stats from CSV file"""
    return pd.read_csv(file_path, index_col=0)

def create_level_scaling_matrix(champion_df):
    """
    Create a transformation matrix that scales champion stats from level 1 to level 18
    
    This demonstrates how linear transformations can be used to model champion growth
    """
    # Extract base stats and per-level stats
    base_stats = champion_df[['hp', 'mp', 'armor', 'spellblock', 'attackdamage']].copy()
    per_level_stats = champion_df[['hpperlevel', 'mpperlevel', 'armorperlevel', 
                                  'spellblockperlevel', 'attackdamageperlevel']].copy()
    
    # Create a dictionary to store level matrices for each champion
    champion_level_matrices = {}
    
    for champion in champion_df.index:
        # Create a diagonal matrix with the per-level scaling values
        scaling_values = per_level_stats.loc[champion].values
        scaling_matrix = np.diag(scaling_values)
        
        # Store in dictionary
        champion_level_matrices[champion] = scaling_matrix
    
    return base_stats, champion_level_matrices

def calculate_stats_at_level(base_stats, champion_level_matrices, champion_name, level):
    """Calculate a champion's stats at a given level using matrix transformations"""
    if champion_name not in base_stats.index:
        raise ValueError(f"Champion {champion_name} not found")
    
    # Get base stats as a vector
    base_stat_vector = base_stats.loc[champion_name].values
    
    # Get level scaling matrix
    level_matrix = champion_level_matrices[champion_name]
    
    # Calculate level scaling: base_stats + (level-1) * per_level_stats
    level_scaling = (level - 1) * np.matmul(level_matrix, np.ones(level_matrix.shape[0]))
    
    # Add to base stats
    stats_at_level = base_stat_vector + level_scaling
    
    return pd.Series(stats_at_level, index=base_stats.columns)

def create_item_effect_matrix(item_stats):
    """
    Create a transformation matrix for item effects
    
    Parameters:
    -----------
    item_stats : dict
        Dictionary with item names as keys and stat effects as values
        
    Returns:
    --------
    item_matrix : numpy.ndarray
        Matrix representation of item effects
    """
    # Define stat columns in the same order as our champion data
    stat_columns = ['hp', 'mp', 'armor', 'spellblock', 'attackdamage']
    n_stats = len(stat_columns)
    
    # Create a matrix for each item (each row is an item, each column is a stat)
    item_matrix = np.zeros((len(item_stats), n_stats))
    
    for i, (item_name, effects) in enumerate(item_stats.items()):
        for j, stat in enumerate(stat_columns):
            if stat in effects:
                item_matrix[i, j] = effects[stat]
    
    return item_matrix, list(item_stats.keys()), stat_columns

def apply_items_to_champion(champion_stats, item_matrix, items_to_apply):
    """
    Apply item effects to champion stats using matrix operations
    
    Parameters:
    -----------
    champion_stats : numpy.ndarray
        Vector of champion stats
    item_matrix : numpy.ndarray
        Matrix of item effects
    items_to_apply : list
        List of item indices to apply
        
    Returns:
    --------
    new_stats : numpy.ndarray
        Vector of champion stats after applying items
    """
    # Create a vector to represent which items to apply
    item_vector = np.zeros(item_matrix.shape[0])
    for idx in items_to_apply:
        item_vector[idx] = 1
    
    # Apply items using matrix multiplication
    item_effects = np.matmul(item_vector, item_matrix)
    
    # Add effects to champion stats
    new_stats = champion_stats + item_effects
    
    return new_stats

def perform_champion_clustering(champion_df, n_clusters=5):
    """
    Cluster champions based on their stats using KMeans
    
    This demonstrates how linear algebra can be used to identify similar champions
    """
    # Select numerical features for clustering
    features = ['hp', 'mp', 'armor', 'spellblock', 'attackdamage', 
                'attackrange', 'movespeed', 'hpregen', 'mpregen']
    X = champion_df[features].copy()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to the DataFrame
    champion_df_with_clusters = champion_df.copy()
    champion_df_with_clusters['cluster'] = clusters
    
    return champion_df_with_clusters, kmeans.cluster_centers_

def perform_pca_analysis(champion_df):
    """
    Apply Principal Component Analysis to reduce dimensionality
    
    This demonstrates how we can use linear algebra to extract key patterns
    """
    # Select numerical features for PCA
    features = ['hp', 'mp', 'armor', 'spellblock', 'attackdamage', 
                'attackrange', 'movespeed', 'hpregen', 'mpregen']
    X = champion_df[features].copy()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA()
    pca_result = pca.fit_transform(X_scaled)
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(
        data=pca_result[:, :2],
        columns=['PC1', 'PC2'],
        index=champion_df.index
    )
    
    # Get component loadings
    loadings = pd.DataFrame(
        data=pca.components_.T[:, :2],
        columns=['PC1', 'PC2'],
        index=features
    )
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    
    return pca_df, loadings, explained_variance

def calculate_champion_synergy(champion1_stats, champion2_stats):
    """
    Calculate synergy between two champions using dot product
    
    A higher dot product indicates better synergy
    """
    # Normalize the vectors
    champion1_norm = champion1_stats / np.linalg.norm(champion1_stats)
    champion2_norm = champion2_stats / np.linalg.norm(champion2_stats)
    
    # Calculate cosine similarity (dot product of normalized vectors)
    synergy = np.dot(champion1_norm, champion2_norm)
    
    return synergy

def visualize_champion_clusters(pca_df, clusters, champion_names):
    """Visualize champion clusters in 2D space using PCA"""
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    sns.scatterplot(x='PC1', y='PC2', hue=clusters, data=pca_df, palette='viridis', s=100)
    
    # Add champion names as labels
    for i, champion in enumerate(champion_names):
        plt.annotate(champion, (pca_df.iloc[i, 0], pca_df.iloc[i, 1]), fontsize=8)
    
    plt.title('Champion Clusters Based on Stats')
    plt.xlabel(f'Principal Component 1')
    plt.ylabel(f'Principal Component 2')
    plt.tight_layout()
    plt.savefig('champion_clusters.png')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load champion data
    champion_df = load_champion_data()
    
    # 1. Create level scaling matrices
    base_stats, level_matrices = create_level_scaling_matrix(champion_df)
    
    # Example: Calculate Ahri's stats at level 18
    ahri_level18 = calculate_stats_at_level(base_stats, level_matrices, 'Ahri', 18)
    print("Ahri's stats at level 18:")
    print(ahri_level18)
    
    # 2. Define some example items
    items = {
        'Infinity Edge': {'attackdamage': 70, 'crit': 20},
        'Rabadon\'s Deathcap': {'ap': 120},
        'Warmog\'s Armor': {'hp': 800, 'hpregen': 5},
        'Spirit Visage': {'hp': 450, 'spellblock': 40, 'hpregen': 10},
        'Thornmail': {'armor': 80}
    }
    
    # Create item effect matrix
    item_matrix, item_names, stat_names = create_item_effect_matrix(items)
    
    # 3. Perform PCA analysis
    pca_results, loadings, variance = perform_pca_analysis(champion_df)
    print("\nPCA Explained Variance:")
    for i, var in enumerate(variance):
        print(f"PC{i+1}: {var:.4f}")
    
    # 4. Perform champion clustering
    champion_clusters, cluster_centers = perform_champion_clustering(champion_df)
    
    # Visualize champion clusters
    visualize_champion_clusters(pca_results, champion_clusters['cluster'], champion_clusters.index)
    
    # 5. Example: Calculate synergy between champions
    # Get vectors for two champions
    champ1 = 'Ashe'
    champ2 = 'Leona'
    
    # Use only numeric columns for synergy calculation
    numeric_cols = ['hp', 'mp', 'armor', 'spellblock', 'attackdamage', 
                   'attackrange', 'movespeed', 'hpregen', 'mpregen']
    
    if champ1 in champion_df.index and champ2 in champion_df.index:
        champ1_stats = champion_df.loc[champ1, numeric_cols].values
        champ2_stats = champion_df.loc[champ2, numeric_cols].values
        
        synergy = calculate_champion_synergy(champ1_stats, champ2_stats)
        print(f"\nSynergy between {champ1} and {champ2}: {synergy:.4f}")