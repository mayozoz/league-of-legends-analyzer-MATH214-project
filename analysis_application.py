import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from champion_data_collector import fetch_champion_data, create_champion_stat_matrix
from analysis import (
    perform_pca_analysis,
    perform_champion_clustering,
    create_level_scaling_matrix,
    calculate_stats_at_level,
    calculate_champion_synergy,
    visualize_champion_clusters
)

def main():
    """Main application to analyze League of Legends champions using linear algebra"""
    print("=== League of Legends Champion Analysis ===")
    print("Collecting champion data...")
    
    # Check if data already exists
    try:
        champion_df = pd.read_csv('champion_stats.csv', index_col=0)
        print(f"Loaded data for {len(champion_df)} champions")
    except FileNotFoundError:
        # If not, collect it
        print("Champion data not found. Collecting from API...")
        champions = fetch_champion_data()
        champion_df = create_champion_stat_matrix(champions)
        champion_df.to_csv('champion_stats.csv')
        print(f"Collected data for {len(champion_df)} champions")
    
    while True:
        print("\nWhat would you like to analyze?")
        print("1. View champion statistics")
        print("2. Analyze champion scaling (level 1 to 18)")
        print("3. Identify champion clusters")
        print("4. Perform PCA analysis on champion stats")
        print("5. Calculate champion synergies")
        print("6. Exit")
        
        choice = input("Enter your choice (1-6): ")
        
        if choice == '1':
            analyze_champion_stats(champion_df)
        elif choice == '2':
            analyze_champion_scaling(champion_df)
        elif choice == '3':
            identify_champion_clusters(champion_df)
        elif choice == '4':
            perform_pca(champion_df)
        elif choice == '5':
            calculate_synergies(champion_df)
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

def analyze_champion_stats(champion_df):
    """Analyze basic champion statistics"""
    # Get top champions by different stats
    stats_to_analyze = ['hp', 'mp', 'armor', 'spellblock', 'attackdamage', 'movespeed']
    
    print("\n--- Top Champions by Stat ---")
    for stat in stats_to_analyze:
        top_champions = champion_df[stat].sort_values(ascending=False).head(5)
        print(f"\nTop 5 champions by {stat}:")
        for champion, value in top_champions.items():
            print(f"  {champion}: {value:.1f}")
    
    # Correlation matrix between stats
    print("\n--- Correlation Matrix ---")
    corr_matrix = champion_df[stats_to_analyze].corr()
    
    # Print most significant correlations
    print("Strongest stat correlations:")
    corr_pairs = []
    for i in range(len(stats_to_analyze)):
        for j in range(i+1, len(stats_to_analyze)):
            stat1 = stats_to_analyze[i]
            stat2 = stats_to_analyze[j]
            correlation = corr_matrix.loc[stat1, stat2]
            corr_pairs.append((stat1, stat2, correlation))
    
    # Sort by absolute correlation
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for stat1, stat2, corr in corr_pairs[:5]:
        print(f"  {stat1} and {stat2}: {corr:.3f}")
    
    # Visualize correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Champion Stats')
    plt.savefig('stat_correlations.png')
    plt.close()
    print("Correlation matrix visualization saved to 'stat_correlations.png'")

def analyze_champion_scaling(champion_df):
    """Analyze how champions scale from level 1 to 18"""
    base_stats, level_matrices = create_level_scaling_matrix(champion_df)
    
    # Let user select a champion
    print("\n--- Champion Scaling Analysis ---")
    print("Available champions:")
    
    # Display champions in rows of 5
    champions = list(champion_df.index)
    for i in range(0, len(champions), 5):
        print(', '.join(champions[i:i+5]))
    
    champion_name = input("\nEnter champion name: ")
    
    if champion_name not in champion_df.index:
        print(f"Champion '{champion_name}' not found.")
        return
    
    # Calculate stats at different levels
    level_stats = []
    levels = [1, 6, 11, 16, 18]
    
    for level in levels:
        stats = calculate_stats_at_level(base_stats, level_matrices, champion_name, level)
        level_stats.append(stats)
    
    # Create DataFrame for display
    level_df = pd.DataFrame(level_stats, index=[f"Level {l}" for l in levels])
    
    print(f"\n{champion_name}'s stats at different levels:")
    print(level_df.round(1))
    
    # Visualize scaling
    plt.figure(figsize=(12, 6))
    
    for stat in level_df.columns:
        plt.plot(levels, level_df[stat], marker='o', label=stat)
    
    plt.title(f"{champion_name}'s Stat Scaling")
    plt.xlabel('Level')
    plt.ylabel('Stat Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{champion_name}_scaling.png")
    plt.close()
    
    print(f"Scaling visualization saved to '{champion_name}_scaling.png'")

def identify_champion_clusters(champion_df):
    """Identify and visualize champion clusters"""
    print("\n--- Champion Clustering ---")
    n_clusters = int(input("Enter number of clusters (2-10): "))
    
    if n_clusters < 2 or n_clusters > 10:
        print("Invalid number of clusters. Using 5 as default.")
        n_clusters = 5
    
    # Perform clustering and PCA
    champion_clusters, _ = perform_champion_clustering(champion_df, n_clusters)
    pca_results, _, _ = perform_pca_analysis(champion_df)
    
    # Combine PCA results with cluster information
    pca_with_clusters = pca_results.copy()
    pca_with_clusters['cluster'] = champion_clusters['cluster']
    
    # Visualize clusters
    visualize_champion_clusters(pca_with_clusters, pca_with_clusters['cluster'], champion_df.index)
    
    # Show champions in each cluster
    for cluster in range(n_clusters):
        champions_in_cluster = champion_clusters[champion_clusters['cluster'] == cluster].index.tolist()
        print(f"\nCluster {cluster + 1}:")
        print(', '.join(champions_in_cluster))

def perform_pca(champion_df):
    """Perform PCA analysis on champion stats"""
    print("\n--- Principal Component Analysis ---")
    
    # Perform PCA
    pca_results, loadings, variance = perform_pca_analysis(champion_df)
    
    # Print explained variance
    print("Explained variance by principal components:")
    for i, var in enumerate(variance[:5]):
        print(f"  PC{i+1}: {var:.4f} ({var*100:.1f}%)")
    
    # Print feature contributions to first two PCs
    print("\nFeature contributions to principal components:")
    print(loadings)
    
    # Visualize PCA results
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    sns.scatterplot(x='PC1', y='PC2', data=pca_results, s=100)
    
    # Add champion names as labels
    for i, champion in enumerate(pca_results.index):
        plt.annotate(champion, (pca_results.iloc[i, 0], pca_results.iloc[i, 1]), fontsize=8)
    
    plt.title('Champion PCA Analysis')
    plt.xlabel(f'Principal Component 1 ({variance[0]*100:.1f}%)')
    plt.ylabel(f'Principal Component 2 ({variance[1]*100:.1f}%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('champion_pca.png')
    plt.close()
    
    print("PCA visualization saved to 'champion_pca.png'")
    
    # Create biplot (PCA with feature vectors)
    plt.figure(figsize=(12, 10))
    
    # Plot champions
    plt.scatter(pca_results['PC1'], pca_results['PC2'], alpha=0.7)
    
    # Plot feature vectors
    for i, feature in enumerate(loadings.index):
        plt.arrow(0, 0, loadings.iloc[i, 0] * 5, loadings.iloc[i, 1] * 5, 
                 head_width=0.2, head_length=0.2, fc='red', ec='red')
        plt.text(loadings.iloc[i, 0] * 5.2, loadings.iloc[i, 1] * 5.2, feature, color='red')
    
    plt.title('PCA Biplot: Champions and Stat Vectors')
    plt.xlabel(f'Principal Component 1 ({variance[0]*100:.1f}%)')
    plt.ylabel(f'Principal Component 2 ({variance[1]*100:.1f}%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('champion_pca_biplot.png')
    plt.close()
    
    print("PCA biplot saved to 'champion_pca_biplot.png'")

def calculate_synergies(champion_df):
    """Calculate synergies between champions"""
    print("\n--- Champion Synergy Analysis ---")
    
    # Let user select two champions
    print("Available champions:")
    
    # Display champions in rows of 5
    champions = list(champion_df.index)
    for i in range(0, len(champions), 5):
        print(', '.join(champions[i:i+5]))
    
    champion1 = input("\nEnter first champion: ")
    champion2 = input("Enter second champion: ")
    
    if champion1 not in champion_df.index or champion2 not in champion_df.index:
        print("One or both champions not found.")
        return
    
    # Use only numeric columns for synergy calculation
    numeric_cols = ['hp', 'mp', 'armor', 'spellblock', 'attackdamage', 
                   'attackrange', 'movespeed', 'hpregen', 'mpregen']
    
    champ1_stats = champion_df.loc[champion1, numeric_cols].values
    champ2_stats = champion_df.loc[champion2, numeric_cols].values
    
    synergy = calculate_champion_synergy(champ1_stats, champ2_stats)
    print(f"\nSynergy between {champion1} and {champion2}: {synergy:.4f}")
    
    # Calculate synergy with all champions
    all_synergies = {}
    for champion in champion_df.index:
        if champion != champion1:
            champ_stats = champion_df.loc[champion, numeric_cols].values
            synergy = calculate_champion_synergy(champ1_stats, champ_stats)
            all_synergies[champion] = synergy
    
    # Show top 5 synergies
    top_synergies = sorted(all_synergies.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nTop 5 champions with highest synergy with {champion1}:")
    for champion, synergy in top_synergies:
        print(f"  {champion}: {synergy:.4f}")

if __name__ == "__main__":
    main()