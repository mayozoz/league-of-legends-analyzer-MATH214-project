import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# Load environment variables (for API key)
load_dotenv()

# Constants
VERSION = "14.7.1"  # Current patch version, update as needed
RIOT_API_KEY = os.getenv("RIOT_API_KEY")  # Your Riot API key stored in .env file

# Function to fetch champion data from Riot's Data Dragon API
def fetch_champion_data():
    url = f"http://ddragon.leagueoflegends.com/cdn/{VERSION}/data/en_US/champion.json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["data"]
    else:
        print(f"Error fetching champion data: {response.status_code}")
        return None

# Function to get detailed champion info
def get_champion_details(champion_id):
    url = f"http://ddragon.leagueoflegends.com/cdn/{VERSION}/data/en_US/champion/{champion_id}.json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["data"][champion_id]
    else:
        print(f"Error fetching details for {champion_id}: {response.status_code}")
        return None

# Create a matrix of champion base stats
def create_champion_stat_matrix(champions_data):
    # Define the stats we want to track
    stats = ['hp', 'hpperlevel', 'mp', 'mpperlevel', 'movespeed', 'armor',
             'armorperlevel', 'spellblock', 'spellblockperlevel', 'attackrange',
             'hpregen', 'hpregenperlevel', 'mpregen', 'mpregenperlevel',
             'crit', 'critperlevel', 'attackdamage', 'attackdamageperlevel',
             'attackspeed', 'attackspeedperlevel']
    
    # Initialize a DataFrame to store the stats
    champion_stats = []
    champion_names = []
    
    # Extract stats for each champion
    for champ_id, champ_data in champions_data.items():
        details = get_champion_details(champ_id)
        if details:
            stat_values = []
            for stat in stats:
                if stat in details['stats']:
                    stat_values.append(details['stats'][stat])
                else:
                    stat_values.append(0)  # Handle missing stats
            
            champion_stats.append(stat_values)
            champion_names.append(champ_data['name'])
    
    # Create DataFrame
    df = pd.DataFrame(champion_stats, index=champion_names, columns=stats)
    return df

# Function to visualize champion stats
def visualize_champion_stats(stat_matrix, stat_name):
    plt.figure(figsize=(12, 8))
    stat_matrix[stat_name].sort_values(ascending=False).head(15).plot(kind='bar')
    plt.title(f'Top 15 Champions by {stat_name}')
    plt.ylabel(stat_name)
    plt.xlabel('Champion')
    plt.tight_layout()
    plt.savefig(f'{stat_name}_comparison.png')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Step 1: Fetch champion data
    print("Fetching champion data...")
    champions = fetch_champion_data()
    if not champions:
        print("Failed to fetch champion data. Exiting.")
        exit()
    
    print(f"Retrieved data for {len(champions)} champions.")
    
    # Step 2: Create champion stat matrix
    print("Creating champion stat matrix...")
    champion_matrix = create_champion_stat_matrix(champions)
    
    # Step 3: Save the data to CSV for further analysis
    champion_matrix.to_csv('champion_stats.csv')
    print("Champion stats saved to champion_stats.csv")
    
    # Step 4: Visualize some stats
    print("Generating visualizations...")
    visualize_champion_stats(champion_matrix, 'hp')
    visualize_champion_stats(champion_matrix, 'attackdamage')
    
    # Step 5: Basic linear algebra operations
    # Normalize the stats for comparison (convert to z-scores)
    normalized_stats = (champion_matrix - champion_matrix.mean()) / champion_matrix.std()
    normalized_stats.to_csv('normalized_champion_stats.csv')
    print("Normalized stats saved to normalized_champion_stats.csv")
    
    print("Initial data collection and preprocessing complete!")
    